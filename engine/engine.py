"""Qwen3-4B greedy decode engine: static KV cache, CUDA-graphed decode, pipelined host sync.

v1: pure PyTorch (no Triton). CPU path verified vs HF; GPU path untested.
"""
import glob
import json
import os

import torch
import torch.nn.functional as F

CAP_GRAN = 128
MAX_STATES = 6
ROPE_LEN = 8192
WARM_SHAPES = [(1, 512, 32), (4, 2048, 32), (16, 512, 128)]


def _rms(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


class _Layer:
    pass


class _State:
    pass


class Engine:
    def __init__(self, model_path, dtype=torch.bfloat16, warmup=True):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        self.L = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or cfg["hidden_size"] // self.nh
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg["rope_theta"])
        self._load(model_path)
        self._build_rope(ROPE_LEN)
        self.states = {}
        self.pool = torch.cuda.graph_pool_handle() if self.dev.type == "cuda" else None
        if warmup and self.dev.type == "cuda":
            self._warmup()

    # ---------------------------------------------------------------- weights
    def _load(self, model_path):
        from safetensors.torch import load_file

        w = {}
        for fn in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            w.update(load_file(fn, device=str(self.dev)))
        g = lambda k: w.pop(k).to(self.dtype)
        self.embed = g("model.embed_tokens.weight")
        self.lm_head = g("lm_head.weight") if "lm_head.weight" in w else self.embed
        self.norm = g("model.norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            l = _Layer()
            l.ln1 = g(p + "input_layernorm.weight")
            l.ln2 = g(p + "post_attention_layernorm.weight")
            l.qn = g(p + "self_attn.q_norm.weight")
            l.kn = g(p + "self_attn.k_norm.weight")
            l.wqkv = torch.cat(
                [g(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0
            ).contiguous()
            l.wo = g(p + "self_attn.o_proj.weight")
            l.wgu = torch.cat(
                [g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")], 0
            ).contiguous()
            l.wd = g(p + "mlp.down_proj.weight")
            self.layers.append(l)
        del w

    def _build_rope(self, n):
        inv = 1.0 / (
            self.theta ** (torch.arange(0, self.hd, 2, device=self.dev).float() / self.hd)
        )
        fr = torch.outer(torch.arange(n, device=self.dev).float(), inv)
        emb = torch.cat((fr, fr), -1)
        self.cos = emb.cos().to(self.dtype)
        self.sin = emb.sin().to(self.dtype)

    # ------------------------------------------------------------------ state
    def _state(self, B, cap):
        key = (B, cap)
        st = self.states.get(key)
        if st is not None:
            return st
        if len(self.states) >= MAX_STATES:
            del self.states[next(iter(self.states))]
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        st = _State()
        st.B, st.cap = B, cap
        shape = (B, self.nkv, cap, self.hd)
        # zeros, not empty: masked slots must be finite (0 * NaN = NaN)
        st.kc = [torch.zeros(shape, dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        st.vc = [torch.zeros(shape, dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        st.tok = torch.zeros(B, dtype=torch.long, device=self.dev)
        st.pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        st.ar = torch.arange(cap, device=self.dev)
        st.graph = None
        st.hcap = 0
        self.states[key] = st
        if self.dev.type == "cuda":
            try:
                self._capture(st)
            except Exception as e:  # fall back to eager decode
                print(f"[engine] graph capture failed for {key}: {e!r}")
                st.graph = None
        return st

    def _host_bufs(self, st, n):
        if st.hcap < n:
            st.hcap = max(n, 256)
            pin = self.dev.type == "cuda"
            st.host = torch.empty((st.hcap, st.B), dtype=torch.long, pin_memory=pin)
            st.events = [torch.cuda.Event() for _ in range(st.hcap)] if pin else None

    def _capture(self, st):
        st.pos.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._decode_body(st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self._decode_body(st)
        st.graph = g
        torch.cuda.synchronize()

    def _warmup(self):
        for B, S, n in WARM_SHAPES:
            ids = [[(7 * i + 13 * b) % 1000 + 100 for i in range(S)] for b in range(B)]
            for _ in range(2):
                for _ in self.generate(ids, n):
                    pass
        torch.cuda.synchronize()

    # --------------------------------------------------------------- forwards
    def _prefill(self, ids, st):
        B, S = ids.shape
        nh, nkv, hd = self.nh, self.nkv, self.hd
        h = self.embed[ids]
        cos, sin = self.cos[:S], self.sin[:S]
        for i, l in enumerate(self.layers):
            a = _rms(h, l.ln1, self.eps)
            q, k, v = F.linear(a, l.wqkv).split([nh * hd, nkv * hd, nkv * hd], -1)
            q = _rms(q.view(B, S, nh, hd), l.qn, self.eps).transpose(1, 2)
            k = _rms(k.view(B, S, nkv, hd), l.kn, self.eps).transpose(1, 2)
            v = v.view(B, S, nkv, hd).transpose(1, 2)
            q = _rope(q, cos, sin)
            k = _rope(k, cos, sin)
            st.kc[i][:, :, :S] = k
            st.vc[i][:, :, :S] = v
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            h = h + F.linear(o.transpose(1, 2).reshape(B, S, nh * hd), l.wo)
            a = _rms(h, l.ln2, self.eps)
            g, u = F.linear(a, l.wgu).chunk(2, -1)
            h = h + F.linear(F.silu(g) * u, l.wd)
        h = _rms(h[:, -1], self.norm, self.eps)
        st.tok.copy_(F.linear(h, self.lm_head).argmax(-1))

    def _decode_body(self, st):
        B, cap = st.B, st.cap
        nh, nkv, hd = self.nh, self.nkv, self.hd
        h = self.embed[st.tok]
        cos = self.cos.index_select(0, st.pos)
        sin = self.sin.index_select(0, st.pos)
        mask = (
            torch.zeros(cap, dtype=self.dtype, device=self.dev)
            .masked_fill_(st.ar > st.pos, float("-inf"))
            .view(1, 1, 1, cap)
        )
        for i, l in enumerate(self.layers):
            a = _rms(h, l.ln1, self.eps)
            q, k, v = F.linear(a, l.wqkv).split([nh * hd, nkv * hd, nkv * hd], -1)
            q = _rope(_rms(q.reshape(B, nh, hd), l.qn, self.eps), cos, sin)
            k = _rope(_rms(k.reshape(B, nkv, hd), l.kn, self.eps), cos, sin)
            st.kc[i].index_copy_(2, st.pos, k.unsqueeze(2))
            st.vc[i].index_copy_(2, st.pos, v.reshape(B, nkv, 1, hd))
            # GQA as q_len=group: [B, nkv, nh//nkv, hd] attends to [B, nkv, cap, hd]
            o = F.scaled_dot_product_attention(
                q.reshape(B, nkv, nh // nkv, hd), st.kc[i], st.vc[i], attn_mask=mask
            )
            h = h + F.linear(o.reshape(B, nh * hd), l.wo)
            a = _rms(h, l.ln2, self.eps)
            g, u = F.linear(a, l.wgu).chunk(2, -1)
            h = h + F.linear(F.silu(g) * u, l.wd)
        h = _rms(h, self.norm, self.eps)
        st.tok.copy_(F.linear(h, self.lm_head).argmax(-1))
        st.pos.add_(1)

    # --------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens):
        B = len(input_ids)
        S = len(input_ids[0])
        if any(len(x) != S for x in input_ids):
            yield from self._generate_ragged(input_ids, max_new_tokens)
            return
        n = max_new_tokens
        cap = -(-(S + n) // CAP_GRAN) * CAP_GRAN
        if cap > self.cos.shape[0]:
            self.states.clear()
            self._build_rope(cap)
        st = self._state(B, cap)
        self._host_bufs(st, n)
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.dev)
        self._prefill(ids, st)
        st.pos.fill_(S)
        host, ev = st.host, st.events
        cuda = ev is not None
        host[0].copy_(st.tok, non_blocking=cuda)
        if cuda:
            ev[0].record()
        for t in range(1, n):
            if st.graph is not None:
                st.graph.replay()
            else:
                self._decode_body(st)
            host[t].copy_(st.tok, non_blocking=cuda)
            if cuda:
                ev[t].record()
                ev[t - 1].synchronize()
            yield host[t - 1].tolist()
        if cuda:
            ev[n - 1].synchronize()
        yield host[n - 1].tolist()

    def _generate_ragged(self, input_ids, n):
        outs = [[t[0] for t in self.generate([seq], n)] for seq in input_ids]
        for t in range(n):
            yield [o[t] for o in outs]
