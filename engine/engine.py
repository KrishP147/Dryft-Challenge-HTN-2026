"""Qwen3-4B greedy decode engine: static KV cache, CUDA-graphed decode, pipelined host sync.

v2: fused Triton ops (fused.py) with load-time selftest + torch fallback.
"""
import glob
import json
import math
import os

import torch
import torch.nn.functional as F

try:
    from fused import TritonOps
except Exception as _e:  # no triton / import failure: torch ops only
    TritonOps = None
    _FUSED_ERR = repr(_e)

CAP_GRAN = 128
MAX_STATES = 6
ROPE_LEN = 8192


def _rms(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


def _build_rope_tables(theta, hd, n, dev, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    fr = torch.outer(torch.arange(n, device=dev).float(), inv)
    emb = torch.cat((fr, fr), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class _TorchOps:
    """Reference-arithmetic ops; also the fallback when fused ops fail."""

    def __init__(self, e):
        self.e = e

    def linear(self, x, w):
        return F.linear(x, w)

    def rms(self, h, w):
        return _rms(h, w, self.e.eps)

    def add_rms(self, h, d, w):
        h = h + d
        return h, _rms(h, w, self.e.eps)

    def silu_mul(self, gu):
        g, u = gu.chunk(2, -1)
        return F.silu(g) * u

    def attn_decode(self, q, kc, vc, pos):
        e = self.e
        B, cap = q.shape[0], kc.shape[2]
        mask = (
            torch.zeros(cap, dtype=q.dtype, device=q.device)
            .masked_fill_(torch.arange(cap, device=q.device) > pos, float("-inf"))
            .view(1, 1, 1, cap)
        )
        return F.scaled_dot_product_attention(
            q.reshape(B, e.nkv, e.nh // e.nkv, e.hd), kc, vc, attn_mask=mask
        ).reshape(B, e.nh * e.hd)

    def qkv_post(self, qkv, l, kc, vc, B, S, pos):
        e = self.e
        nh, nkv, hd = e.nh, e.nkv, e.hd
        q, k, v = qkv.split([nh * hd, nkv * hd, nkv * hd], -1)
        q = _rms(q.reshape(B, S, nh, hd), l.qn, e.eps).transpose(1, 2)
        k = _rms(k.reshape(B, S, nkv, hd), l.kn, e.eps).transpose(1, 2)
        v = v.reshape(B, S, nkv, hd).transpose(1, 2)
        if pos is None:
            cos, sin = e.cos[:S], e.sin[:S]
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            kc[:, :, :S] = k
            vc[:, :, :S] = v
        else:
            cos, sin = e.cos.index_select(0, pos), e.sin.index_select(0, pos)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            kc.index_copy_(2, pos, k)
            vc.index_copy_(2, pos, v)
        return q


class _Layer:
    pass


class _State:
    pass


class Engine:
    def __init__(self, model_path, dtype=torch.bfloat16):
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
        self._dbg = None
        self.ops = _TorchOps(self)
        self.pool = torch.cuda.graph_pool_handle() if self.dev.type == "cuda" else None
        if self.dev.type == "cuda":
            self._selftest()

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
        self.cos, self.sin = _build_rope_tables(self.theta, self.hd, n, self.dev, self.dtype)

    # ------------------------------------------------------------------ state
    def _state(self, B, cap, graph=True):
        key = (B, cap)
        st = self.states.get(key)
        if st is not None and (st.tried_graph or not graph):
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
        st.tried_graph = graph
        st.hcap = 0
        self.states[key] = st
        if graph and self.dev.type == "cuda":
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

    # --------------------------------------------------------------- forwards
    def _forward(self, st, tokens, S, pos):
        """One pass over B*S tokens. pos=None: prefill from position 0 (logits
        for each sequence's last token only). pos=tensor: one decode step."""
        B = st.B
        T = B * S
        nh, nkv, hd, ops = self.nh, self.nkv, self.hd, self.ops
        decode = pos is not None
        h = self.embed[tokens]
        a = ops.rms(h, self.layers[0].ln1)
        last = self.L - 1
        for i, l in enumerate(self.layers):
            kc, vc = st.kc[i], st.vc[i]
            q = ops.qkv_post(ops.linear(a, l.wqkv), l, kc, vc, B, S, pos)
            if decode:
                o = ops.attn_decode(q, kc, vc, pos)
            elif i == last:
                # The final layer only contributes the last prompt position's
                # logits. Its last query can attend to every cached key.
                o = F.scaled_dot_product_attention(
                    q[:, :, -1:, :], kc[:, :, :S], vc[:, :, :S], enable_gqa=True
                ).transpose(1, 2).reshape(B, nh * hd)
                h = h[S - 1::S].contiguous()
            else:
                o = F.scaled_dot_product_attention(
                    q, kc[:, :, :S], vc[:, :, :S], is_causal=True, enable_gqa=True
                ).transpose(1, 2).reshape(T, nh * hd)
            h, a = ops.add_rms(h, ops.linear(o, l.wo), l.ln2)
            m = ops.silu_mul(ops.linear(a, l.wgu))
            nxt = self.layers[i + 1].ln1 if i < last else self.norm
            h, a = ops.add_rms(h, ops.linear(m, l.wd), nxt)
        logits = ops.linear(a, self.lm_head)
        if self._dbg is not None:
            self._dbg.append(logits.float().cpu())
        return logits.argmax(-1)

    def _prefill(self, ids, st):
        st.tok.copy_(self._forward(st, ids.reshape(-1), ids.shape[1], None))

    def _decode_body(self, st):
        st.tok.copy_(self._forward(st, st.tok, 1, st.pos))
        st.pos.add_(1)

    def _selftest(self):
        """Fused ops vs torch ops on the real weights; keep fused only if close."""
        if TritonOps is None:
            print(f"[engine] fused ops unavailable: {_FUSED_ERR}")
            return
        torch.manual_seed(0)
        B, S, n = 2, 96, 4
        high = min(5000, self.embed.shape[0])
        ids = torch.randint(min(100, high - 1), high, (B, S), device=self.dev)

        def run(ops, reference_tokens=None):
            self.ops = ops
            try:
                st = self._state(B, 128, graph=False)
                self._dbg, toks = [], []
                st.pos.fill_(S)
                self._prefill(ids, st)
                for t in range(n):
                    toks.append(st.tok.clone())
                    if t < n - 1:
                        if reference_tokens is not None:
                            st.tok.copy_(reference_tokens[t])
                        self._decode_body(st)
                return self._dbg, toks
            finally:
                self._dbg = None
                self.states.clear()

        fallback = _TorchOps(self)
        try:
            ref, reference_tokens = run(fallback)
            for label, use_gemv in (("full", True), ("without GEMV", False)):
                try:
                    candidate = TritonOps(self, use_gemv=use_gemv)
                    got, _ = run(candidate, reference_tokens)
                    diffs = [(r - g).abs().max().item() for r, g in zip(ref, got)]
                    diff = max(diffs) if all(math.isfinite(d) for d in diffs) else float("inf")
                    print(f"[engine] fused {label} selftest max logit diff {diff:.4f}")
                    if diff <= 0.5:
                        self.ops = candidate
                        return
                except Exception as e:
                    print(f"[engine] fused {label} selftest error: {e!r}")
        except Exception as e:
            print(f"[engine] fused selftest error: {e!r}")
        self.ops = fallback

    # --------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens <= 0:
            return
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
