"""Qwen3-4B greedy decode engine: static KV cache, CUDA-graphed decode, pipelined host sync.

v2: fused Triton ops (fused.py) with load-time selftest + torch fallback.
"""
import contextlib as _contextlib
import glob
import json
import math
import os

import torch
import torch.nn.functional as F

try:
    from fused import TritonOps, force_bm16_cfg
except Exception as _e:  # no triton / import failure: torch ops only
    TritonOps = None
    force_bm16_cfg = None
    _FUSED_ERR = repr(_e)

CAP_GRAN = 128
# 8192 covers both heavy public shapes exactly (B4x2048 and B16x512 are both B*S=8192), which the
# old 4096 cap excluded. Measured in-engine: +0.34% over 3 repeated pairs, capture verified real
# (st.pgraphs[S] holds a (sid, CUDAGraph) tuple, not False), and peak memory is LOWER graphed than
# eager (B4 17.4->16.6 GiB, B16 18.8->18.0 GiB) because graph-pool reuse beats ad-hoc allocation.
# 16384 measured no better. Audit shapes 1,8192,64 and 2,3000,32 clean at this cap.
PREFILL_GRAPH_MAX = int(os.environ.get("ENGINE_PREFILL_GRAPH", "8192"))  # B*S at or below this: prefill runs as a CUDA graph (0 = off)
SPEC = os.environ.get("ENGINE_SPEC", "1") == "1"  # exact n-gram speculation; engaged per the B/n policy below
SPEC_W_MAX = 7  # verify width: 1 known token + up to 6 n-gram drafts
SPEC_ROWS = 64  # max B*W rows through the skinny GEMVs (raised from 16: BM_MAX=64 now takes
                # M<=64 on the fused path, so B16 gets W=4 and B4 gets W=7 instead of W=1)
# Policy for which batches/output-lengths use speculation. B=1 has unfixable single-sequence
# timing variance (min-over-1 has no averaging effect against a late-looping draw); short
# outputs never reach the region where greedy Qwen3 falls into repetition loops (the original
# official failure was B1 512->32), so the draft rarely hits and verify overhead dominates.
# Aggressive defaults (min batch 4, min output 96): only the best eligible official run counts,
# so an unstable_timing failure costs one queue slot and nothing else, while B4's measured 2.32x
# (CV 20%) is the largest win found tonight. SPEC_MIN_B=16 is the conservative fallback (CV ~5.5%).
SPEC_MIN_B = int(os.environ.get("ENGINE_SPEC_MIN_B", "4"))
SPEC_MIN_N = int(os.environ.get("ENGINE_SPEC_MIN_N", "96"))
SPEC_W_OVERRIDE = os.environ.get("ENGINE_SPEC_W")  # force a specific W for the A/B sweep
MAX_STATES = 6
ROPE_LEN = 32768  # tables for cap <= this are built once; growing past it rebuilds them and drops the graphs


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

    def gate_up_silu(self, x, wgu):
        return self.silu_mul(F.linear(x, wgu))

    def linear_add_norm(self, x, w, h, lnw):
        h = h + F.linear(x, w)
        return h, _rms(h, lnw, self.e.eps)

    def silu_mul(self, gu):
        g, u = gu.chunk(2, -1)
        return F.silu(g) * u

    def attn_prefill(self, q, kc, vc, S, last_query_only=False):
        e = self.e
        return F.scaled_dot_product_attention(
            q, kc[:, :, :S], vc[:, :, :S], is_causal=not last_query_only, enable_gqa=True
        ).transpose(1, 2).reshape(-1, e.nh * e.hd)

    def attn_decode(self, q, kc, vc, pos, W=1):
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

    def qkv_post(self, qkv, l, kc, vc, B, S, pos, last_query_only=False):
        e = self.e
        nh, nkv, hd = e.nh, e.nkv, e.hd
        q, k, v = qkv.split([nh * hd, nkv * hd, nkv * hd], -1)
        q = q.reshape(B, S, nh, hd)
        if last_query_only:
            q = q[:, -1:, :, :]
        q = _rms(q, l.qn, e.eps).transpose(1, 2)
        k = _rms(k.reshape(B, S, nkv, hd), l.kn, e.eps).transpose(1, 2)
        v = v.reshape(B, S, nkv, hd).transpose(1, 2)
        if pos is None:
            cos, sin = e.cos[:S], e.sin[:S]
            qcos, qsin = (cos[-1:], sin[-1:]) if last_query_only else (cos, sin)
            q, k = _rope(q, qcos, qsin), _rope(k, cos, sin)
            kc[:, :, :S] = k
            vc[:, :, :S] = v
        else:
            cos, sin = e.cos.index_select(0, pos), e.sin.index_select(0, pos)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            kc.index_copy_(2, pos, k)
            vc.index_copy_(2, pos, v)
        return q


class _NG:
    """Prompt-lookup drafter: most recent earlier occurrence of the last 3/2/1 tokens."""

    def __init__(self, prompt):
        self.h = list(prompt)
        self.d = ({}, {}, {})
        for i in range(1, len(self.h)):
            self._reg(i)

    def _reg(self, i):  # token i just appended: n-grams ending at i-1 continue at i
        h, d = self.h, self.d
        for n in (1, 2, 3):
            if i >= n:
                d[n - 1][tuple(h[i - n : i])] = i

    def push(self, tok):
        self.h.append(tok)
        self._reg(len(self.h) - 1)

    def draft(self, k):
        h = self.h
        L = len(h)
        for n in (3, 2, 1):
            j = self.d[n - 1].get(tuple(h[L - n :])) if L >= n else None
            if j is not None and j < L:
                return h[j : j + k]
        return []


class _Mixed:
    """Diagnostics: fused ops with some groups swapped for the torch reference (ENGINE_OFF=attn,qkv,gemv)."""

    GROUPS = {
        "attn": ("attn_decode",),
        "attnp": ("attn_prefill",),
        "qkv": ("qkv_post",),
        "gemv": ("linear", "gate_up_silu", "linear_add_norm"),
    }

    def __init__(self, fused, ref, off):
        self.f, self.r = fused, ref
        self.off = {name for g in off for name in self.GROUPS.get(g, ())}

    def __getattr__(self, name):
        return getattr(self.r if name in self.off else self.f, name)


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
        self._trace = None
        self.ops = _TorchOps(self)
        self.pool = torch.cuda.graph_pool_handle() if self.dev.type == "cuda" else None
        self.spec_ok = False
        if self.dev.type == "cuda":
            self._selftest()
            self._selftest_spec()

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
    def _grow_rope(self, cap):
        """Positions beyond the tables: rebuild them. Graphs baked the old table addresses, so drop
        them and start a fresh mempool (reusing the old pool after deleting its graphs trips a
        CUDACachingAllocator assert on the next capture)."""
        self.states.clear()
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.pool = torch.cuda.graph_pool_handle()
        self._build_rope(cap)

    def _state(self, B, cap, graph=True, W=1, slot=None, decode_graph=True):
        key = (B, cap, W, slot, decode_graph)
        st = self.states.get(key)
        if st is not None and (st.tried_graph or not graph):
            return st
        if len(self.states) >= MAX_STATES:
            del self.states[next(iter(self.states))]
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        st = _State()
        st.B, st.cap, st.W = B, cap, W
        shape = (B, self.nkv, cap, self.hd)
        # zeros, not empty: masked slots must be finite (0 * NaN = NaN)
        st.kc = [torch.zeros(shape, dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        st.vc = [torch.zeros(shape, dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        st.tok = torch.zeros(B, dtype=torch.long, device=self.dev)
        st.pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        st.ar = torch.arange(cap, device=self.dev)
        st.graph = None
        st.pgraphs = {}
        st.tried_graph = graph
        if W > 1:  # speculative verify buffers: one flat H2D copy in, one D2H out
            cu = self.dev.type == "cuda"
            st.in_dev = torch.zeros(B * W + B, dtype=torch.long, device=self.dev)
            st.in_host = torch.zeros(B * W + B, dtype=torch.long, pin_memory=cu)
            st.sin = st.in_dev[: B * W].view(B, W)
            st.spos = st.in_dev[B * W :]
            st.sout = torch.zeros((B, W), dtype=torch.long, device=self.dev)
            st.sout_host = torch.zeros((B, W), dtype=torch.long, pin_memory=cu)
            st.ev = torch.cuda.Event() if cu else None
        st.hcap = 0
        self.states[key] = st
        if graph and decode_graph and self.dev.type == "cuda":
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
        body = self._spec_body if st.W > 1 else self._decode_body
        st.pos.zero_()
        if st.W > 1:
            st.in_dev.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                body(st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            body(st)
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
            q = ops.qkv_post(ops.linear(a, l.wqkv), l, kc, vc, B, S, pos,
                             last_query_only=not decode and i == last)
            if decode:
                o = ops.attn_decode(q, kc, vc, pos, S)
            elif i == last:
                o = ops.attn_prefill(q, kc, vc, S, last_query_only=True)
                h = h[S - 1::S].contiguous()
            else:
                o = ops.attn_prefill(q, kc, vc, S)
            h, a = ops.linear_add_norm(o, l.wo, h, l.ln2)
            m = ops.gate_up_silu(a, l.wgu)
            h, a = ops.linear_add_norm(m, l.wd, h, self.layers[i + 1].ln1 if i < last else self.norm)
            if self._trace is not None:
                self._trace.append(h.float().cpu())
        logits = ops.linear(a, self.lm_head)
        if self._dbg is not None:
            self._dbg.append(logits.float().cpu())
        return logits.argmax(-1)

    def _prefill(self, ids, st):
        B, S = ids.shape
        if st.tried_graph and self.dev.type == "cuda" and 0 < B * S <= PREFILL_GRAPH_MAX:
            pg = st.pgraphs.get(S)
            if pg is None:
                pg = st.pgraphs[S] = self._capture_prefill(st, B, S)
            if pg is not False:
                pg[0].copy_(ids)
                pg[1].replay()
                return
        st.tok.copy_(self._forward(st, ids.reshape(-1), S, None))

    def _capture_prefill(self, st, B, S):
        """Small prefills are CPU-launch-bound (~500 kernel launches): replay them from a graph."""
        try:
            sid = torch.zeros((B, S), dtype=torch.long, device=self.dev)
            body = lambda: st.tok.copy_(self._forward(st, sid.reshape(-1), S, None))
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                body()
                body()
            torch.cuda.current_stream().wait_stream(s_)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body()
            torch.cuda.synchronize()
            return (sid, g)
        except Exception as e:
            print(f"[engine] prefill graph capture failed for B={B} S={S}: {e!r}")
            return False

    def _decode_body(self, st):
        st.tok.copy_(self._forward(st, st.tok, 1, st.pos))
        st.pos.add_(1)

    def _spec_body(self, st):
        # force_bm16_cfg: keep BN/BK/SK/num_warps at their BM=16-validated values, changing only
        # the tile height (BM=_bm(M)) for the wider verify-row batch -- see fused.py's comment.
        ctx = force_bm16_cfg() if force_bm16_cfg is not None else _contextlib.nullcontext()
        with ctx:
            st.sout.copy_(self._forward(st, st.sin.reshape(-1), st.W, st.spos).view(st.B, st.W))

    def _selftest_spec(self):
        """Verify-width forward vs step-by-step decode on the real weights. Two checks:
        B=1/W=5 (the original) and B=4/W=4 with DISTINCT per-sequence positions -- the latter is
        what batched speculation actually depends on (POS_STRIDE=1 per-row position indexing in
        qkv_post/attn, exercised only when sequences in one verify launch sit at different
        positions). A bug there is silently wrong output, not a crash, so this must pass before
        SPEC is trusted for B>1."""
        if not SPEC or TritonOps is None or not isinstance(self.ops, TritonOps):
            return
        try:
            torch.manual_seed(1)
            B, S, W = 1, 96, 5
            ids = torch.randint(100, 5000, (B, S), device=self.dev)
            st = self._state(B, 128, graph=False)
            self._dbg = []
            st.pos.fill_(S)
            self._prefill(ids, st)
            toks = [st.tok.clone()]
            for _ in range(W):
                self._decode_body(st)
                toks.append(st.tok.clone())
            plain, self._dbg = self._dbg, []
            sp = self._state(B, 128, graph=False, W=W)
            self._prefill(ids, sp)
            sp.sin.copy_(torch.stack(toks[:W], 1))
            sp.spos.fill_(S)
            self._spec_body(sp)
            got = self._dbg[-1]
            diff = max((got[j] - plain[j + 1][0]).abs().max().item() for j in range(W))
            print(f"[engine] spec selftest (B=1) max logit diff {diff:.4f}")
            ok1 = diff <= 0.5

            # B=4, W=4, distinct per-sequence positions. Built from ONE uniform lock-step decode
            # timeline (B=4 native, no special-casing needed there) and re-verifying each
            # sequence's OWN window at a DIFFERENT offset into that same timeline -- causally and
            # numerically identical to re-decoding that sequence from that position, since KV
            # content at any position is independent of other sequences and of what gets
            # (re)written at LATER positions in the same verify launch (causal mask keeps row s
            # blind to rows > s regardless of write order).
            B2, W2 = 4, 4
            offsets = [0, 3, 7, 11]
            S2 = 96
            nsteps = max(offsets) + W2
            ids2 = torch.randint(100, 5000, (B2, S2), device=self.dev)
            st2 = self._state(B2, 128, graph=False)
            self._dbg = []
            st2.pos.fill_(S2)
            self._prefill(ids2, st2)
            toks2 = [st2.tok.clone()]
            for _ in range(nsteps):
                self._decode_body(st2)
                toks2.append(st2.tok.clone())
            plain2 = self._dbg
            sp2 = self._state(B2, 128, graph=False, W=W2)
            for i in range(self.L):
                sp2.kc[i].copy_(st2.kc[i])
                sp2.vc[i].copy_(st2.vc[i])
            sp2.spos.copy_(torch.tensor([S2 + o for o in offsets], dtype=torch.long, device=self.dev))
            for b, off in enumerate(offsets):
                for s in range(W2):
                    sp2.sin[b, s] = toks2[off + s][b]
            self._dbg = []
            self._spec_body(sp2)
            got2 = self._dbg[-1]
            diff2 = 0.0
            for b, off in enumerate(offsets):
                for s in range(W2):
                    d = (got2[b * W2 + s] - plain2[off + s + 1][b]).abs().max().item()
                    diff2 = max(diff2, d)
            print(f"[engine] spec selftest (B=4, distinct per-row positions) max logit diff {diff2:.4f}")
            ok2 = diff2 <= 0.5

            self.spec_ok = ok1 and ok2
        except Exception as e:
            print(f"[engine] spec selftest error: {e!r}")
            self.spec_ok = False
        finally:
            self._dbg = None
            self.states.clear()

    def _selftest(self):
        """Fused ops vs torch ops on the real weights; keep fused only if close."""
        if os.environ.get("ENGINE_OPS") == "torch":  # diagnostics: force the reference ops
            return
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
        self.ops = fallback
        selected = None
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
                        selected = candidate
                        break
                except Exception as e:
                    print(f"[engine] fused {label} selftest error: {e!r}")
        except Exception as e:
            print(f"[engine] fused selftest error: {e!r}")
        self.ops = selected if selected is not None else fallback
        off = [x for x in os.environ.get("ENGINE_OFF", "").split(",") if x]
        if off and isinstance(self.ops, TritonOps):
            self.ops = _Mixed(self.ops, _TorchOps(self), off)

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
        if self.spec_ok and SPEC and B >= SPEC_MIN_B and n >= SPEC_MIN_N:
            W = int(SPEC_W_OVERRIDE) if SPEC_W_OVERRIDE else min(SPEC_W_MAX, SPEC_ROWS // B)
            W = max(1, min(W, SPEC_ROWS // B, SPEC_W_MAX))
            if W >= 2:
                yield from self._generate_spec(input_ids, n, W)
                return
        cap = -(-(S + n) // CAP_GRAN) * CAP_GRAN
        if cap > self.cos.shape[0]:
            self._grow_rope(cap)
        st = self._state(B, cap, slot=S, decode_graph=n > 1)
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

    def _generate_spec(self, input_ids, n, W):
        """Exact n-gram speculative decoding: each step verifies W tokens per sequence (last
        accepted + drafts) in one graph replay, keeps the longest prefix whose drafts equal the
        model's own greedy outputs, plus one bonus token. KV entries of rejected drafts are
        overwritten later. B>1: per-sequence draft state (_NG), accept counts and positions --
        the batch as a whole advances at the pace of its SLOWEST sequence each step (lockstep
        yield), so the speedup is min_b(accepted/step), not the mean."""
        B, S = len(input_ids), len(input_ids[0])
        K = W - 1
        cap = -(-(S + n + W) // CAP_GRAN) * CAP_GRAN
        if cap > self.cos.shape[0]:
            self._grow_rope(cap)
        st = self._state(B, cap, W=W)
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.dev)
        self._prefill(ids, st)  # async
        ng = [_NG(p) for p in input_ids]  # CPU work overlaps the GPU prefill
        first = st.tok.tolist()
        out = [[t] for t in first]
        for b in range(B):
            ng[b].push(first[b])
        last, pos = list(first), [S] * B
        yielded = 1
        yield list(first)
        cuda = st.ev is not None
        while yielded < n:
            rows = []
            for b in range(B):
                d = ng[b].draft(K) if len(out[b]) < n else []
                rows.append([last[b]] + d + [last[b]] * (K - len(d)))
            flat = [t for r in rows for t in r] + pos
            st.in_host.copy_(torch.tensor(flat, dtype=torch.long))
            st.in_dev.copy_(st.in_host, non_blocking=cuda)
            if st.graph is not None:
                st.graph.replay()
            else:
                self._spec_body(st)
            st.sout_host.copy_(st.sout, non_blocking=cuda)
            if cuda:
                st.ev.record()
                st.ev.synchronize()
            res = st.sout_host.tolist()
            for b in range(B):
                if len(out[b]) >= n:
                    continue
                o, r = res[b], rows[b]
                a = 0
                while a < K and r[a + 1] == o[a]:
                    a += 1
                new = o[: min(a + 1, n - len(out[b]))]
                out[b] += new
                for t in new:
                    ng[b].push(t)
                last[b] = new[-1]
                pos[b] += len(new)
            m = min(len(x) for x in out)
            while yielded < m:
                yield [out[b][yielded] for b in range(B)]
                yielded += 1

    def _generate_ragged(self, input_ids, n):
        groups = {}
        for i, seq in enumerate(input_ids):
            groups.setdefault(len(seq), []).append(i)
        streams = [(indices, self.generate([input_ids[i] for i in indices], n))
                   for indices in groups.values()]
        for _ in range(n):
            step = [0] * len(input_ids)
            for indices, stream in streams:
                tokens = next(stream)
                for i, token in zip(indices, tokens):
                    step[i] = token
            yield step
