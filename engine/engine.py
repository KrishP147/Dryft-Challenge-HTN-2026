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
    from fused import TritonOps, force_bm16_cfg, quant_fp8_tensorwise
except Exception as _e:  # no triton / import failure: torch ops only
    TritonOps = None
    force_bm16_cfg = None
    quant_fp8_tensorwise = None
    _FUSED_ERR = repr(_e)

CAP_GRAN = 128
# 8192 covers both heavy public shapes exactly (B4x2048 and B16x512 are both B*S=8192), which the
# old 4096 cap excluded. Measured in-engine: +0.34% over 3 repeated pairs, capture verified real
# (st.pgraphs[S] holds a (sid, CUDAGraph) tuple, not False), and peak memory is LOWER graphed than
# eager (B4 17.4->16.6 GiB, B16 18.8->18.0 GiB) because graph-pool reuse beats ad-hoc allocation.
# 16384 measured no better. Audit shapes 1,8192,64 and 2,3000,32 clean at this cap.
PREFILL_GRAPH_MAX = int(os.environ.get("ENGINE_PREFILL_GRAPH", "8192"))  # B*S at or below this: prefill runs as a CUDA graph (0 = off)
SPEC = os.environ.get("ENGINE_SPEC", "1") == "1"  # exact speculation; engaged per the B/n policy below
SPEC_ROWS = 64  # max B*W rows through the skinny GEMVs
SPEC_MODE = os.environ.get("ENGINE_SPEC_MODE", "gpu")  # "gpu" (default): drafting + verification +
                # acceptance run entirely inside the CUDA graph (_generate_gspec, dense per-sequence
                # bigram/token-recycling table drafter), host only harvests results one step behind,
                # pipelined exactly like the plain decode path -- no per-step host round trip.
                # "host": CPU n-gram drafter with a synchronous per-step round trip
                # (_generate_spec). The two algorithms were tuned independently (different
                # floor/upside tradeoffs), so the B/n/W gate defaults below switch with the mode;
                # ENGINE_SPEC_* env vars still override either one explicitly.
if SPEC_MODE == "gpu":
    # Tuned on natural-prose pod measurements (tests/bench.py --corpus prose) for the in-graph
    # token-recycling drafter: B4 2048->{64,128} +4.4%/+11.2% CV 2-7%, B2 3000->96 +7.0% CV 8%.
    # B=1 shows the BIGGEST win (+20-29% at n>=64) but an unfixable single-sequence timing spread
    # that hits 22-27% at n=128 -- over the platform's 25% CV limit -- so it stays out of the
    # gate pending a per-batch-width fix. n=32 excluded too (B4 2048->32 measured -0.3%, noise,
    # not a win). MIN_B=2/MAX_B=4/MIN_N=64 => all three public shapes (B1 512->*, B4 2048->32,
    # B16 512->128) are OUTSIDE this gate and run bit-identical to plain decode.
    _MIN_B_DEF, _MAX_B_DEF, _MIN_N_DEF, _W_MAX_DEF = "2", "4", "64", "4"
else:
    # Tuned for the host n-gram drafter as an explicit floor-vs-upside A/B (pod, 5 samples). The
    # FLOOR is what a build pays when the draft never hits (poisoned with random ids) -- what most
    # official spec runs mostly collected, since platform acceptance is low:
    #   W:        2       3       4       7
    #   floor:   -2.0%   -3.2%   -3.8%   -8.0%   (geomean B4-2048-32/128, B8-1024-64)
    #   n=32:    +5.2%   +5.4%   +5.7%   +4.5%   (B4 2048, pydoc, where drafts do hit)
    #   n=128:  +13.0%  +13.7%  +15.0%  +11.8%
    # W=2 gives up almost nothing at short outputs and costs a quarter of W=7 when the draft
    # misses. B16 stays excluded: floor -5.5%/-2.9% against a pydoc upside of only +4.4%/+4.7% --
    # the one batch where the floor exceeds the best case. No length gate: at W=2 the floor is
    # only -2% and n=32 already measures +5.2%, so there is nothing left to dodge.
    _MIN_B_DEF, _MAX_B_DEF, _MIN_N_DEF, _W_MAX_DEF = "2", "8", "1", "2"
SPEC_MIN_B = int(os.environ.get("ENGINE_SPEC_MIN_B", _MIN_B_DEF))
SPEC_MAX_B = int(os.environ.get("ENGINE_SPEC_MAX_B", _MAX_B_DEF))
SPEC_MIN_N = int(os.environ.get("ENGINE_SPEC_MIN_N", _MIN_N_DEF))
SPEC_W_MAX = int(os.environ.get("ENGINE_SPEC_W_MAX", _W_MAX_DEF))
SPEC_W_OVERRIDE = os.environ.get("ENGINE_SPEC_W")  # force a specific W for the A/B sweep
MAX_STATES = 6
ROPE_LEN = 32768  # tables for cap <= this are built once; growing past it rebuilds them and drops the graphs
FP8 = os.environ.get("ENGINE_FP8", "1") == "1"  # tensorwise-fp8 prefill GEMMs (compute-bound, ~1.7x)
# which prefill projections use fp8. o/down write the residual stream (h += proj), so fp8 there
# perturbs the bistable layer-~16 massive activation and can flip a token past the 2-logit gate;
# excluding them ("qkv,gu") trades ~half the prefill-fp8 speedup for a safer rollout.
FP8_OPS = set(os.environ.get("ENGINE_FP8_OPS", "qkv,gu").split(","))


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
        self.gspec_ok = False
        # fp8 prefill is disabled during the 0.5-tolerance selftests (fp8's ~0.5-logit drift would
        # trip the bf16 fallback); enabled for real generation, validated separately at the 2.0 gate.
        self.fp8_prefill = False
        if self.dev.type == "cuda":
            self._selftest()
            self._selftest_spec()
            if SPEC_MODE == "gpu":
                self._selftest_gspec()
            self.fp8_prefill = FP8 and isinstance(self.ops, TritonOps) and hasattr(self.layers[0], "wqkv8")

    def _fp8_lin(self, x, w8, ws, wbf):
        """x[M,K] bf16 -> tensorwise-fp8 GEMM against fp8 weight w8[N,K] (scale ws[1,1]) -> bf16[M,N].
        Fast cuBLASLt fp8 path. Falls back to bf16 for skinny/unaligned M (fp8 GEMM needs M>=16 and
        16-aligned, and is not worth it for tiny M anyway, e.g. the last prefill layer's B rows)."""
        x = x.contiguous()
        M = x.shape[0]
        if M < 16 or M % 16 != 0:
            return F.linear(x, wbf)
        xf, xs = quant_fp8_tensorwise(x)  # fused cast kernel (~3ms/prefill vs ~45ms naive torch)
        return torch._scaled_mm(xf, w8.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)

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
        if FP8 and self.dev.type == "cuda":
            # tensorwise (scalar-scale) e4m3 weights for the prefill GEMM path: passes the 2-logit
            # gate at ~0.5 (measured) and hits cuBLASLt's fast fp8 kernel (~1.7x over bf16 cuBLAS).
            # bf16 weights are kept for the memory-bound decode GEMV (fp8 there is ramp-bound, no win).
            for l in self.layers:
                for wn in ("wqkv", "wo", "wgu", "wd"):
                    wt = getattr(l, wn)
                    s = (wt.abs().amax().clamp(min=1e-6) / 448.0).float().reshape(1, 1)
                    setattr(l, wn + "8", (wt / s).to(torch.float8_e4m3fn).contiguous())
                    setattr(l, wn + "8s", s)

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
        use8 = self.fp8_prefill and not decode  # fp8 GEMMs on the compute-bound prefill path
        h = self.embed[tokens]
        a = ops.rms(h, self.layers[0].ln1)
        last = self.L - 1
        for i, l in enumerate(self.layers):
            kc, vc = st.kc[i], st.vc[i]
            qkv = self._fp8_lin(a, l.wqkv8, l.wqkv8s, l.wqkv) if (use8 and "qkv" in FP8_OPS) else ops.linear(a, l.wqkv)
            q = ops.qkv_post(qkv, l, kc, vc, B, S, pos, last_query_only=not decode and i == last)
            if decode:
                o = ops.attn_decode(q, kc, vc, pos, S)
            elif i == last:
                o = ops.attn_prefill(q, kc, vc, S, last_query_only=True)
                h = h[S - 1::S].contiguous()
            else:
                o = ops.attn_prefill(q, kc, vc, S)
            if use8:  # per-op fp8; o/down (residual writers) can be kept bf16 via FP8_OPS
                if "o" in FP8_OPS:
                    hn = h + self._fp8_lin(o, l.wo8, l.wo8s, l.wo); a = ops.rms(hn, l.ln2); h = hn
                else:
                    h, a = ops.linear_add_norm(o, l.wo, h, l.ln2)
                m = (ops.silu_mul(self._fp8_lin(a, l.wgu8, l.wgu8s, l.wgu)) if "gu" in FP8_OPS
                     else ops.gate_up_silu(a, l.wgu))
                nln = self.layers[i + 1].ln1 if i < last else self.norm
                if "down" in FP8_OPS:
                    hn = h + self._fp8_lin(m, l.wd8, l.wd8s, l.wd); a = ops.rms(hn, nln); h = hn
                else:
                    h, a = ops.linear_add_norm(m, l.wd, h, nln)
            else:
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
        if self.spec_ok and SPEC and SPEC_MIN_B <= B <= SPEC_MAX_B and n >= SPEC_MIN_N:
            W = int(SPEC_W_OVERRIDE) if SPEC_W_OVERRIDE else min(SPEC_W_MAX, SPEC_ROWS // B)
            W = max(1, min(W, SPEC_ROWS // B, SPEC_W_MAX))
            if W >= 2:
                if SPEC_MODE == "gpu" and self.gspec_ok:
                    yield from self._generate_gspec(input_ids, n, W)
                else:
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

    # ----------------------------------------------------------- GPU-graph spec
    # Drafting, verification and acceptance all run ON DEVICE inside the captured graph,
    # so the host never blocks on a per-step round trip -- replay(t+1) is launched before
    # the host reads step t's result, exactly like the plain decode path.
    #
    # Drafter: a dense per-sequence "token recycling" table T (B, vocab) long, T[b, v] =
    # the model's own last argmax prediction for whenever token v was fed as input to
    # sequence b (agent C's sim: beats n-gram prompt-lookup on natural prose, and needs
    # ~5 device ops/step instead of ~20-25 for an in-graph n-gram search over `hist`).
    # d1 = T[last], d2 = T[d1], ... chains K=W-1 gathers. Seeded from the prompt's own
    # bigrams (T[prompt[i]] = prompt[i+1]) and refreshed every step from the verify
    # forward's real output (T[row] = out), so it tracks the model's actual behavior on
    # THIS sequence, not just literal repeats. Exactness is unaffected by what the
    # drafter guesses (see _generate_spec's docstring / this file's history): any
    # accepted token is validated by the verify forward's own induction regardless of
    # how it was drafted, so a table miss is merely wasted work, never wrong output.
    #   tok (B,) long: last confirmed token; NO KV yet (same invariant as plain `tok`).
    #   gpos (B,) long: tok's absolute position.
    #   limit (1,) long: S + n - 1 (max valid position), refilled per call.
    def _gstate(self, B, cap, W):
        key = ("g", B, cap, W)
        st = self.states.get(key)
        if st is not None:
            return st
        if len(self.states) >= MAX_STATES:
            del self.states[next(iter(self.states))]
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        st = _State()
        st.B, st.cap, st.W = B, cap, W
        st.gpu = True
        shape = (B, self.nkv, cap, self.hd)
        st.kc = [torch.zeros(shape, dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        st.vc = [torch.zeros(shape, dtype=self.dtype, device=self.dev) for _ in range(self.L)]
        st.tok = torch.zeros(B, dtype=torch.long, device=self.dev)
        st.gpos = torch.zeros(B, dtype=torch.long, device=self.dev)
        st.limit = torch.zeros(1, dtype=torch.long, device=self.dev)
        st.T = torch.zeros(B, self.embed.shape[0], dtype=torch.long, device=self.dev)
        st.sin = torch.zeros(B, W, dtype=torch.long, device=self.dev)
        st.spos = torch.zeros(B, dtype=torch.long, device=self.dev)
        st.sout = torch.zeros(B, W, dtype=torch.long, device=self.dev)
        st.gacc = torch.zeros(B, dtype=torch.long, device=self.dev)
        st.pos = torch.zeros(1, dtype=torch.long, device=self.dev)  # unused; kept so any
                # shared helper that touches st.pos (none currently) does not crash
        st.tried_graph = True
        st.graph = None
        st.pgraphs = {}
        st.hcap = 0
        self.states[key] = st
        if self.dev.type == "cuda":
            try:
                self._capture_gspec(st)
            except Exception as e:  # fall back to eager
                print(f"[engine] gspec graph capture failed for {key}: {e!r}")
                st.graph = None
        return st

    def _capture_gspec(self, st):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._gspec_body(st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self._gspec_body(st)
        st.graph = g
        torch.cuda.synchronize()

    def _gspec_body(self, st):
        """One in-graph step: chain K=W-1 table lookups for the draft, verify all W rows
        in one forward, accept the longest correct prefix + 1 bonus token, recycle every
        row's (input, model-output) pair into the table, advance tok/gpos."""
        B, W = st.B, st.W
        K = W - 1
        tok, T, limit = st.tok, st.T, st.limit
        drafts = []
        cur = tok
        for _ in range(K):
            cur = T.gather(1, cur.unsqueeze(1)).squeeze(1)
            drafts.append(cur)
        rows = torch.stack([tok] + drafts, dim=1) if K else tok.unsqueeze(1)  # (B, W)
        st.sin.copy_(rows)
        st.spos.copy_(st.gpos)
        ctx = force_bm16_cfg() if force_bm16_cfg is not None else _contextlib.nullcontext()
        with ctx:
            st.sout.copy_(self._forward(st, st.sin.reshape(-1), W, st.spos).view(B, W))
        if K:
            d = torch.stack(drafts, dim=1)  # (B, K)
            match = (d == st.sout[:, :K]).long()
            acc = torch.cumprod(match, 1).sum(1)
        else:
            acc = torch.zeros(B, dtype=torch.long, device=tok.device)
        T.scatter_(1, rows, st.sout)  # token recycling: T[input] = model's own next-token
        st.gacc.copy_(acc)
        st.tok.copy_(st.sout.gather(1, acc.unsqueeze(1)).squeeze(1))
        st.gpos.copy_(torch.minimum(st.gpos + acc + 1, limit))

    def _generate_gspec(self, input_ids, n, W):
        """Host side: launch replay(t+1) before consuming replay(t)'s result (pipelined,
        like the plain decode path), reconstruct each sequence's stream from the device-
        computed (out, acc) pair, yield lockstep at the min length over sequences."""
        B, S = len(input_ids), len(input_ids[0])
        cap = -(-(S + n + 2 * W) // CAP_GRAN) * CAP_GRAN
        if cap > self.cos.shape[0]:
            self._grow_rope(cap)
        st = self._gstate(B, cap, W)
        st.limit.fill_(S + n - 1)
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.dev)
        self._prefill(ids, st)  # writes st.tok = first generated token via st.kc/st.vc
        st.T.zero_()
        if S >= 2:
            st.T.scatter_(1, ids[:, :-1], ids[:, 1:])
        st.T.scatter_(1, ids[:, -1:], st.tok.unsqueeze(1))
        st.gpos.fill_(S)
        first = st.tok.tolist()
        out = [[t] for t in first]
        yielded = 1
        yield list(first)
        if yielded >= n:
            return
        cuda = self.dev.type == "cuda"
        max_steps = n  # worst case: 1 confirmed token per step (acc=0 every step)
        gout = torch.zeros((max_steps, B, W), dtype=torch.long, pin_memory=cuda)
        gacc = torch.zeros((max_steps, B), dtype=torch.long, pin_memory=cuda)
        gev = [torch.cuda.Event() for _ in range(max_steps)] if cuda else [None] * max_steps

        def launch(t):
            if st.graph is not None:
                st.graph.replay()
            else:
                self._gspec_body(st)
            gout[t].copy_(st.sout, non_blocking=cuda)
            gacc[t].copy_(st.gacc, non_blocking=cuda)
            if cuda:
                gev[t].record()

        launch(0)
        t = 0
        while True:
            finished = all(len(x) >= n for x in out)
            advanced = not finished and t + 1 < max_steps
            if advanced:
                launch(t + 1)
            if cuda:
                gev[t].synchronize()
            o, a = gout[t].tolist(), gacc[t].tolist()
            for b in range(B):
                if len(out[b]) >= n:
                    continue
                take = min(a[b] + 1, n - len(out[b]))
                out[b] += o[b][:take]
            m = min(len(x) for x in out)
            while yielded < m:
                yield [out[b][yielded] for b in range(B)]
                yielded += 1
            if yielded >= n or not advanced:
                break
            t += 1

    def _selftest_gspec(self):
        """Token-exact check against real autoregressive plain decode (not just a logit-gap
        tolerance): the challenge requires bit-identical greedy output, so this must pass
        before ENGINE_SPEC_MODE=gpu is trusted."""
        if not SPEC or TritonOps is None or not isinstance(self.ops, TritonOps):
            return
        try:
            torch.manual_seed(3)
            for (B, S, n, W) in [(1, 80, 24, 4), (3, 96, 30, 3), (4, 64, 20, 5)]:
                ids = torch.randint(100, 5000, (B, S), device=self.dev)
                input_ids = ids.tolist()
                cap = -(-(S + n) // CAP_GRAN) * CAP_GRAN
                stp = self._state(B, cap, graph=False)
                stp.pos.fill_(S)
                self._prefill(ids, stp)
                truth = [stp.tok.tolist()]
                for _ in range(n - 1):
                    self._decode_body(stp)
                    truth.append(stp.tok.tolist())
                self.states.clear()
                # fresh mempool: repeated capture/free of graphs sharing self.pool across
                # these iterations otherwise trips a CUDACachingAllocator assert (same
                # hazard _grow_rope guards against).
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                self.pool = torch.cuda.graph_pool_handle()
                got = list(self._generate_gspec(input_ids, n, W))
                self.states.clear()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                self.pool = torch.cuda.graph_pool_handle()
                ok = got == truth
                print(f"[engine] gspec selftest B={B} S={S} n={n} W={W} match={ok}")
                if not ok:
                    self.gspec_ok = False
                    return
            self.gspec_ok = True
        except Exception as e:
            print(f"[engine] gspec selftest error: {e!r}")
            self.gspec_ok = False
        finally:
            self.states.clear()

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
