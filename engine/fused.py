"""Fused Triton ops for the Qwen3 engine.

Each kernel mirrors the reference's bf16 rounding points (explicit casts), so
results track the torch path within the tie margin. The engine self-tests this
op set against the torch ops at load and falls back if anything is off.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_rms_kernel(
    x_ptr, d_ptr, w_ptr, h_ptr, y_ptr, n_cols, eps,
    HAS_ADD: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    off = row * n_cols + cols
    x = tl.load(x_ptr + off, mask=m, other=0.0)
    if HAS_ADD:
        d = tl.load(d_ptr + off, mask=m, other=0.0)
        # residual add rounds to bf16 before the norm, like the reference
        x = (x.to(tl.float32) + d.to(tl.float32)).to(x_ptr.dtype.element_ty)
        tl.store(h_ptr + off, x, mask=m)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / n_cols
    normed = xf * tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    # round to bf16 BEFORE the weight multiply (matches Qwen3RMSNorm)
    tl.store(y_ptr + off, normed.to(y_ptr.dtype.element_ty) * w, mask=m)


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, n, inter, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    row = idx // inter
    col = idx % inter
    g = tl.load(gu_ptr + row * 2 * inter + col, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * inter + inter + col, mask=m, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + idx, (s.to(tl.float32) * u).to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _rope_half(a, b, ca, cb, sa, sb, DT: tl.constexpr):
    # reference: x * cos + rotate_half(x) * sin, each op rounded to bf16
    ta = (a.to(tl.float32) * ca.to(tl.float32)).to(DT)
    ua = ((-b).to(tl.float32) * sa.to(tl.float32)).to(DT)
    tb = (b.to(tl.float32) * cb.to(tl.float32)).to(DT)
    ub = (a.to(tl.float32) * sb.to(tl.float32)).to(DT)
    return (ta.to(tl.float32) + ua.to(tl.float32)).to(DT), (
        tb.to(tl.float32) + ub.to(tl.float32)
    ).to(DT)


@triton.jit
def _qkv_post_kernel(
    qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_ptr, kc_ptr, vc_ptr, S, cap, eps,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, DECODE: tl.constexpr,
):
    # one program per (token, head): q/k get qk-norm + rope, v is copied; k/v go
    # straight into the static cache, q into [B, NH, S, HD].
    t = tl.program_id(0)
    hidx = tl.program_id(1)
    b = t // S
    s = t % S
    DT: tl.constexpr = q_ptr.dtype.element_ty
    HALF: tl.constexpr = HD // 2
    offs = tl.arange(0, HALF)
    TOTAL: tl.constexpr = (NH + 2 * NKV) * HD
    base = qkv_ptr + t * TOTAL + hidx * HD
    if DECODE:
        position = tl.load(pos_ptr)
    else:
        position = s
    if hidx < NH + NKV:
        x1 = tl.load(base + offs)
        x2 = tl.load(base + HALF + offs)
        f1 = x1.to(tl.float32)
        f2 = x2.to(tl.float32)
        var = (tl.sum(f1 * f1, axis=0) + tl.sum(f2 * f2, axis=0)) / HD
        r = tl.math.rsqrt(var + eps)
        w1 = tl.where(hidx < NH, tl.load(qn_ptr + offs), tl.load(kn_ptr + offs))
        w2 = tl.where(hidx < NH, tl.load(qn_ptr + HALF + offs), tl.load(kn_ptr + HALF + offs))
        n1 = (f1 * r).to(DT) * w1
        n2 = (f2 * r).to(DT) * w2
        cb = position * HD
        c1 = tl.load(cos_ptr + cb + offs)
        c2 = tl.load(cos_ptr + cb + HALF + offs)
        s1 = tl.load(sin_ptr + cb + offs)
        s2 = tl.load(sin_ptr + cb + HALF + offs)
        o1, o2 = _rope_half(n1, n2, c1, c2, s1, s2, DT)
        if hidx < NH:
            qo = q_ptr + ((b * NH + hidx) * S + s) * HD
            tl.store(qo + offs, o1)
            tl.store(qo + HALF + offs, o2)
        else:
            ko = kc_ptr + ((b * NKV + (hidx - NH)) * cap + position) * HD
            tl.store(ko + offs, o1)
            tl.store(ko + HALF + offs, o2)
    else:
        v1 = tl.load(base + offs)
        v2 = tl.load(base + HALF + offs)
        vo = vc_ptr + ((b * NKV + (hidx - NH - NKV)) * cap + position) * HD
        tl.store(vo + offs, v1)
        tl.store(vo + HALF + offs, v2)


@triton.jit
def _attn_split_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, ws_ptr, out_ptr, cap, sm_scale,
    NSPLIT: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # flash-decoding partial: one program per (batch*kv_head, split). The G query
    # heads sharing a kv head are the M rows of the dot (padded to GP >= 16).
    bk = tl.program_id(0)
    sp = tl.program_id(1)
    L = tl.load(pos_ptr) + 1
    chunk = (L + NSPLIT - 1) // NSPLIT
    start = sp * chunk
    end = tl.minimum(start + chunk, L)
    rows = tl.arange(0, GP)
    d = tl.arange(0, HD)
    rmask = rows < G
    q = tl.load(
        q_ptr + (bk * G + rows[:, None]) * HD + d[None, :], mask=rmask[:, None], other=0.0
    )
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, HD], tl.float32)
    kv_base = bk.to(tl.int64) * cap * HD
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        k = tl.load(k_ptr + kv_base + n[:, None] * HD + d[None, :], mask=nm[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * sm_scale
        sc = tl.where(nm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + kv_base + n[:, None] * HD + d[None, :], mask=nm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    if NSPLIT == 1:
        # One split needs no log-sum-exp merge or workspace allocation.
        tl.store(out_ptr + (bk * G + rows[:, None]) * HD + d[None, :],
                 (acc / l_i[:, None]).to(out_ptr.dtype.element_ty),
                 mask=rmask[:, None])
    else:
        wb = ((bk * G + rows) * NSPLIT + sp) * (HD + 2)
        tl.store(ws_ptr + wb[:, None] + d[None, :], acc, mask=rmask[:, None])
        tl.store(ws_ptr + wb + HD, m_i, mask=rmask)
        tl.store(ws_ptr + wb + HD + 1, l_i, mask=rmask)


@triton.jit
def _attn_combine_kernel(
    ws_ptr, out_ptr, NSPLIT: tl.constexpr, SP: tl.constexpr, HD: tl.constexpr
):
    r = tl.program_id(0)
    s = tl.arange(0, SP)
    sm = s < NSPLIT
    base = (r * NSPLIT + s) * (HD + 2)
    m = tl.load(ws_ptr + base + HD, mask=sm, other=float("-inf"))
    l = tl.load(ws_ptr + base + HD + 1, mask=sm, other=0.0)
    mx = tl.max(m, 0)
    w = tl.exp(m - mx)
    lt = tl.sum(l * w, 0)
    d = tl.arange(0, HD)
    acc = tl.load(ws_ptr + base[:, None] + d[None, :], mask=sm[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], 0) / lt
    tl.store(out_ptr + r * HD + d, o.to(out_ptr.dtype.element_ty))


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, out_ptr, M, N, K, kps, stride_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FINAL: tl.constexpr,
):
    # skinny GEMM: out[M, N] = x[M, K] @ w[N, K]^T, M <= BM (16). Each program owns
    # BN rows of w and one K-slice (split-K). Bandwidth-bound: w is read once.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    mm = rm < M
    nm = rn < N
    k_lo = pid_k * kps
    k_hi = tl.minimum(k_lo + kps, K)
    acc = tl.zeros([BM, BN], tl.float32)
    for k in range(k_lo, k_hi, BK):
        rk = k + tl.arange(0, BK)
        km = rk < k_hi
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None] & km[None, :], other=0.0)
        w = tl.load(w_ptr + rn[None, :] * K + rk[:, None], mask=nm[None, :] & km[:, None], other=0.0)
        acc = tl.dot(x, w, acc)
    if FINAL:
        tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty),
                 mask=mm[:, None] & nm[None, :])
    else:
        tl.store(out_ptr + pid_k * M * stride_om + rm[:, None] * stride_om + rn[None, :], acc,
                 mask=mm[:, None] & nm[None, :])


@triton.jit
def _splitk_reduce_kernel(ws_ptr, out_ptr, n, SK: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    acc = tl.zeros([BLOCK], tl.float32)
    for i in range(SK):
        acc += tl.load(ws_ptr + i * n + idx, mask=m, other=0.0)
    tl.store(out_ptr + idx, acc.to(out_ptr.dtype.element_ty), mask=m)


# (N, K) -> (BN, BK, SPLIT_K, stages, warps), swept on H100 (tests/gemv_bench.py)
GEMM_CFG = {
    (6144, 2560): (64, 128, 1, 5, 4),
    (2560, 4096): (64, 256, 2, 4, 4),
    (19456, 2560): (64, 64, 1, 4, 4),
    (2560, 9728): (32, 128, 4, 4, 4),
    (151936, 2560): (64, 256, 1, 3, 4),
}


class TritonOps:
    def __init__(self, e, use_gemv=True):
        self.e = e
        self.use_gemv = use_gemv
        assert e.hd & (e.hd - 1) == 0 and e.hd >= 2
        self.dummy_pos = torch.zeros(1, dtype=torch.long, device=e.dev)

    def _rms_launch(self, h, d, w, has_add):
        rows, cols = h.shape
        y = torch.empty_like(h)
        hn = torch.empty_like(h) if has_add else h
        _add_rms_kernel[(rows,)](
            h, d if has_add else h, w, hn, y, cols, self.e.eps,
            HAS_ADD=has_add, BLOCK=triton.next_power_of_2(cols), num_warps=8,
        )
        return hn, y

    def rms(self, h, w):
        return self._rms_launch(h.contiguous(), None, w, False)[1]

    def add_rms(self, h, d, w):
        return self._rms_launch(h.contiguous(), d.contiguous(), w, True)

    def silu_mul(self, gu):
        rows = gu.shape[0]
        inter = gu.shape[1] // 2
        out = torch.empty((rows, inter), dtype=gu.dtype, device=gu.device)
        n = rows * inter
        _silu_mul_kernel[(triton.cdiv(n, 1024),)](gu, out, n, inter, BLOCK=1024)
        return out

    def qkv_post(self, qkv, l, kc, vc, B, S, pos):
        e = self.e
        q = torch.empty((B, e.nh, S, e.hd), dtype=qkv.dtype, device=qkv.device)
        _qkv_post_kernel[(B * S, e.nh + 2 * e.nkv)](
            qkv.contiguous(), l.qn, l.kn, e.cos, e.sin,
            pos if pos is not None else self.dummy_pos,
            q, kc, vc, S, kc.shape[2], e.eps,
            NH=e.nh, NKV=e.nkv, HD=e.hd, DECODE=pos is not None, num_warps=1,
        )
        return q

    def attn_decode(self, q, kc, vc, pos):
        """q: [B, nh, 1, hd] (contiguous). Attends over kc/vc[:, :, :pos+1]. -> [B, nh*hd]"""
        e = self.e
        B, G = q.shape[0], e.nh // e.nkv
        bk = B * e.nkv
        nsplit = 1
        while bk * nsplit < 256 and nsplit < 32:
            nsplit *= 2
        out = torch.empty((B, e.nh * e.hd), dtype=q.dtype, device=q.device)
        ws = (out if nsplit == 1 else
              torch.empty((bk * G, nsplit, e.hd + 2), dtype=torch.float32, device=q.device))
        _attn_split_kernel[(bk, nsplit)](
            q, kc, vc, pos, ws, out, kc.shape[2], e.hd ** -0.5,
            NSPLIT=nsplit, G=G, GP=max(16, triton.next_power_of_2(G)), HD=e.hd,
            BLOCK_N=64, num_warps=4, num_stages=2,
        )
        if nsplit > 1:
            _attn_combine_kernel[(bk * G,)](
                ws, out, NSPLIT=nsplit, SP=nsplit, HD=e.hd, num_warps=1
            )
        return out

    def linear(self, x, w):
        M, K = x.shape
        N = w.shape[0]
        cfg = GEMM_CFG.get((N, K))
        # These launch parameters and tl.dot arithmetic were tuned for bf16 on
        # H100. Preserve full-precision behavior for fp32/fp16 callers.
        if (not self.use_gemv or cfg is None or M > 16
                or not x.is_contiguous() or not w.is_contiguous()
                or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16):
            return torch.nn.functional.linear(x, w)
        BN, BK, SK, ST, NW = cfg
        kps = triton.cdiv(triton.cdiv(K, SK), BK) * BK
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        if SK == 1:
            _gemv_kernel[(triton.cdiv(N, BN), 1)](
                x, w, out, M, N, K, kps, N,
                BM=16, BN=BN, BK=BK, FINAL=True, num_warps=NW, num_stages=ST,
            )
            return out
        ws = torch.empty((SK, M, N), dtype=torch.float32, device=x.device)
        _gemv_kernel[(triton.cdiv(N, BN), SK)](
            x, w, ws, M, N, K, kps, N,
            BM=16, BN=BN, BK=BK, FINAL=False, num_warps=NW, num_stages=ST,
        )
        _splitk_reduce_kernel[(triton.cdiv(M * N, 1024),)](ws, out, M * N, SK=SK, BLOCK=1024)
        return out
