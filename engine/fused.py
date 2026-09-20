"""Fused Triton ops for the Qwen3 engine.

Each kernel mirrors the reference's bf16 rounding points (explicit casts), so
results track the torch path within the tie margin. The engine self-tests this
op set against the torch ops at load and falls back if anything is off.
"""
import os

import torch
import triton
import triton.language as tl

# Programmatic dependent launch (PDL): a kernel may start (and prefetch its weights)
# while the previous kernel in the stream is still finishing; it calls
# griddepcontrol.wait before touching anything the previous kernel produced.
# Enabled by patching Triton's generated C launcher: kernels whose packed metadata
# has cluster_dim_x == _PDL_MAGIC launch via cuLaunchKernelEx + the PDL attribute.
_PDL_MAGIC = 7
PDL = os.environ.get("ENGINE_PDL", "1") != "0" and os.environ.get("TRITON_INTERPRET") != "1"


def _install_pdl():
    from triton.runtime import driver

    # the live driver module is not importable under its package name: patch the
    # globals dict the active launcher class actually resolves make_launcher in
    g = driver.active.launcher_cls.__init__.__globals__
    orig = g["make_launcher"]
    old = "CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));"
    new = (
        "if (clusterDimX == 7) {\n"
        "  CUlaunchAttribute pdlAttr[1];\n"
        "  pdlAttr[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;\n"
        "  pdlAttr[0].value.programmaticStreamSerializationAllowed = 1;\n"
        "  CUlaunchConfig pdlCfg;\n"
        "  pdlCfg.gridDimX = gridX; pdlCfg.gridDimY = gridY; pdlCfg.gridDimZ = gridZ;\n"
        "  pdlCfg.blockDimX = 32 * num_warps; pdlCfg.blockDimY = 1; pdlCfg.blockDimZ = 1;\n"
        "  pdlCfg.sharedMemBytes = shared_memory; pdlCfg.hStream = stream;\n"
        "  pdlCfg.attrs = pdlAttr; pdlCfg.numAttrs = 1;\n"
        "  static cuLaunchKernelEx_t pdlHandle = NULL;\n"
        "  if (pdlHandle == NULL) pdlHandle = getLaunchKernelExHandle();\n"
        "  CUDA_CHECK(pdlHandle(&pdlCfg, function, params, 0));\n"
        "} else {\n"
        "  " + old + "\n}"
    )

    def patched(constants, signature, ids):
        src = orig(constants, signature, ids)
        assert old in src
        return src.replace(old, new, 1)

    g["make_launcher"] = patched


if PDL:
    try:
        _install_pdl()
    except Exception as _e:  # unknown triton layout: plain launches
        print(f"[fused] PDL launcher patch failed: {_e!r}")
        PDL = False


def _launch(fn, grid, *args, **kw):
    """fn[grid](*args, **kw), then mark the compiled kernel for PDL launches."""
    k = fn[grid](*args, **kw)
    if PDL and k is not None and k.packed_metadata[3] != _PDL_MAGIC:
        k.packed_metadata = (*k.packed_metadata[:3], _PDL_MAGIC, 1, 1)
    return k


@triton.jit
def _gdc_wait():
    tl.inline_asm_elementwise("griddepcontrol.wait;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _gdc_launch():
    tl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _pdl_probe_kernel(x_ptr, PDL: tl.constexpr):
    if PDL:
        _gdc_launch()
        _gdc_wait()
    i = tl.arange(0, 1024)
    tl.store(x_ptr + i, tl.load(x_ptr + i) + 1.0)


def _probe_pdl():
    """Chain of dependent PDL launches inside a CUDA graph must still count exactly;
    on any error or wrong result PDL is switched off (plain launches)."""
    global PDL
    if not PDL:
        return
    try:
        x = torch.zeros(1024, device="cuda")
        n = 64
        for _ in range(3):  # compile + mark
            _launch(_pdl_probe_kernel, (1,), x, PDL=True)
        torch.cuda.synchronize()
        x.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n):
                _launch(_pdl_probe_kernel, (1,), x, PDL=True)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        assert bool((x == n).all()), "eager PDL chain miscounted"
        x.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                _launch(_pdl_probe_kernel, (1,), x, PDL=True)
        for r in range(3):
            g.replay()
        torch.cuda.synchronize()
        assert bool((x == 3 * n).all()), "graph PDL chain miscounted"
    except Exception as e:
        print(f"[fused] PDL probe failed, disabling: {e!r}")
        PDL = False


@triton.jit
def _l2_prefetch(ptrs):
    tl.inline_asm_elementwise("prefetch.global.L2 [$1];", "=r,l", [ptrs], dtype=tl.int32, is_pure=False, pack=1)


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
def _head_norm_rope(x_base, w_ptr, cb, cos_ptr, sin_ptr, offs, eps,
                    HALF: tl.constexpr, HD: tl.constexpr, DT: tl.constexpr):
    x1 = tl.load(x_base + offs)
    x2 = tl.load(x_base + HALF + offs)
    f1 = x1.to(tl.float32)
    f2 = x2.to(tl.float32)
    var = (tl.sum(f1 * f1, axis=0) + tl.sum(f2 * f2, axis=0)) / HD
    r = tl.math.rsqrt(var + eps)
    n1 = (f1 * r).to(DT) * tl.load(w_ptr + offs)
    n2 = (f2 * r).to(DT) * tl.load(w_ptr + HALF + offs)
    c1 = tl.load(cos_ptr + cb + offs)
    c2 = tl.load(cos_ptr + cb + HALF + offs)
    s1 = tl.load(sin_ptr + cb + offs)
    s2 = tl.load(sin_ptr + cb + HALF + offs)
    return _rope_half(n1, n2, c1, c2, s1, s2, DT)


@triton.jit
def _qkv_post_prefill_kernel(
    qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, q_ptr, kc_ptr, vc_ptr, S, cap, eps,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, LAST_QUERY_ONLY: tl.constexpr,
):
    # one program per (token, kv head): its G q heads (qk-norm + rope -> q [T, NH, HD],
    # token-major so flash attention returns a contiguous [T, NH*HD]), the k head
    # (norm + rope -> cache) and the v head (copy -> cache).
    t = tl.program_id(0)
    kvh = tl.program_id(1)
    b = t // S
    s = t % S
    G: tl.constexpr = NH // NKV
    HALF: tl.constexpr = HD // 2
    DT: tl.constexpr = q_ptr.dtype.element_ty
    offs = tl.arange(0, HALF)
    TOTAL: tl.constexpr = (NH + 2 * NKV) * HD
    base = qkv_ptr + t.to(tl.int64) * TOTAL
    cb = s * HD
    if not LAST_QUERY_ONLY or s == S - 1:
        for g in tl.static_range(G):
            hq = kvh * G + g
            o1, o2 = _head_norm_rope(base + hq * HD, qn_ptr, cb, cos_ptr, sin_ptr, offs, eps, HALF, HD, DT)
            if LAST_QUERY_ONLY:
                qo = q_ptr + (b.to(tl.int64) * NH + hq) * HD
            else:
                qo = q_ptr + (t.to(tl.int64) * NH + hq) * HD
            tl.store(qo + offs, o1)
            tl.store(qo + HALF + offs, o2)
    o1, o2 = _head_norm_rope(base + (NH + kvh) * HD, kn_ptr, cb, cos_ptr, sin_ptr, offs, eps, HALF, HD, DT)
    ko = kc_ptr + ((b * NKV + kvh).to(tl.int64) * cap + s) * HD
    tl.store(ko + offs, o1)
    tl.store(ko + HALF + offs, o2)
    vb = base + (NH + NKV + kvh) * HD
    vo = vc_ptr + ((b * NKV + kvh).to(tl.int64) * cap + s) * HD
    tl.store(vo + offs, tl.load(vb + offs))
    tl.store(vo + HALF + offs, tl.load(vb + HALF + offs))


@triton.jit
def _qkv_post_prefill_kernel_t(
    qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, q_ptr, kc_ptr, vc_ptr, S, cap, eps,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, LAST_QUERY_ONLY: tl.constexpr,
):
    # same math as _qkv_post_prefill_kernel, one program per TOKEN (all NKV kv heads
    # looped inside): 1/NKV as many launches, NKV x the work per launch. Grid (B*S,)
    # amortizes fixed per-launch overhead over more work; identical arithmetic and
    # store order per head, so bit-for-bit the same as the per-(token,kv head) kernel.
    t = tl.program_id(0)
    b = t // S
    s = t % S
    G: tl.constexpr = NH // NKV
    HALF: tl.constexpr = HD // 2
    DT: tl.constexpr = q_ptr.dtype.element_ty
    offs = tl.arange(0, HALF)
    TOTAL: tl.constexpr = (NH + 2 * NKV) * HD
    base = qkv_ptr + t.to(tl.int64) * TOTAL
    cb = s * HD
    do_q = (not LAST_QUERY_ONLY) or s == S - 1
    for kvh in tl.static_range(NKV):
        if do_q:
            for g in tl.static_range(G):
                hq = kvh * G + g
                o1, o2 = _head_norm_rope(base + hq * HD, qn_ptr, cb, cos_ptr, sin_ptr, offs, eps, HALF, HD, DT)
                if LAST_QUERY_ONLY:
                    qo = q_ptr + (b.to(tl.int64) * NH + hq) * HD
                else:
                    qo = q_ptr + (t.to(tl.int64) * NH + hq) * HD
                tl.store(qo + offs, o1)
                tl.store(qo + HALF + offs, o2)
        o1, o2 = _head_norm_rope(base + (NH + kvh) * HD, kn_ptr, cb, cos_ptr, sin_ptr, offs, eps, HALF, HD, DT)
        ko = kc_ptr + ((b * NKV + kvh).to(tl.int64) * cap + s) * HD
        tl.store(ko + offs, o1)
        tl.store(ko + HALF + offs, o2)
        vb = base + (NH + NKV + kvh) * HD
        vo = vc_ptr + ((b * NKV + kvh).to(tl.int64) * cap + s) * HD
        tl.store(vo + offs, tl.load(vb + offs))
        tl.store(vo + HALF + offs, tl.load(vb + HALF + offs))


@triton.jit
def _flash_prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, S, cap, qk_scale,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    # causal GQA flash-attention forward for prefill. q/o token-major [B, S, NH, HD]
    # (what _qkv_post_prefill_kernel writes / what the epilogue GEMV expects), k/v
    # cache [B, NKV, cap, HD]. Heaviest (longest causal prefix) query blocks are
    # scheduled first to even out SM occupancy across the wave.
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    num_m = tl.num_programs(0)
    m_blk = num_m - 1 - pid_m
    b = bh // NH
    h = bh % NH
    kvh = h // (NH // NKV)
    rm = m_blk * BM + tl.arange(0, BM)
    d = tl.arange(0, HD)
    rmask = rm < S
    q_off = ((b * S + rm[:, None]).to(tl.int64) * NH + h) * HD + d[None, :]
    q = tl.load(q_ptr + q_off, mask=rmask[:, None], other=0.0)
    kv_base = (b * NKV + kvh).to(tl.int64) * cap * HD
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    hi = tl.minimum((m_blk + 1) * BM, S)
    for n0 in range(0, hi, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < S
        k = tl.load(k_ptr + kv_base + rn[:, None] * HD + d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        valid = (rm[:, None] >= rn[None, :]) & nmask[None, :]
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + kv_base + rn[:, None] * HD + d[None, :], mask=nmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + q_off, o.to(o_ptr.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _qkv_post_prefill_kernel_v(
    qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, q_ptr, kc_ptr, vc_ptr, S, cap, eps,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, LAST_QUERY_ONLY: tl.constexpr,
):
    # one program per token; Q (NH heads), K and V (NKV heads each) are each handled
    # as one 2D [heads, HALF] tile per rotation half, instead of one small load per
    # head sequentially -- wide vectorized loads/stores give the hardware more
    # independent memory ops in flight per thread. Same RMS/rope arithmetic and bf16
    # rounding points as _qkv_post_prefill_kernel, just batched across heads; store
    # order does not affect the result (every head's address is independent), so
    # this is bit-for-bit identical.
    t = tl.program_id(0)
    b = t // S
    s = t % S
    HALF: tl.constexpr = HD // 2
    DT: tl.constexpr = q_ptr.dtype.element_ty
    offs = tl.arange(0, HALF)
    TOTAL: tl.constexpr = (NH + 2 * NKV) * HD
    base = qkv_ptr + t.to(tl.int64) * TOTAL
    cb = s * HD
    c1 = tl.load(cos_ptr + cb + offs)
    c2 = tl.load(cos_ptr + cb + HALF + offs)
    sn1 = tl.load(sin_ptr + cb + offs)
    sn2 = tl.load(sin_ptr + cb + HALF + offs)

    if (not LAST_QUERY_ONLY) or s == S - 1:
        hq = tl.arange(0, NH)
        x1 = tl.load(base + hq[:, None].to(tl.int64) * HD + offs[None, :])
        x2 = tl.load(base + hq[:, None].to(tl.int64) * HD + HALF + offs[None, :])
        f1 = x1.to(tl.float32)
        f2 = x2.to(tl.float32)
        var = (tl.sum(f1 * f1, axis=1) + tl.sum(f2 * f2, axis=1)) / HD
        r = tl.math.rsqrt(var + eps)
        qw1 = tl.load(qn_ptr + offs)
        qw2 = tl.load(qn_ptr + HALF + offs)
        n1 = (f1 * r[:, None]).to(DT) * qw1[None, :]
        n2 = (f2 * r[:, None]).to(DT) * qw2[None, :]
        o1, o2 = _rope_half(n1, n2, c1[None, :], c2[None, :], sn1[None, :], sn2[None, :], DT)
        if LAST_QUERY_ONLY:
            qo = q_ptr + b.to(tl.int64) * NH * HD
        else:
            qo = q_ptr + t.to(tl.int64) * NH * HD
        tl.store(qo + hq[:, None].to(tl.int64) * HD + offs[None, :], o1)
        tl.store(qo + hq[:, None].to(tl.int64) * HD + HALF + offs[None, :], o2)

    hk = tl.arange(0, NKV)
    xk1 = tl.load(base + (NH + hk)[:, None].to(tl.int64) * HD + offs[None, :])
    xk2 = tl.load(base + (NH + hk)[:, None].to(tl.int64) * HD + HALF + offs[None, :])
    fk1 = xk1.to(tl.float32)
    fk2 = xk2.to(tl.float32)
    vark = (tl.sum(fk1 * fk1, axis=1) + tl.sum(fk2 * fk2, axis=1)) / HD
    rk = tl.math.rsqrt(vark + eps)
    kw1 = tl.load(kn_ptr + offs)
    kw2 = tl.load(kn_ptr + HALF + offs)
    nk1 = (fk1 * rk[:, None]).to(DT) * kw1[None, :]
    nk2 = (fk2 * rk[:, None]).to(DT) * kw2[None, :]
    ok1, ok2 = _rope_half(nk1, nk2, c1[None, :], c2[None, :], sn1[None, :], sn2[None, :], DT)
    ko = kc_ptr + (b * NKV + hk).to(tl.int64) * cap * HD + s * HD
    tl.store(ko[:, None] + offs[None, :], ok1)
    tl.store(ko[:, None] + HALF + offs[None, :], ok2)

    xv1 = tl.load(base + (NH + NKV + hk)[:, None].to(tl.int64) * HD + offs[None, :])
    xv2 = tl.load(base + (NH + NKV + hk)[:, None].to(tl.int64) * HD + HALF + offs[None, :])
    vo = vc_ptr + (b * NKV + hk).to(tl.int64) * cap * HD + s * HD
    tl.store(vo[:, None] + offs[None, :], xv1)
    tl.store(vo[:, None] + HALF + offs[None, :], xv2)


@triton.jit
def _qkv_post_kernel(
    qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_ptr, kc_ptr, vc_ptr, S, cap, eps,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, DECODE: tl.constexpr,
    POS_STRIDE: tl.constexpr, PDL: tl.constexpr = False, TRIG: tl.constexpr = 0,
):
    # one program per (token, head): q/k get qk-norm + rope, v is copied; k/v go
    # straight into the static cache, q into [B, NH, S, HD].
    if PDL:
        if TRIG < 2:
            _gdc_launch()
        _gdc_wait()
    t = tl.program_id(0)
    hidx = tl.program_id(1)
    b = t // S
    s = t % S
    DT: tl.constexpr = q_ptr.dtype.element_ty
    HALF: tl.constexpr = HD // 2
    offs = tl.arange(0, HALF)
    TOTAL: tl.constexpr = (NH + 2 * NKV) * HD
    base = qkv_ptr + t * TOTAL + hidx * HD
    G: tl.constexpr = NH // NKV
    if DECODE:
        position = tl.load(pos_ptr + b * POS_STRIDE) + s
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
            if DECODE:  # [B, NKV, S, G, HD]: rows (s, g) contiguous per kv head
                qo = q_ptr + ((((b * NKV + hidx // G) * S + s) * G + hidx % G)) * HD
            else:  # [B, NH, S, HD]
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
    NSPLIT: tl.constexpr, G: tl.constexpr, W: tl.constexpr, GP: tl.constexpr,
    HD: tl.constexpr, BLOCK_N: tl.constexpr, NKV: tl.constexpr, POS_STRIDE: tl.constexpr,
    PDL: tl.constexpr = False, TRIG: tl.constexpr = 0,
):
    if PDL:
        if TRIG < 2:
            _gdc_launch()
        _gdc_wait()
    # flash-decoding partial: one program per (batch*kv_head, split). The W query
    # tokens x G query heads sharing a kv head are the M rows of the dot (padded to
    # GP >= 16); row r = s*G + g sees keys <= pos + s (causal among the W tokens).
    bk = tl.program_id(0)
    sp = tl.program_id(1)
    b = bk // NKV
    p0 = tl.load(pos_ptr + b * POS_STRIDE)
    L = p0 + W
    chunk = (L + NSPLIT - 1) // NSPLIT
    start = sp * chunk
    end = tl.minimum(start + chunk, L)
    rows = tl.arange(0, GP)
    d = tl.arange(0, HD)
    rmask = rows < W * G
    lim = p0 + rows // G + 1
    q = tl.load(
        q_ptr + (bk * (W * G) + rows[:, None]) * HD + d[None, :], mask=rmask[:, None], other=0.0
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
        valid = (n[None, :] < lim[:, None]) & nm[None, :] & rmask[:, None]
        sc = tl.where(valid, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + kv_base + n[:, None] * HD + d[None, :], mask=nm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    if NSPLIT == 1:
        kvh = bk % NKV
        ob = ((b * W + rows // G) * (NKV * G) + kvh * G + rows % G) * HD
        tl.store(out_ptr + ob[:, None] + d[None, :],
                 (acc / l_i[:, None]).to(out_ptr.dtype.element_ty),
                 mask=rmask[:, None])
    else:
        wb = ((bk * (W * G) + rows) * NSPLIT + sp) * (HD + 2)
        tl.store(ws_ptr + wb[:, None] + d[None, :], acc, mask=rmask[:, None])
        tl.store(ws_ptr + wb + HD, m_i, mask=rmask)
        tl.store(ws_ptr + wb + HD + 1, l_i, mask=rmask)


@triton.jit
def _attn_combine_kernel(
    ws_ptr, out_ptr, NSPLIT: tl.constexpr, SP: tl.constexpr, HD: tl.constexpr,
    NKV: tl.constexpr, G: tl.constexpr, W: tl.constexpr, PDL: tl.constexpr = False, TRIG: tl.constexpr = 0,
):
    if PDL:
        if TRIG < 2:
            _gdc_launch()
        _gdc_wait()
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
    rl = r % (W * G)
    bk = r // (W * G)
    tok = rl // G
    g = rl % G
    out_row = ((bk // NKV) * W + tok) * NKV + bk % NKV  # (b*W + tok)*NKV + kvh
    tl.store(out_ptr + (out_row * G + g) * HD + d, o.to(out_ptr.dtype.element_ty))


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, out_ptr, M, N, K, kps, stride_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FINAL: tl.constexpr,
    CM: tl.constexpr = "", EV: tl.constexpr = "", PDL: tl.constexpr = False, PF: tl.constexpr = 16,
    TRIG: tl.constexpr = 0, EVENK: tl.constexpr = False,
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
    if PDL:
        # weights don't depend on the previous kernel: start pulling the head of this
        # program's slice into L2 while that kernel drains, then wait for x.
        if TRIG == 0:
            _gdc_launch()
        if PF > 0:
            pk = tl.minimum(k_lo + tl.arange(0, PF) * 64, k_hi - 1)
            _l2_prefetch(w_ptr + tl.minimum(rn, N - 1)[:, None].to(tl.int64) * K + pk[None, :])
        _gdc_wait()
    acc = tl.zeros([BM, BN], tl.float32)
    if EVENK:
        # N % BN == 0 and every K split is a whole number of BK tiles: no K/N masks, so the
        # loads along K can be fully vectorised (16-byte cp.async).
        for k in range(k_lo, k_lo + kps, BK):
            rk = tl.max_contiguous(tl.multiple_of(k + tl.arange(0, BK), BK), BK)
            x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
            w = tl.load(w_ptr + rn[None, :] * K + rk[:, None], cache_modifier=CM, eviction_policy=EV)
            acc = tl.dot(x, w, acc)
    else:
        for k in range(k_lo, k_hi, BK):
            rk = k + tl.arange(0, BK)
            km = rk < k_hi
            x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None] & km[None, :], other=0.0)
            w = tl.load(w_ptr + rn[None, :] * K + rk[:, None], mask=nm[None, :] & km[:, None], other=0.0,
                        cache_modifier=CM, eviction_policy=EV)
            acc = tl.dot(x, w, acc)
    if PDL and TRIG > 0:
        _gdc_launch()
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


@triton.jit
def _gemv_silu_kernel(
    x_ptr, w_ptr, out_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    CM: tl.constexpr = "", EV: tl.constexpr = "", PDL: tl.constexpr = False, PF: tl.constexpr = 16,
    TRIG: tl.constexpr = 0, EVENK: tl.constexpr = False,
):
    # w is [2N, K] = gate rows then up rows: out[:, n] = silu(x@wg_n) * (x@wu_n),
    # rounded to bf16 at the same points as silu_mul(linear(x, w)).
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    mm = rm < M
    nm = rn < N
    if PDL:
        if TRIG == 0:
            _gdc_launch()
        if PF > 0:
            pk = tl.arange(0, PF) * 64
            rc = tl.minimum(rn, N - 1)[:, None].to(tl.int64)
            _l2_prefetch(w_ptr + rc * K + pk[None, :])
            _l2_prefetch(w_ptr + (rc + N) * K + pk[None, :])
        _gdc_wait()
    accg = tl.zeros([BM, BN], tl.float32)
    accu = tl.zeros([BM, BN], tl.float32)
    if EVENK:
        for k in range(0, K, BK):
            rk = tl.max_contiguous(tl.multiple_of(k + tl.arange(0, BK), BK), BK)
            x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
            wg = tl.load(w_ptr + rn[None, :] * K + rk[:, None], cache_modifier=CM, eviction_policy=EV)
            wu = tl.load(w_ptr + (rn[None, :] + N) * K + rk[:, None], cache_modifier=CM, eviction_policy=EV)
            accg = tl.dot(x, wg, accg)
            accu = tl.dot(x, wu, accu)
    else:
        for k in range(0, K, BK):
            rk = k + tl.arange(0, BK)
            km = rk < K
            x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None] & km[None, :], other=0.0)
            wg = tl.load(w_ptr + rn[None, :] * K + rk[:, None], mask=nm[None, :] & km[:, None], other=0.0,
                         cache_modifier=CM, eviction_policy=EV)
            wu = tl.load(w_ptr + (rn[None, :] + N) * K + rk[:, None], mask=nm[None, :] & km[:, None], other=0.0,
                         cache_modifier=CM, eviction_policy=EV)
            accg = tl.dot(x, wg, accg)
            accu = tl.dot(x, wu, accu)
        if PDL and TRIG > 0:
            _gdc_launch()
    dt = out_ptr.dtype.element_ty
    g = accg.to(dt).to(tl.float32)
    u = accu.to(dt).to(tl.float32)
    sg = (g / (1.0 + tl.exp(-g))).to(dt)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], (sg.to(tl.float32) * u).to(dt),
             mask=mm[:, None] & nm[None, :])


@triton.jit
def _reduce_add_rms_kernel(
    ws_ptr, h_ptr, w_ptr, hn_ptr, y_ptr, M, n_cols, eps,
    SK: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False, TRIG: tl.constexpr = 0,
):
    # one program per row: hn = h + bf16(sum_splits ws); y = rmsnorm(hn) * w
    if PDL:
        if TRIG < 2:
            _gdc_launch()
        _gdc_wait()
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    acc = tl.zeros([BLOCK], tl.float32)
    for i in range(SK):
        acc += tl.load(ws_ptr + (i * M + row) * n_cols + cols, mask=m, other=0.0)
    hv = tl.load(h_ptr + row * n_cols + cols, mask=m, other=0.0)
    dt = hv.dtype
    hn = (hv.to(tl.float32) + acc.to(dt).to(tl.float32)).to(dt)
    tl.store(hn_ptr + row * n_cols + cols, hn, mask=m)
    xf = hn.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / n_cols
    normed = xf * tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, normed.to(dt) * w, mask=m)

# (N, K) -> (BN, BK, SPLIT_K, stages, warps), swept on H100 (tests/gemv_bench.py)
GEMM_CFG = {
    (6144, 2560): (64, 128, 1, 5, 4),
    (2560, 4096): (64, 256, 2, 4, 4),
    (19456, 2560): (64, 64, 1, 4, 4),
    (2560, 9728): (32, 128, 4, 4, 4),
    (151936, 2560): (64, 256, 1, 3, 4),
}


# gate/up + silu epilogue tile configs: (BN, BK, stages, warps)
SILU_CFG = {(9728, 2560): (32, 128, 3, 2)}


EVEN_K = os.environ.get("ENGINE_EVENK", "1") != "0"  # mask-free GEMV when the shape divides evenly
ATTN_TARGET = int(os.environ.get("ENGINE_ATTN_TARGET", "256"))
ATTN_TARGET_BIG = int(os.environ.get("ENGINE_ATTN_TARGET_BIG", "128"))
ATTN_ST = int(os.environ.get("ENGINE_ATTN_ST", "3"))
PF = int(os.environ.get("ENGINE_PF", "4"))
TRIG = int(os.environ.get("ENGINE_TRIG", "1"))
# Largest M the fused GEMV path will take. Default 16 = today's behaviour exactly (above it we
# fall back to cuBLAS, which costs the whole fused path: no Triton, no split-K, no PDL, and the
# silu/add-rmsnorm epilogues decompose). The kernels are already generic in BM (`mm = rm < M`
# masks), so raising this is a tile-tuning problem, not a correctness one. Hidden workloads run
# shapes the 3 public ones (all B<=16) never exercise. Tune GEMM_CFG/SILU_CFG per BM before use.
BM_MAX = int(os.environ.get("ENGINE_BM_MAX", "32"))


def _bm(M):
    """Tile rows for a given batch: >=16 (tl.dot minimum), power of 2, never below M."""
    return max(16, triton.next_power_of_2(M))

# Triton causal flash-attention for prefill (token-major q, exp2 softmax): env-gated,
# default off until the in-engine A/B clears the bar (SHARED-CONTEXT rule 6).
FLASH_PREFILL = os.environ.get("ENGINE_FLASH_PREFILL", "1") == "1"
_LOG2E = 1.4426950408889634
_flash_cfg_override = os.environ.get("ENGINE_FLASH_CFG")  # "BM,BN,NW,ST" for sweeps
FLASH_CFG_DEFAULT = (
    tuple(int(x) for x in _flash_cfg_override.split(",")) if _flash_cfg_override else (128, 128, 8, 3)
)
# (S bucket upper bound) -> (BM, BN, num_warps, num_stages); swept in-engine on H100
FLASH_CFG = {}

# _qkv_post_prefill_kernel warp count: one program per (token, kv head), G=NH/NKV
# query heads processed via a static loop; sweep to find the roofline-friendly count.
QKV_PREFILL_WARPS = int(os.environ.get("ENGINE_QKV_PREFILL_WARPS", "2"))
QKV_PREFILL_V2 = os.environ.get("ENGINE_QKV_PREFILL_V2", "0") == "1"  # 1 program/token vs 1/(token,kv head)
QKV_PREFILL_MODE = os.environ.get("ENGINE_QKV_PREFILL_MODE", "vec")  # orig | tok | vec
# _add_rms_kernel / _silu_mul_kernel: both prefill-only in practice at M=8192 (decode
# always M<=16 and uses the GEMV-fused split-K reduce / gemv-silu kernels instead).
RMS_WARPS = int(os.environ.get("ENGINE_RMS_WARPS", "8"))
SILU_BLOCK = int(os.environ.get("ENGINE_SILU_BLOCK", "1024"))
SILU_WARPS = int(os.environ.get("ENGINE_SILU_WARPS", "4"))


def _flash_cfg(S):
    if _flash_cfg_override:
        return FLASH_CFG_DEFAULT
    for bound, cfg in sorted(FLASH_CFG.items()):
        if S <= bound:
            return cfg
    return FLASH_CFG_DEFAULT


class TritonOps:
    def __init__(self, e, use_gemv=True):
        self.e = e
        self.use_gemv = use_gemv
        assert e.hd & (e.hd - 1) == 0 and e.hd >= 2
        _probe_pdl()
        self.dummy_pos = torch.zeros(1, dtype=torch.long, device=e.dev)

    def _rms_launch(self, h, d, w, has_add):
        rows, cols = h.shape
        y = torch.empty_like(h)
        hn = torch.empty_like(h) if has_add else h
        _add_rms_kernel[(rows,)](
            h, d if has_add else h, w, hn, y, cols, self.e.eps,
            HAS_ADD=has_add, BLOCK=triton.next_power_of_2(cols), num_warps=RMS_WARPS,
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
        _silu_mul_kernel[(triton.cdiv(n, SILU_BLOCK),)](gu, out, n, inter, BLOCK=SILU_BLOCK, num_warps=SILU_WARPS)
        return out

    def qkv_post(self, qkv, l, kc, vc, B, S, pos, last_query_only=False):
        e = self.e
        if pos is None:  # prefill: q token-major [B, S, nh, hd], returned as a [B, nh, S, hd] view
            q = torch.empty((B, 1 if last_query_only else S, e.nh, e.hd), dtype=qkv.dtype, device=qkv.device)
            if QKV_PREFILL_MODE == "vec":
                _qkv_post_prefill_kernel_v[(B * S,)](
                    qkv.contiguous(), l.qn, l.kn, e.cos, e.sin, q, kc, vc, S, kc.shape[2], e.eps,
                    NH=e.nh, NKV=e.nkv, HD=e.hd, LAST_QUERY_ONLY=last_query_only, num_warps=QKV_PREFILL_WARPS,
                )
            elif QKV_PREFILL_MODE == "tok":
                _qkv_post_prefill_kernel_t[(B * S,)](
                    qkv.contiguous(), l.qn, l.kn, e.cos, e.sin, q, kc, vc, S, kc.shape[2], e.eps,
                    NH=e.nh, NKV=e.nkv, HD=e.hd, LAST_QUERY_ONLY=last_query_only, num_warps=QKV_PREFILL_WARPS,
                )
            else:
                _qkv_post_prefill_kernel[(B * S, e.nkv)](
                    qkv.contiguous(), l.qn, l.kn, e.cos, e.sin, q, kc, vc, S, kc.shape[2], e.eps,
                    NH=e.nh, NKV=e.nkv, HD=e.hd, LAST_QUERY_ONLY=last_query_only, num_warps=QKV_PREFILL_WARPS,
                )
            return q.transpose(1, 2)
        q = torch.empty((B, e.nh, S, e.hd), dtype=qkv.dtype, device=qkv.device)
        _launch(
            _qkv_post_kernel, (B * S, e.nh + 2 * e.nkv),
            qkv.contiguous(), l.qn, l.kn, e.cos, e.sin,
            pos if pos is not None else self.dummy_pos,
            q, kc, vc, S, kc.shape[2], e.eps,
            NH=e.nh, NKV=e.nkv, HD=e.hd, DECODE=pos is not None,
            POS_STRIDE=1 if (pos is not None and pos.numel() > 1) else 0, num_warps=1, PDL=PDL, TRIG=TRIG,
        )
        return q

    def attn_prefill(self, q, kc, vc, S, last_query_only=False):
        """q: [B, nh, S', hd] (from qkv_post; S'=1 iff last_query_only), a transposed
        view over a token-major [B, S', nh, hd] buffer. -> [B*S', nh*hd]."""
        e = self.e
        B = q.shape[0]
        if last_query_only or not FLASH_PREFILL:
            return torch.nn.functional.scaled_dot_product_attention(
                q, kc[:, :, :S], vc[:, :, :S], is_causal=not last_query_only, enable_gqa=True
            ).transpose(1, 2).reshape(-1, e.nh * e.hd)
        qtm = q.transpose(1, 2)  # recover the contiguous [B, S, nh, hd] token-major buffer
        out = torch.empty_like(qtm)
        BM, BN, NW, ST = _flash_cfg(S)
        _flash_prefill_kernel[(triton.cdiv(S, BM), B * e.nh)](
            qtm, kc, vc, out, S, kc.shape[2], e.hd ** -0.5 * _LOG2E,
            NH=e.nh, NKV=e.nkv, HD=e.hd, BM=BM, BN=BN, num_warps=NW, num_stages=ST,
        )
        return out.reshape(B * S, e.nh * e.hd)

    def attn_decode(self, q, kc, vc, pos, W=1):
        """q: [B, nkv, W, G, hd] (from qkv_post). Token j of seq b attends to
        kc/vc[b, :, :pos[b]+j+1]. -> [B*W, nh*hd]"""
        e = self.e
        G = e.nh // e.nkv
        B = q.shape[0]
        bk = B * e.nkv
        nsplit = 1
        target = ATTN_TARGET_BIG if bk >= 128 else ATTN_TARGET  # big grids: one wave, nsplit=1 skips the combine
        while bk * nsplit < target and nsplit < 32:
            nsplit *= 2
        out = torch.empty((B * W, e.nh * e.hd), dtype=q.dtype, device=q.device)
        ws = (out if nsplit == 1 else
              torch.empty((bk * W * G, nsplit, e.hd + 2), dtype=torch.float32, device=q.device))
        _launch(
            _attn_split_kernel, (bk, nsplit),
            q, kc, vc, pos, ws, out, kc.shape[2], e.hd ** -0.5,
            NSPLIT=nsplit, G=G, W=W, GP=max(16, triton.next_power_of_2(W * G)), HD=e.hd,
            BLOCK_N=64, NKV=e.nkv, POS_STRIDE=1 if pos.numel() > 1 else 0,
            num_warps=4, num_stages=ATTN_ST, PDL=PDL, TRIG=TRIG,
        )
        if nsplit > 1:
            _launch(
                _attn_combine_kernel, (bk * W * G,),
                ws, out, NSPLIT=nsplit, SP=nsplit, HD=e.hd, NKV=e.nkv, G=G, W=W, num_warps=1, PDL=PDL, TRIG=TRIG,
            )
        return out

    def linear(self, x, w):
        M, K = x.shape
        N = w.shape[0]
        cfg = GEMM_CFG.get((N, K))
        if (not self.use_gemv or cfg is None or M > BM_MAX or not x.is_contiguous()
                or not w.is_contiguous() or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16):
            return torch.nn.functional.linear(x, w)
        BN, BK, SK, ST, NW = cfg
        kps = triton.cdiv(triton.cdiv(K, SK), BK) * BK
        even = EVEN_K and N % BN == 0 and kps * SK == K
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        if SK == 1:
            _launch(
                _gemv_kernel, (triton.cdiv(N, BN), 1),
                x, w, out, M, N, K, kps, N,
                BM=_bm(M), BN=BN, BK=BK, FINAL=True, num_warps=NW, num_stages=ST, PDL=PDL, PF=PF, TRIG=TRIG, EVENK=even,
            )
            return out
        ws = torch.empty((SK, M, N), dtype=torch.float32, device=x.device)
        _launch(
            _gemv_kernel, (triton.cdiv(N, BN), SK),
            x, w, ws, M, N, K, kps, N,
            BM=_bm(M), BN=BN, BK=BK, FINAL=False, num_warps=NW, num_stages=ST, PDL=PDL, PF=PF, TRIG=TRIG, EVENK=even,
        )
        _splitk_reduce_kernel[(triton.cdiv(M * N, 1024),)](ws, out, M * N, SK=SK, BLOCK=1024)
        return out

    def gate_up_silu(self, x, wgu):
        """silu(g) * u for [g; u] = x @ wgu^T"""
        M, K = x.shape
        I = wgu.shape[0] // 2
        cfg = SILU_CFG.get((I, K))
        if (not self.use_gemv or cfg is None or M > BM_MAX or not x.is_contiguous()
                or not wgu.is_contiguous() or x.dtype != torch.bfloat16 or wgu.dtype != torch.bfloat16):
            return self.silu_mul(self.linear(x, wgu))
        BN, BK, ST, NW = cfg
        even = EVEN_K and I % BN == 0 and K % BK == 0
        out = torch.empty((M, I), dtype=x.dtype, device=x.device)
        _launch(
            _gemv_silu_kernel, (triton.cdiv(I, BN),),
            x, wgu, out, M, I, K, BM=_bm(M), BN=BN, BK=BK, num_warps=NW, num_stages=ST, PDL=PDL, PF=PF, TRIG=TRIG,
            EVENK=even,
        )
        return out

    def linear_add_norm(self, x, w, h, lnw):
        """hn = h + x @ w^T ; returns (hn, rmsnorm(hn, lnw))"""
        M, K = x.shape
        N = w.shape[0]
        cfg = GEMM_CFG.get((N, K))
        if (not self.use_gemv or cfg is None or cfg[2] == 1 or M > BM_MAX
                or not x.is_contiguous() or not w.is_contiguous() or not h.is_contiguous()
                or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or h.dtype != torch.bfloat16):
            return self.add_rms(h, self.linear(x, w), lnw)
        BN, BK, SK, ST, NW = cfg
        kps = triton.cdiv(triton.cdiv(K, SK), BK) * BK
        even = EVEN_K and N % BN == 0 and kps * SK == K
        ws = torch.empty((SK, M, N), dtype=torch.float32, device=x.device)
        _launch(
            _gemv_kernel, (triton.cdiv(N, BN), SK),
            x, w, ws, M, N, K, kps, N,
            BM=_bm(M), BN=BN, BK=BK, FINAL=False, num_warps=NW, num_stages=ST, PDL=PDL, PF=PF, TRIG=TRIG, EVENK=even,
        )
        hn = torch.empty_like(h)
        y = torch.empty_like(h)
        _launch(
            _reduce_add_rms_kernel, (M,),
            ws, h, lnw, hn, y, M, N, self.e.eps,
            SK=SK, BLOCK=triton.next_power_of_2(N), num_warps=8 if SK >= 4 else 4, PDL=PDL, TRIG=TRIG,
        )
        return hn, y
