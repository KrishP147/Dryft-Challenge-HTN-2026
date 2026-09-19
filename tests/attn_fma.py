"""FMA (no tensor core) split-KV decode attention vs the tl.dot kernel. W=1, G=4 (Qwen3-4B GQA)."""
import itertools, os, sys
import torch, triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused

NH, NKV, HD, G = 32, 8, 128, 4
bf = torch.bfloat16


@triton.jit
def _fma_step(k, v, q, m, l, acc, scale):
    s = tl.sum(k * q[None, :], axis=1) * scale                # [BN] scores for one query head
    m_new = tl.maximum(m, tl.max(s, 0))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp(m - m_safe)
    p = tl.exp(s - m_safe)                                     # -inf -> 0
    l = l * alpha + tl.sum(p, 0)
    pv = tl.sum(p.to(tl.bfloat16).to(tl.float32)[:, None] * v, axis=0)   # P rounded to bf16 like flash
    return m_new, l, acc * alpha + pv


@triton.jit
def _attn_split_fma_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, ws_ptr, cap, sm_scale,
    NSPLIT: tl.constexpr, HD: tl.constexpr, BLOCK_N: tl.constexpr, NKV: tl.constexpr,
    POS_STRIDE: tl.constexpr,
):
    bk = tl.program_id(0)
    sp = tl.program_id(1)
    b = bk // NKV
    L = tl.load(pos_ptr + b * POS_STRIDE) + 1
    chunk = (L + NSPLIT - 1) // NSPLIT
    start = sp * chunk
    end = tl.minimum(start + chunk, L)
    d = tl.arange(0, HD)
    q0 = tl.load(q_ptr + (bk * 4 + 0) * HD + d).to(tl.float32)
    q1 = tl.load(q_ptr + (bk * 4 + 1) * HD + d).to(tl.float32)
    q2 = tl.load(q_ptr + (bk * 4 + 2) * HD + d).to(tl.float32)
    q3 = tl.load(q_ptr + (bk * 4 + 3) * HD + d).to(tl.float32)
    ninf = float("-inf")
    m0 = ninf; m1 = ninf; m2 = ninf; m3 = ninf
    l0 = 0.0; l1 = 0.0; l2 = 0.0; l3 = 0.0
    a0 = tl.zeros([HD], tl.float32); a1 = tl.zeros([HD], tl.float32)
    a2 = tl.zeros([HD], tl.float32); a3 = tl.zeros([HD], tl.float32)
    kv_base = bk.to(tl.int64) * cap * HD
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        offs = kv_base + n[:, None] * HD + d[None, :]
        k = tl.load(k_ptr + offs, mask=nm[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offs, mask=nm[:, None], other=0.0).to(tl.float32)
        # masked keys: force score -inf by adding -inf through q*k? use explicit mask on scores instead
        km = tl.where(nm, 0.0, ninf)
        s0 = tl.sum(k * q0[None, :], axis=1) * sm_scale + km
        s1 = tl.sum(k * q1[None, :], axis=1) * sm_scale + km
        s2 = tl.sum(k * q2[None, :], axis=1) * sm_scale + km
        s3 = tl.sum(k * q3[None, :], axis=1) * sm_scale + km
        mn0 = tl.maximum(m0, tl.max(s0, 0)); mn1 = tl.maximum(m1, tl.max(s1, 0))
        mn2 = tl.maximum(m2, tl.max(s2, 0)); mn3 = tl.maximum(m3, tl.max(s3, 0))
        al0 = tl.exp(m0 - mn0); al1 = tl.exp(m1 - mn1); al2 = tl.exp(m2 - mn2); al3 = tl.exp(m3 - mn3)
        p0 = tl.exp(s0 - mn0); p1 = tl.exp(s1 - mn1); p2 = tl.exp(s2 - mn2); p3 = tl.exp(s3 - mn3)
        l0 = l0 * al0 + tl.sum(p0, 0); l1 = l1 * al1 + tl.sum(p1, 0)
        l2 = l2 * al2 + tl.sum(p2, 0); l3 = l3 * al3 + tl.sum(p3, 0)
        a0 = a0 * al0 + tl.sum(p0.to(tl.bfloat16).to(tl.float32)[:, None] * v, axis=0)
        a1 = a1 * al1 + tl.sum(p1.to(tl.bfloat16).to(tl.float32)[:, None] * v, axis=0)
        a2 = a2 * al2 + tl.sum(p2.to(tl.bfloat16).to(tl.float32)[:, None] * v, axis=0)
        a3 = a3 * al3 + tl.sum(p3.to(tl.bfloat16).to(tl.float32)[:, None] * v, axis=0)
        m0 = mn0; m1 = mn1; m2 = mn2; m3 = mn3
    wb0 = ((bk * 4 + 0) * NSPLIT + sp) * (HD + 2)
    wb1 = ((bk * 4 + 1) * NSPLIT + sp) * (HD + 2)
    wb2 = ((bk * 4 + 2) * NSPLIT + sp) * (HD + 2)
    wb3 = ((bk * 4 + 3) * NSPLIT + sp) * (HD + 2)
    tl.store(ws_ptr + wb0 + d, a0); tl.store(ws_ptr + wb0 + HD, m0); tl.store(ws_ptr + wb0 + HD + 1, l0)
    tl.store(ws_ptr + wb1 + d, a1); tl.store(ws_ptr + wb1 + HD, m1); tl.store(ws_ptr + wb1 + HD + 1, l1)
    tl.store(ws_ptr + wb2 + d, a2); tl.store(ws_ptr + wb2 + HD, m2); tl.store(ws_ptr + wb2 + HD + 1, l2)
    tl.store(ws_ptr + wb3 + d, a3); tl.store(ws_ptr + wb3 + HD, m3); tl.store(ws_ptr + wb3 + HD + 1, l3)


def t(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(2): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 5 * 1e3


for B, L, cap in ((1, 544, 640), (4, 2080, 2176), (16, 640, 640)):
    NL = 12 if B * cap > 20000 else 36
    kcs = [torch.randn(B, NKV, cap, HD, device="cuda", dtype=bf) for _ in range(NL)]
    vcs = [torch.randn_like(kcs[0]) for _ in range(NL)]
    q = torch.randn(B, NKV, 1, G, HD, device="cuda", dtype=bf); pos = torch.tensor([L - 1], device="cuda")
    bk = B * NKV; gb = 2 * B * NKV * L * HD * 2 / 1e9
    out = torch.empty((B, NH * HD), dtype=bf, device="cuda")
    # reference: current dot kernel (default nsplit rule)
    ns = 1
    while bk * ns < 256 and ns < 32: ns *= 2
    ws_ref = torch.empty((bk * G, ns, HD + 2), dtype=torch.float32, device="cuda")
    def ref_fn():
        for i in range(NL):
            fused._attn_split_kernel[(bk, ns)](q, kcs[i], vcs[i], pos, ws_ref, out, cap, HD ** -0.5, NSPLIT=ns, G=G, W=1, GP=16, HD=HD, BLOCK_N=64, NKV=NKV, POS_STRIDE=0, num_warps=4, num_stages=2)
            fused._attn_combine_kernel[(bk * G,)](ws_ref, out, NSPLIT=ns, SP=ns, HD=HD, NKV=NKV, G=G, W=1, num_warps=1)
    tref = t(ref_fn) / NL
    ref_fn(); ref_out = out.clone()
    res = []
    for nsplit, BN, NW in itertools.product((1, 2, 4, 8, 16, 32), (16, 32, 64), (1, 2, 4, 8)):
        ws = torch.empty((bk * G, nsplit, HD + 2), dtype=torch.float32, device="cuda")
        def fn():
            for i in range(NL):
                _attn_split_fma_kernel[(bk, nsplit)](q, kcs[i], vcs[i], pos, ws, cap, HD ** -0.5, NSPLIT=nsplit, HD=HD, BLOCK_N=BN, NKV=NKV, POS_STRIDE=0, num_warps=NW)
                fused._attn_combine_kernel[(bk * G,)](ws, out, NSPLIT=nsplit, SP=nsplit, HD=HD, NKV=NKV, G=G, W=1, num_warps=1)
        try:
            fn(); torch.cuda.synchronize()
            d = (out.float() - ref_out.float()).abs().max().item()
            if d > 0.02: continue
            res.append((t(fn) / NL, (nsplit, BN, NW), d))
        except Exception as ex:
            continue
    res.sort()
    print(f"B{B} L{L}: dot-kernel {tref:.1f} us ({gb/tref*1e6:.0f} GB/s) | fma best", [(round(a, 1), c, round(dd, 4)) for a, c, dd in res[:3]],
          f"-> {gb/res[0][0]*1e6:.0f} GB/s" if res else "(none valid)", flush=True)
    del kcs, vcs
