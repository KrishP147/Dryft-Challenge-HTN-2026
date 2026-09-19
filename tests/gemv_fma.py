"""CUDA-core FMA GEMV (no tensor cores) for M<=4 vs the tl.dot split-K kernel."""
import itertools, os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from fused import _gemv_kernel
bf = torch.bfloat16

@triton.jit
def _gemv_fma_kernel(x_ptr, w_ptr, out_ptr, M, N, K, kps, stride_om,
                     MM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FINAL: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); nm = rn < N
    k_lo = pid_k * kps; k_hi = tl.minimum(k_lo + kps, K)
    a0 = tl.zeros([BN, BK], tl.float32); a1 = tl.zeros([BN, BK], tl.float32)
    a2 = tl.zeros([BN, BK], tl.float32); a3 = tl.zeros([BN, BK], tl.float32)
    for k in range(k_lo, k_hi, BK):
        rk = k + tl.arange(0, BK); km = rk < k_hi
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=nm[:, None] & km[None, :], other=0.0).to(tl.float32)
        a0 += w * tl.load(x_ptr + rk, mask=km, other=0.0).to(tl.float32)[None, :]
        if MM > 1:
            a1 += w * tl.load(x_ptr + K + rk, mask=km & (M > 1), other=0.0).to(tl.float32)[None, :]
        if MM > 2:
            a2 += w * tl.load(x_ptr + 2 * K + rk, mask=km & (M > 2), other=0.0).to(tl.float32)[None, :]
        if MM > 3:
            a3 += w * tl.load(x_ptr + 3 * K + rk, mask=km & (M > 3), other=0.0).to(tl.float32)[None, :]
    base = out_ptr + pid_k * M * stride_om
    dt = out_ptr.dtype.element_ty
    if FINAL:
        tl.store(base + rn, tl.sum(a0, 1).to(dt), mask=nm)
    else:
        tl.store(base + rn, tl.sum(a0, 1), mask=nm)
    if MM > 1:
        if FINAL: tl.store(base + stride_om + rn, tl.sum(a1, 1).to(dt), mask=nm & (M > 1))
        else: tl.store(base + stride_om + rn, tl.sum(a1, 1), mask=nm & (M > 1))
    if MM > 2:
        if FINAL: tl.store(base + 2 * stride_om + rn, tl.sum(a2, 1).to(dt), mask=nm & (M > 2))
        else: tl.store(base + 2 * stride_om + rn, tl.sum(a2, 1), mask=nm & (M > 2))
    if MM > 3:
        if FINAL: tl.store(base + 3 * stride_om + rn, tl.sum(a3, 1).to(dt), mask=nm & (M > 3))
        else: tl.store(base + 3 * stride_om + rn, tl.sum(a3, 1), mask=nm & (M > 3))

def timeit(fn, ws):
    fn(ws[0]); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws: fn(w)
    for _ in range(2): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(4): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 4 / len(ws) * 1e3

dot_best = {(6144, 2560): (64, 128, 1, 5, 4), (2560, 4096): (64, 256, 2, 4, 4), (2560, 9728): (32, 128, 4, 4, 4), (19456, 2560): (64, 64, 1, 4, 4)}
for M in (1, 4):
    for (N, K), (BN, BK, SK, ST, NW) in dot_best.items():
        ws = [(torch.randn(N, K, device="cuda") * 0.02).to(bf) for _ in range(16)]
        x = torch.randn(M, K, device="cuda").to(bf)
        ref = torch.nn.functional.linear(x, ws[0]).float()
        out = torch.empty(M, N, device="cuda", dtype=bf)
        kps = triton.cdiv(triton.cdiv(K, SK), BK) * BK
        wsp = torch.empty(SK, M, N, device="cuda")
        def dot(w):
            if SK == 1: _gemv_kernel[(triton.cdiv(N, BN), 1)](x, w, out, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=True, num_warps=NW, num_stages=ST)
            else: _gemv_kernel[(triton.cdiv(N, BN), SK)](x, w, wsp, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=False, num_warps=NW, num_stages=ST)
        tdot = timeit(dot, ws)
        best = None
        for bn, bk, sk, st, nw in itertools.product((4, 8, 16), (128, 256, 512), (1, 2, 4, 8), (2, 3, 4), (2, 4, 8)):
            if sk > 1 and K // sk < bk: continue
            kp = triton.cdiv(triton.cdiv(K, sk), bk) * bk
            o2 = torch.empty(sk, M, N, device="cuda") if sk > 1 else out
            def fma(w):
                _gemv_fma_kernel[(triton.cdiv(N, bn), sk)](x, w, o2, M, N, K, kp, N, MM=M, BN=bn, BK=bk, FINAL=(sk == 1), num_warps=nw, num_stages=st)
            try:
                fma(ws[0]); torch.cuda.synchronize()
                got = (o2.float().sum(0) if sk > 1 else o2.float())
                if (got - ref).abs().max().item() > 0.05 * max(1, ref.abs().max().item()): continue
                t = timeit(fma, ws)
            except Exception: continue
            if best is None or t < best[0]: best = (t, (bn, bk, sk, st, nw))
        print(f"M={M} N={N:6d} K={K:5d}: dot {tdot:6.1f} us | fma {best[0]:6.1f} us cfg(BN,BK,SK,ST,NW)={best[1]}", flush=True)
