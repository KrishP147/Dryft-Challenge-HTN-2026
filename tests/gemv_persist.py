"""Persistent, SM-balanced GEMV: NCTA programs each take a contiguous slice of (row-tile, K-split) units.

Hypothesis: small GEMVs (qkv: 96 CTAs on 132 SMs) leave ~27% of SMs idle; a balanced persistent
grid keeps every SM streaming its share of the weights.
"""
import itertools, os, sys
import torch, triton, triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from fused import _gemv_kernel

bf = torch.bfloat16


@triton.jit
def _gemv_persist_kernel(
    x_ptr, w_ptr, out_ptr, M, N, K, kps, stride_om, T, NCTA,
    SK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    U = T * SK
    u_lo = pid * U // NCTA
    u_hi = (pid + 1) * U // NCTA
    ipu = kps // BK                      # BK-iterations per unit
    total = (u_hi - u_lo) * ipu
    rm = tl.arange(0, BM)
    mm = rm < M
    ln = tl.arange(0, BN)
    lk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], tl.float32)
    for it in range(total):
        u = u_lo + it // ipu
        kk = it % ipu
        tile = u // SK
        ks = u % SK
        rn = tile * BN + ln
        rk = ks * kps + kk * BK + lk
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + rn[None, :] * K + rk[:, None])
        acc = tl.dot(x, w, acc)
        if kk == ipu - 1:
            if FINAL:
                tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty), mask=mm[:, None])
            else:
                tl.store(out_ptr + ks * M * stride_om + rm[:, None] * stride_om + rn[None, :], acc, mask=mm[:, None])
            acc = tl.zeros([BM, BN], tl.float32)


def timeit(fn, ws):
    fn(ws[0]); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws: fn(w)
    for _ in range(2): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(4): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 4 / len(ws) * 1e3


dot_best = {"qkv": (6144, 2560, (64, 128, 1, 5, 4)), "o": (2560, 4096, (64, 256, 2, 4, 4)),
            "down": (2560, 9728, (32, 128, 4, 4, 4)), "gate_up": (19456, 2560, (64, 64, 1, 4, 4))}
M = 4
for name, (N, K, (BN0, BK0, SK0, ST0, NW0)) in dot_best.items():
    ws = [(torch.randn(N, K, device="cuda") * 0.02).to(bf) for _ in range(16)]
    x = torch.randn(M, K, device="cuda").to(bf)
    ref = torch.nn.functional.linear(x.float(), ws[0].float())
    out = torch.empty(M, N, device="cuda", dtype=bf)
    kps0 = K // SK0
    wsp0 = torch.empty(SK0, M, N, device="cuda")
    def cur(w):
        if SK0 == 1:
            _gemv_kernel[(N // BN0, 1)](x, w, out, M, N, K, kps0, N, BM=16, BN=BN0, BK=BK0, FINAL=True, EVENK=True, num_warps=NW0, num_stages=ST0)
        else:
            _gemv_kernel[(N // BN0, SK0)](x, w, wsp0, M, N, K, kps0, N, BM=16, BN=BN0, BK=BK0, FINAL=False, EVENK=True, num_warps=NW0, num_stages=ST0)
    tcur = timeit(cur, ws)
    best = []
    for BN, BK, SK, ST, NW, NCTA in itertools.product((16, 32), (128, 256, 512), (1, 2, 4), (3, 4, 5), (4, 8), (132, 264)):
        if N % BN or K % (SK * BK): continue
        kps = K // SK
        T = N // BN
        wsp = torch.empty(SK, M, N, device="cuda")
        o2 = out if SK == 1 else wsp
        def fn(w):
            _gemv_persist_kernel[(NCTA,)](x, w, o2, M, N, K, kps, N, T, NCTA, SK=SK, BM=16, BN=BN, BK=BK, FINAL=(SK == 1), num_warps=NW, num_stages=ST)
        try:
            fn(ws[0]); torch.cuda.synchronize()
            got = (wsp.sum(0) if SK > 1 else out).float()
            if (got - ref).abs().max().item() > 0.05 * max(1, ref.abs().max().item()): continue
            best.append((timeit(fn, ws), (BN, BK, SK, ST, NW, NCTA)))
        except Exception:
            continue
    best.sort()
    gb = N * K * 2 / 1e9
    print(f"{name:8s}: dot-kernel {tcur:6.2f} us ({gb/tcur*1e6:4.0f} GB/s) | persistent best {[(round(t, 2), c) for t, c in best[:3]]} -> {gb/best[0][0]*1e6:.0f} GB/s", flush=True)
