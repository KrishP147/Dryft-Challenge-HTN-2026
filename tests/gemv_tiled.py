"""GEMV with weights re-tiled so each [BN, BK] tile is one contiguous chunk vs row-major."""
import itertools, os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from fused import _gemv_kernel
bf = torch.bfloat16

@triton.jit
def _gemv_tiled_kernel(x_ptr, w_ptr, out_ptr, M, N, K, kps, stride_om,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FINAL: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN); rm = tl.arange(0, BM); mm = rm < M; nm = rn < N
    ln = tl.arange(0, BN); lk = tl.arange(0, BK)
    KB = K // BK
    acc = tl.zeros([BM, BN], tl.float32)
    kb0 = pid_k * (kps // BK)
    kb1 = tl.minimum(kb0 + kps // BK, KB)
    tile_base = (pid_n * KB).to(tl.int64) * BN * BK
    for kb in range(kb0, kb1):
        rk = kb * BK + lk
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + tile_base + kb * BN * BK + ln[None, :] * BK + lk[:, None])
        acc = tl.dot(x, w, acc)
    if FINAL:
        tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty), mask=mm[:, None] & nm[None, :])
    else:
        tl.store(out_ptr + pid_k * M * stride_om + rm[:, None] * stride_om + rn[None, :], acc, mask=mm[:, None] & nm[None, :])

def timeit(fn, ws):
    fn(ws[0]); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws: fn(w)
    for _ in range(2): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(4): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 4 / len(ws) * 1e3

dot_best = {"qkv": (6144, 2560, (64, 128, 1, 5, 4)), "o": (2560, 4096, (64, 256, 2, 4, 4)), "down": (2560, 9728, (32, 128, 4, 4, 4)), "gate_up": (19456, 2560, (64, 64, 1, 4, 4))}
M = 4
for name, (N, K, (BN, BK, SK, ST, NW)) in dot_best.items():
    ws = [(torch.randn(N, K, device="cuda") * 0.02).to(bf) for _ in range(16)]
    x = torch.randn(M, K, device="cuda").to(bf)
    ref = torch.nn.functional.linear(x, ws[0]).float()
    out = torch.empty(M, N, device="cuda", dtype=bf); kps = triton.cdiv(triton.cdiv(K, SK), BK) * BK
    wsp = torch.empty(SK, M, N, device="cuda")
    def dot(w):
        if SK == 1: _gemv_kernel[(triton.cdiv(N, BN), 1)](x, w, out, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=True, num_warps=NW, num_stages=ST)
        else: _gemv_kernel[(triton.cdiv(N, BN), SK)](x, w, wsp, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=False, num_warps=NW, num_stages=ST)
    tdot = timeit(dot, ws)
    best = None
    for bn, bk, sk, st, nw in itertools.product((16, 32, 64), (64, 128, 256), (1, 2, 4), (3, 4, 5), (4, 8)):
        if N % bn or K % bk: continue
        if sk > 1 and (K // bk) % sk: continue
        tiles = [w.view(N // bn, bn, K // bk, bk).permute(0, 2, 1, 3).contiguous() for w in ws[:1]]
        wt = [tiles[0].clone() for _ in ws]  # distinct copies so L2 cannot help
        kp = (K // bk // sk) * bk
        o2 = torch.empty(sk, M, N, device="cuda") if sk > 1 else out
        def tl_(w):
            _gemv_tiled_kernel[(N // bn, sk)](x, w, o2, M, N, K, kp, N, BM=16, BN=bn, BK=bk, FINAL=(sk == 1), num_warps=nw, num_stages=st)
        try:
            tl_(wt[0]); torch.cuda.synchronize()
            got = o2.float().sum(0) if sk > 1 else o2.float()
            if (got - ref).abs().max().item() > 0.05 * max(1, ref.abs().max().item()): continue
            t = timeit(tl_, wt)
        except Exception: continue
        del wt
        if best is None or t < best[0]: best = (t, (bn, bk, sk, st, nw))
    gb = N * K * 2 / 1e9
    print(f"{name:8s}: row-major {tdot:6.1f} us ({gb/tdot*1e6:4.0f} GB/s) | tiled {best[0]:6.1f} us ({gb/best[0]*1e6:4.0f} GB/s) cfg={best[1]}", flush=True)
