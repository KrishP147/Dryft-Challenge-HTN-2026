"""Upper bound for weight prefetch: GEMV time with weights cold (HBM) vs already in L2."""
import os, sys, torch, triton
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from fused import _gemv_kernel
bf = torch.bfloat16
def t(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 10 * 1e3
M = 4
for name, N, K, (BN, BK, SK, ST, NW) in (("o", 2560, 4096, (64, 256, 2, 4, 4)), ("qkv", 6144, 2560, (64, 128, 1, 5, 4)), ("down", 2560, 9728, (32, 128, 4, 4, 4))):
    ws = [(torch.randn(N, K, device="cuda") * 0.02).to(bf) for _ in range(24)]
    x = torch.randn(M, K, device="cuda").to(bf); out = torch.empty(M, N, device="cuda", dtype=bf); wsp = torch.empty(SK, M, N, device="cuda")
    kps = triton.cdiv(triton.cdiv(K, SK), BK) * BK
    def run(w):
        if SK == 1: _gemv_kernel[(triton.cdiv(N, BN), 1)](x, w, out, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=True, num_warps=NW, num_stages=ST)
        else: _gemv_kernel[(triton.cdiv(N, BN), SK)](x, w, wsp, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=False, num_warps=NW, num_stages=ST)
    cold = t(lambda: [run(w) for w in ws]) / len(ws)
    warm = t(lambda: [run(ws[0]) for _ in ws]) / len(ws)
    print(f"{name:4s} {N*K*2/1e6:5.1f} MB  cold {cold:5.1f} us  L2-warm {warm:5.1f} us  (max saving {cold-warm:4.1f} us/layer -> {(cold-warm)*36/1e3:.2f} ms/step)")
