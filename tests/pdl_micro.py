"""Does PDL overlap dependent kernels inside a CUDA graph? Chain of tiny reduce kernels, and of real GEMVs."""
import os, sys
import torch, triton
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
from fused import _launch, _reduce_add_rms_kernel, _gemv_kernel

bf = torch.bfloat16
N = 2560


def graph_time(fn, reps=10):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1e3


ws = torch.randn(2, 1, N, device="cuda")
h = torch.randn(1, N, device="cuda", dtype=bf); w = torch.randn(N, device="cuda", dtype=bf)
hn = torch.empty_like(h); y = torch.empty_like(h)
NK = 400
for pdl in (False, True):
    def chain():
        for _ in range(NK):
            _launch(_reduce_add_rms_kernel, (1,), ws, h, w, hn, y, 1, N, 1e-6, SK=2, BLOCK=4096, num_warps=4, PDL=pdl)
    fused.PDL = pdl
    print(f"tiny chain PDL={pdl}: {graph_time(chain) / NK:.2f} us/kernel")

# real GEMV chain, 24 distinct weight matrices (cold L2)
M, K = 4, 4096
Wt = [(torch.randn(2560, K, device="cuda") * 0.02).to(bf) for _ in range(24)]
x = torch.randn(M, K, device="cuda").to(bf); out = torch.empty(M, 2560, device="cuda", dtype=bf); wsp = torch.empty(2, M, 2560, device="cuda")
for pdl, pf in ((False, 0), (True, 0), (True, 4), (True, 16)):
    fused.PDL = pdl
    def chain():
        for wt in Wt:
            _launch(_gemv_kernel, (40, 2), x, wt, wsp, M, 2560, K, 2048, 2560, BM=16, BN=64, BK=256, FINAL=False, num_warps=4, num_stages=4, PDL=pdl, PF=pf)
    print(f"gemv o chain PDL={pdl} PF={pf}: {graph_time(chain) / len(Wt):.2f} us/kernel")
