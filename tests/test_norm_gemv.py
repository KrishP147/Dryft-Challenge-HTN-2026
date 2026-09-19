"""GPU: fused norm/silu/add GEMV ops vs torch ops; sweeps norm-kernel configs."""
import itertools, os, sys, types
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
from engine import _TorchOps
from fused import TritonOps

dev = "cuda"
e = types.SimpleNamespace(dev=dev, nh=32, nkv=8, hd=128, eps=1e-6, dtype=torch.bfloat16)
tor, tri = _TorchOps(e), TritonOps(e)
torch.manual_seed(0)
bf = torch.bfloat16
H, I = 2560, 9728
lnw = (torch.randn(H, device=dev) * 0.5 + 1).to(bf)
wqkv = (torch.randn(6144, H, device=dev) * 0.02).to(bf)
wgu = (torch.randn(2 * I, H, device=dev) * 0.02).to(bf)
wo = (torch.randn(H, 4096, device=dev) * 0.02).to(bf)
wd = (torch.randn(H, I, device=dev) * 0.02).to(bf)

def gap(a, b):
    return (a.float() - b.float()).abs().max().item(), b.float().abs().max().item()

for M in (1, 4, 16):
    h = torch.randn(M, H, device=dev).to(bf) * 3
    a_ = torch.randn(M, H, device=dev).to(bf)
    print(f"M={M}", "gate_up_silu", gap(tri.gate_up_silu(a_, wgu), tor.gate_up_silu(a_, wgu)))
    for nm, K, w in (("o", 4096, wo), ("down", I, wd)):
        x = torch.randn(M, K, device=dev).to(bf)
        th, ty = tor.linear_add_norm(x, w, h, lnw); fh, fy = tri.linear_add_norm(x, w, h, lnw)
        print(f"   linear_add_norm {nm}: h", gap(fh, th), "y", gap(fy, ty))

def timeit(fn, ws):
    g = torch.cuda.CUDAGraph(); fn(ws[0]); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for w in ws: fn(w)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); s, t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5): g.replay()
    t.record(); torch.cuda.synchronize()
    return s.elapsed_time(t) / 5 / len(ws) * 1e3

M = 4
xx = torch.randn(M, H, device=dev).to(bf)
N = I
ws = [(torch.randn(2 * N, H, device=dev) * 0.02).to(bf) for _ in range(12)]
out = torch.empty((M, N), dtype=bf, device=dev)
best = []
for BN, BK, ST, NW in itertools.product((16, 32, 64), (64, 128, 256), (3, 4, 5), (2, 4, 8)):
    if BN * BK > 16384: continue
    def fn(w):
        fused._gemv_silu_kernel[((N + BN - 1) // BN,)](xx, w, out, M, N, H, BM=16, BN=BN, BK=BK, num_warps=NW, num_stages=ST)
    try: t_ = timeit(fn, ws)
    except Exception: continue
    best.append((t_, (BN, BK, ST, NW)))
best.sort()
print("gate_up_silu", [(round(t_, 1), c) for t_, c in best[:4]], f"-> {2*N*H*2/1e9/best[0][0]*1e6:.0f} GB/s")
