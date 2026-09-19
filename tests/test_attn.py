"""GPU: Triton split-KV decode attention vs SDPA-with-mask reference."""
import os, sys, types
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import _TorchOps
from fused import TritonOps

dev = torch.device("cuda")
e = types.SimpleNamespace(dev=dev, nh=32, nkv=8, hd=128, eps=1e-6, dtype=torch.bfloat16)
tor, tri = _TorchOps(e), TritonOps(e)
torch.manual_seed(0)
worst = 0
for B, cap, L in [(1, 640, 513), (1, 640, 1), (1, 640, 640), (4, 2304, 2049), (16, 640, 513), (16, 640, 640), (3, 384, 200)]:
    kc = torch.randn(B, 8, cap, 128, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(B, 8, cap, 128, device=dev, dtype=torch.bfloat16)
    q = torch.randn(B, 32, 1, 128, device=dev, dtype=torch.bfloat16)
    pos = torch.tensor([L - 1], device=dev)
    a, b = tri.attn_decode(q, kc, vc, pos), tor.attn_decode(q, kc, vc, pos)
    d = (a.float() - b.float()).abs().max().item()
    worst = max(worst, d)
    print(f"B{B} cap{cap} L{L}: max|diff|={d:.5f} ref|max|={b.float().abs().max().item():.3f}")
    # timing (graph-free, sync'd)
    for name, f in (("triton", tri), ("sdpa", tor)):
        for _ in range(5): f.attn_decode(q, kc, vc, pos)
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); t = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50): f.attn_decode(q, kc, vc, pos)
        t.record(); torch.cuda.synchronize()
        print(f"    {name}: {s.elapsed_time(t)/50*1e3:.1f} us")
assert worst < 0.05, worst
print("ATTN OK")
