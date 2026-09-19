"""GPU: Triton split-KV decode attention (multi-token causal, per-seq pos) vs fp32 reference."""
import os, sys, types
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from fused import TritonOps

dev = torch.device("cuda")
NH, NKV, HD = 32, 8, 128
G = NH // NKV
e = types.SimpleNamespace(dev=dev, nh=NH, nkv=NKV, hd=HD, eps=1e-6, dtype=torch.bfloat16)
tri = TritonOps(e)
torch.manual_seed(0)

def ref(q, kc, vc, pos, W):
    B = q.shape[0]
    out = torch.zeros(B * W, NH * HD, device=dev)
    for b in range(B):
        for j in range(W):
            L = int(pos[b if pos.numel() > 1 else 0]) + j + 1
            K = kc[b, :, :L].float(); V = vc[b, :, :L].float()      # [NKV, L, HD]
            qj = q[b, :, j].float()                                 # [NKV, G, HD]
            p = torch.softmax(qj @ K.transpose(1, 2) * HD ** -0.5, -1)
            out[b * W + j] = (p @ V).reshape(-1)
    return out

worst = 0
for B, W, cap, posl in [(1, 1, 640, [512]), (1, 7, 640, [512]), (1, 7, 640, [0]), (2, 7, 384, [200, 11]),
                        (3, 5, 2304, [2048, 900, 5]), (4, 4, 640, [512, 512, 300, 0]),
                        (16, 1, 640, [512] * 16), (32, 1, 640, [512] * 32)]:
    kc = torch.randn(B, NKV, cap, HD, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(B, NKV, cap, HD, device=dev, dtype=torch.bfloat16)
    q = torch.randn(B, NKV, W, G, HD, device=dev, dtype=torch.bfloat16)
    pos = torch.tensor(posl if B > 1 else posl[:1], device=dev)
    if B > 1 and len(posl) == 1: pos = pos.expand(B).contiguous()
    a = tri.attn_decode(q.view(B, NH, W, HD) if False else q, kc, vc, pos, W).float()
    r = ref(q, kc, vc, pos, W)
    d = (a - r).abs().max().item(); worst = max(worst, d)
    print(f"B{B} W{W} cap{cap} pos{posl}: max|diff|={d:.5f}")
assert worst < 0.02, worst
print("ATTN OK")
