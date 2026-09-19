"""Interpreter (CPU) check: _gemv_kernel and _gemv_silu_kernel EVENK path == masked path == torch."""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch, triton
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
torch.manual_seed(0)
M, N, K = 3, 64, 256
x = torch.randn(M, K); w = torch.randn(N, K) * 0.05
ref = x @ w.T
for SK, BN, BK in ((1, 16, 64), (2, 32, 64), (4, 16, 32)):
    kps = K // SK
    for even in (False, True):
        out = torch.zeros(SK, M, N)
        fused._gemv_kernel[(N // BN, SK)](x, w, out, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=False, EVENK=even, num_warps=4)
        d = (out.sum(0) - ref).abs().max().item()
        print(f"gemv SK={SK} BN={BN} BK={BK} even={even}: {d:.2e}"); assert d < 1e-4
I = 32; wgu = torch.randn(2 * I, K) * 0.05
g, u = (x @ wgu.T).chunk(2, -1); ref2 = torch.nn.functional.silu(g) * u
for even in (False, True):
    out = torch.zeros(M, I)
    fused._gemv_silu_kernel[(I // 16,)](x, wgu, out, M, I, K, BM=16, BN=16, BK=64, EVENK=even, num_warps=4)
    d = (out - ref2).abs().max().item(); print(f"silu even={even}: {d:.2e}"); assert d < 1e-4
print("EVEN OK")
