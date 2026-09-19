"""The H100 bf16 GEMM configuration must preserve other dtype behavior."""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from engine import _TorchOps  # noqa: E402
from fused import GEMM_CFG, TritonOps  # noqa: E402

dev = torch.device("cuda")
e = types.SimpleNamespace(dev=dev, hd=8, eps=1e-6)
tor, tri = _TorchOps(e), TritonOps(e)

# Force the configured-shape branch on small matrices.
GEMM_CFG[(8, 8)] = (8, 8, 1, 1, 4)
for dtype in (torch.float32, torch.float16):
    x = torch.randn(3, 8, dtype=dtype, device=dev)
    w = torch.randn(8, 8, dtype=dtype, device=dev)
    assert torch.equal(tri.linear(x, w), tor.linear(x, w)), dtype

x = torch.randn(3, 8, dtype=torch.bfloat16, device=dev)
w = torch.randn(8, 8, dtype=torch.bfloat16, device=dev).T
assert not w.is_contiguous()
assert torch.equal(tri.linear(x, w), tor.linear(x, w))
print("linear dtype fallback OK")
