"""Fused Triton ops vs torch ops. Run with TRITON_INTERPRET=1 (CPU) or on a GPU."""
import os
import sys
import types

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from engine import _TorchOps, _Layer, _build_rope_tables  # noqa: E402
from fused import TritonOps  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bf = torch.bfloat16 if dev.type == "cuda" else torch.float32
torch.manual_seed(0)
NH, NKV, HD, H, I = 8, 2, 128, 256, 192
e = types.SimpleNamespace(dev=dev, eps=1e-6, nh=NH, nkv=NKV, hd=HD, dtype=bf)
e.cos, e.sin = _build_rope_tables(5e6, HD, 64, dev, bf)
tor, tri = _TorchOps(e), TritonOps(e)


def close(name, a, b, tol=0.02):
    d = (a.float() - b.float()).abs().max().item()
    exact = torch.equal(a, b)
    print(f"{name:22s} max|diff|={d:.5f} exact={exact}")
    assert d <= tol * max(1.0, b.float().abs().max().item()), name


h = torch.randn(5, H, device=dev, dtype=bf)
d = torch.randn(5, H, device=dev, dtype=bf)
w = torch.randn(H, device=dev, dtype=bf)
close("rms", tri.rms(h, w), tor.rms(h, w))
th, ty = tor.add_rms(h, d, w)
fh, fy = tri.add_rms(h, d, w)
close("add_rms h", fh, th)
close("add_rms y", fy, ty)
gu = torch.randn(7, 2 * I, device=dev, dtype=bf)
close("silu_mul", tri.silu_mul(gu), tor.silu_mul(gu))

l = _Layer()
l.qn = torch.randn(HD, device=dev, dtype=bf)
l.kn = torch.randn(HD, device=dev, dtype=bf)
for B, S, pos in [(2, 6, None), (3, 1, torch.tensor([9], device=dev))]:
    cap = 16
    qkv = torch.randn(B * S, (NH + 2 * NKV) * HD, device=dev, dtype=bf)
    out = {}
    for name, ops in (("torch", tor), ("triton", tri)):
        kc = torch.zeros(B, NKV, cap, HD, device=dev, dtype=bf)
        vc = torch.zeros_like(kc)
        q = ops.qkv_post(qkv, l, kc, vc, B, S, pos)
        out[name] = (q, kc, vc)
    tag = f"B{B} S{S} {'dec' if pos is not None else 'pre'}"
    for i, n in enumerate(("q", "k", "v")):
        close(f"qkv_post {tag} {n}", out["triton"][i], out["torch"][i])
print("all OK")
