"""Sanity: with PDL, kernel B's pre-wait work overlaps kernel A's tail."""
import os, sys
import torch, triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
from fused import _launch, _gdc_wait, _gdc_launch


@triton.jit
def spin(x_ptr, y_ptr, iters, PRE, PDL: tl.constexpr):
    v = tl.load(x_ptr + tl.arange(0, 32))
    if PDL:
        _gdc_launch()
        for _ in range(PRE):
            v = v * 1.0001 + 0.5
            tl.store(y_ptr + tl.arange(0, 32), v)
        _gdc_wait()
    for _ in range(iters):
        v = v * 1.0001 + 0.5
    tl.store(x_ptr + tl.arange(0, 32), v)


x = torch.zeros(32, device="cuda"); y = torch.zeros(32, device="cuda")


def t(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 10 * 1e3


IT = 20000
for pdl, pre in ((False, 0), (True, 0), (True, IT)):
    fused.PDL = pdl
    def chain():
        for _ in range(4):
            _launch(spin, (1,), x, y, IT, pre, PDL=pdl, num_warps=1)
    print(f"4 x spin({IT})  PDL={pdl} pre-wait={pre}: {t(chain):.1f} us total")

K = [None]
# eager (no graph)
def teager(fn):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 10 * 1e3
for pdl, pre in ((False, 0), (True, IT)):
    fused.PDL = pdl
    def chain():
        for _ in range(4):
            K[0] = _launch(spin, (1,), x, y, IT, pre, PDL=pdl, num_warps=1)
    print(f"EAGER 4 x spin  PDL={pdl} pre-wait={pre}: {teager(chain):.1f} us total; meta={K[0].packed_metadata}")
