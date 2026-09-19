"""Skinny-GEMM microbench: cuBLAS F.linear vs Triton split-K kernel, decode shapes.

Cycles through 32 distinct weight copies so L2 (50 MB) cannot help: this is what a
real decode step sees. Reports us/call and effective GB/s.
"""
import itertools
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from fused import _gemv_kernel, _splitk_reduce_kernel  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
SHAPES = {  # name: (N, K)
    "qkv": (6144, 2560),
    "o": (2560, 4096),
    "gate_up": (19456, 2560),
    "down": (2560, 9728),
    "lm_head": (151936, 2560),
}
NCOPY = {"lm_head": 2}


def gemv(x, w, cfg):
    M, K = x.shape
    N = w.shape[0]
    BN, BK, SK, ST, NW = cfg
    kps = triton.cdiv(triton.cdiv(K, SK), BK) * BK
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if SK == 1:
        _gemv_kernel[(triton.cdiv(N, BN), 1)](x, w, out, M, N, K, kps, out.stride(0),
                                              BM=16, BN=BN, BK=BK, FINAL=True, num_warps=NW, num_stages=ST)
        return out
    ws = torch.empty((SK, M, N), dtype=torch.float32, device=x.device)
    _gemv_kernel[(triton.cdiv(N, BN), SK)](x, w, ws, M, N, K, kps, N,
                                           BM=16, BN=BN, BK=BK, FINAL=False, num_warps=NW, num_stages=ST)
    _splitk_reduce_kernel[(triton.cdiv(M * N, 1024),)](ws, out, M * N, SK=SK, BLOCK=1024)
    return out


def timeit(fn, n_iter=3):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1e3  # us per graph


for name, (N, K) in SHAPES.items():
    nc = NCOPY.get(name, 32)
    ws_ = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(nc)]
    gb = N * K * 2 / 1e9
    for M in (1, 4, 16):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, ws_[0])
        base = timeit(lambda: [torch.nn.functional.linear(x, w) for w in ws_]) / nc
        best = None
        for cfg in itertools.product((16, 32, 64), (64, 128, 256), (1, 2, 4, 8, 16), (3, 4, 5), (4, 8)):
            BN, BK, SK, ST, NW = cfg
            if SK > 1 and K // SK < BK:
                continue
            if BN * BK > 16384:
                continue
            try:
                out = gemv(x, ws_[0], cfg)
                if (out.float() - ref.float()).abs().max().item() > 0.02 * max(1, ref.float().abs().max().item()):
                    continue
                t = timeit(lambda: [gemv(x, w, cfg) for w in ws_], 2) / nc
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, cfg)
        print(f"{name:8s} M={M:2d}: cublas {base:7.1f} us ({gb/base*1e6:6.0f} GB/s) | "
              f"triton {best[0]:7.1f} us ({gb/best[0]*1e6:6.0f} GB/s) cfg(BN,BK,SK,ST,NW)={best[1]}", flush=True)
    del ws_
