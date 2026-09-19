"""Neighbourhood sweep of GEMV knobs (stages/warps/cache modifier/eviction) around known-good configs."""
import itertools, os, sys
import torch, triton
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
from fused import _gemv_kernel, _gemv_silu_kernel

dev = "cuda"; bf = torch.bfloat16
def timeit(fn, ws):
    fn(ws[0]); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws: fn(w)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); s, t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5): g.replay()
    t.record(); torch.cuda.synchronize()
    return s.elapsed_time(t) / 5 / len(ws) * 1e3

M = 4
cases = [  # name, N, K, base cfg (BN, BK, SK), silu
    ("qkv", 6144, 2560, (64, 128, 1), False),
    ("o", 2560, 4096, (64, 256, 2), False),
    ("down", 2560, 9728, (32, 128, 4), False),
    ("gate_up_silu", 9728, 2560, (32, 128, 1), True),
]
for name, N, K, (BN, BK, SK), silu in cases:
    mult = 2 if silu else 1
    ws = [(torch.randn(mult * N, K, device=dev) * 0.02).to(bf) for _ in range(16 if silu else 24)]
    x = torch.randn(M, K, device=dev).to(bf)
    kps = triton.cdiv(triton.cdiv(K, SK), BK) * BK
    out = torch.empty((M, N), dtype=bf, device=dev)
    wsp = torch.empty((SK, M, N), dtype=torch.float32, device=dev)
    res = []
    for ST, NW, CM, EV in itertools.product((3, 4, 5, 6, 8), (2, 4, 8), ("", ".cg"), ("", "evict_first")):
        def fn(w):
            if silu:
                _gemv_silu_kernel[(triton.cdiv(N, BN),)](x, w, out, M, N, K, BM=16, BN=BN, BK=BK, CM=CM, EV=EV, num_warps=NW, num_stages=ST)
            elif SK == 1:
                _gemv_kernel[(triton.cdiv(N, BN), 1)](x, w, out, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=True, CM=CM, EV=EV, num_warps=NW, num_stages=ST)
            else:
                _gemv_kernel[(triton.cdiv(N, BN), SK)](x, w, wsp, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=False, CM=CM, EV=EV, num_warps=NW, num_stages=ST)
        try: t = timeit(fn, ws)
        except Exception as e: continue
        res.append((t, (ST, NW, CM, EV)))
    res.sort()
    base = [t for t, c in res if c == (4, 4, "", "")]
    gb = mult * N * K * 2 / 1e9
    print(f"{name:13s} base(BN,BK,SK)={(BN,BK,SK)} default-ish {base[0] if base else float('nan'):.1f}us | best:",
          [(round(t, 1), c) for t, c in res[:3]], f"-> {gb/res[0][0]*1e6:.0f} GB/s", flush=True)
