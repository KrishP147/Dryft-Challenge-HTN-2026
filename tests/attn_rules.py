"""Compare decode-attention config RULES (nsplit heuristic x num_stages x BLOCK_N) across regimes.

Realistic traffic: cycles NL distinct per-layer KV caches so L2 cannot help. Reports us per layer
(split + combine) for each rule and a regime-weighted total (weights = score-model sensitivities).
"""
import os, sys
import torch, triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused

NH, NKV, HD, G = 32, 8, 128, 4
bf = torch.bfloat16


def pow2_at_least(target, bk, cap=32):
    n = 1
    while bk * n < target and n < cap:
        n *= 2
    return n


RULES = {  # name -> f(bk, L) -> (nsplit, BN, NW, ST)
    "current-v15": lambda bk, L: (1 if bk >= 128 else pow2_at_least(256, bk), 64, 4, 3),
    "v14(>=256,st2)": lambda bk, L: (pow2_at_least(256, bk), 64, 4, 2),
    "st3 (>=256)": lambda bk, L: (pow2_at_least(256, bk), 64, 4, 3),
    "st3 (>=128)": lambda bk, L: (pow2_at_least(128, bk), 64, 4, 3),
    "st3 (>=64)": lambda bk, L: (pow2_at_least(64, bk), 64, 4, 3),
    "st3 (>=192)": lambda bk, L: (pow2_at_least(192, bk), 64, 4, 3),
    "st3 (>=128) BN32": lambda bk, L: (pow2_at_least(128, bk), 32, 2, 3),
    "st3 nsplit=1 if bk>=64": lambda bk, L: (1 if bk >= 64 else pow2_at_least(256, bk), 64, 4, 3),
}
# (B, L, cap, weight): weights ~ regime sensitivities; long shapes get a small weight
SHAPES = [(1, 544, 640, 0.14), (4, 2080, 2176, 0.45), (16, 640, 640, 0.44), (16, 2080, 2176, 0.15), (8, 2080, 2176, 0.10), (32, 576, 640, 0.10)]


def t(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(2): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 5 * 1e3


totals = {k: 0.0 for k in RULES}
for B, L, cap, wgt in SHAPES:
    NL = 6 if B * cap > 60000 else (12 if B * cap > 20000 else 36)
    kcs = [torch.randn(B, NKV, cap, HD, device="cuda", dtype=bf) for _ in range(NL)]
    vcs = [torch.randn_like(kcs[0]) for _ in range(NL)]
    q = torch.randn(B, NKV, 1, G, HD, device="cuda", dtype=bf)
    pos = torch.tensor([L - 1], device="cuda")
    bk = B * NKV
    out = torch.empty((B, NH * HD), dtype=bf, device="cuda")
    row = {}
    for name, rule in RULES.items():
        ns, BN, NW, ST = rule(bk, L)
        ws = (out if ns == 1 else
              torch.empty((bk * G, ns, HD + 2), dtype=torch.float32, device="cuda"))

        def fn():
            for i in range(NL):
                fused._attn_split_kernel[(bk, ns)](q, kcs[i], vcs[i], pos, ws, out, cap, HD ** -0.5, NSPLIT=ns, G=G, W=1, GP=16, HD=HD, BLOCK_N=BN, NKV=NKV, POS_STRIDE=0, num_warps=NW, num_stages=ST)
                if ns > 1:
                    fused._attn_combine_kernel[(bk * G,)](ws, out, NSPLIT=ns, SP=ns, HD=HD, NKV=NKV, G=G, W=1, num_warps=1)
        row[name] = t(fn) / NL
        totals[name] += wgt * row[name]
    base = row["current-v15"]
    print(f"B{B:2d} L{L}: " + " | ".join(f"{k.split(' ')[0] if False else k}: {v:5.1f}us" for k, v in row.items()), flush=True)
    del kcs, vcs
print("\nregime-weighted us/layer (lower is better):")
base = totals["current-v15"]
for k, v in sorted(totals.items(), key=lambda kv: kv[1]):
    print(f"  {k:28s} {v:7.2f}  ({100 * (base / v - 1):+.1f}% vs current)")
