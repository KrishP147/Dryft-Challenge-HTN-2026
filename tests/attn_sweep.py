"""Sweep split-KV decode attention knobs with realistic (36-layer, L2-cold) KV traffic."""
import itertools, os, sys, types
import torch, triton
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
NH, NKV, HD, G = 32, 8, 128, 4
bf = torch.bfloat16
def t(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(2): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 5 * 1e3
for B, L, cap in ((1, 544, 640), (4, 2080, 2176), (16, 640, 640), (16, 128, 256)):
    NL = 12 if B * cap > 20000 else 36
    kcs = [torch.randn(B, NKV, cap, HD, device="cuda", dtype=bf) for _ in range(NL)]; vcs = [torch.randn_like(kcs[0]) for _ in range(NL)]
    q = torch.randn(B, NKV, 1, G, HD, device="cuda", dtype=bf); pos = torch.tensor([L - 1], device="cuda")
    bk = B * NKV; gb = 2 * B * NKV * L * HD * 2 / 1e9
    res = []
    for nsplit, BN, NW, ST in itertools.product((1, 2, 4, 8, 16, 32), (32, 64, 128), (2, 4, 8), (2, 3, 4)):
        ws = torch.empty((bk * G, nsplit, HD + 2), dtype=torch.float32, device="cuda"); out = torch.empty((B, NH * HD), dtype=bf, device="cuda")
        def fn():
            for i in range(NL):
                fused._attn_split_kernel[(bk, nsplit)](q, kcs[i], vcs[i], pos, ws, cap, HD ** -0.5, NSPLIT=nsplit, G=G, W=1, GP=16, HD=HD, BLOCK_N=BN, NKV=NKV, POS_STRIDE=0, num_warps=NW, num_stages=ST)
                fused._attn_combine_kernel[(bk * G,)](ws, out, NSPLIT=nsplit, SP=nsplit, HD=HD, NKV=NKV, G=G, W=1, num_warps=1)
        try: res.append((t(fn) / NL, (nsplit, BN, NW, ST)))
        except Exception: pass
    res.sort()
    cur = [r for r in res if r[1] == (32 if B * NKV * 32 < 256 * 32 and False else 1, 64, 4, 2)]
    print(f"B{B} L{L}: best", [(round(a, 1), c) for a, c in res[:3]], f"-> {gb/res[0][0]*1e6:.0f} GB/s", flush=True)
    del kcs, vcs
