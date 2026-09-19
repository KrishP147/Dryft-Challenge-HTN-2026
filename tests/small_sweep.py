"""num_warps sweep for the small decode kernels: qkv_post (decode), attn combine, add_rms."""
import os, sys, types, torch, triton
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
bf = torch.bfloat16
NH, NKV, HD = 32, 8, 128
def t(fn, rep=40):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(rep): fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 10 / rep * 1e3
cap = 640
cos = torch.randn(cap, HD, device="cuda").to(bf); sin = torch.randn(cap, HD, device="cuda").to(bf)
qn = torch.randn(HD, device="cuda").to(bf); kn = torch.randn(HD, device="cuda").to(bf)
for B in (1, 4, 16):
    qkv = torch.randn(B, (NH + 2 * NKV) * HD, device="cuda").to(bf)
    q = torch.empty(B, NH, 1, HD, device="cuda", dtype=bf)
    kc = torch.zeros(B, NKV, cap, HD, device="cuda", dtype=bf); vc = torch.zeros_like(kc)
    pos = torch.tensor([500], device="cuda")
    r = []
    for nw in (1, 2, 4):
        f = lambda: fused._qkv_post_kernel[(B, NH + 2 * NKV)](qkv, qn, kn, cos, sin, pos, q, kc, vc, 1, cap, 1e-6, NH=NH, NKV=NKV, HD=HD, DECODE=True, POS_STRIDE=0, num_warps=nw)
        r.append((round(t(f), 2), nw))
    print(f"qkv_post decode B={B}:", r)
    G = 4
    for nsplit in (2, 8, 32):
        ws = torch.randn(B * NKV * G, nsplit, HD + 2, device="cuda"); out = torch.empty(B, NH * HD, device="cuda", dtype=bf)
        r = []
        for nw in (1, 2, 4):
            f = lambda: fused._attn_combine_kernel[(B * NKV * G,)](ws, out, NSPLIT=nsplit, SP=nsplit, HD=HD, NKV=NKV, G=G, W=1, num_warps=nw)
            r.append((round(t(f), 2), nw))
        print(f"  combine B={B} nsplit={nsplit}:", r)
for M in (1, 4, 16):
    h = torch.randn(M, 2560, device="cuda").to(bf); d = torch.randn(M, 2560, device="cuda").to(bf); w = torch.randn(2560, device="cuda").to(bf)
    hn = torch.empty_like(h); y = torch.empty_like(h)
    r = []
    for nw in (1, 2, 4, 8):
        f = lambda: fused._add_rms_kernel[(M,)](h, d, w, hn, y, 2560, 1e-6, HAS_ADD=False, BLOCK=4096, num_warps=nw)
        r.append((round(t(f), 2), nw))
    print(f"rms-only M={M}:", r)
