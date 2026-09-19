import os, sys, torch, triton
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
bf = torch.bfloat16
def t(fn):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(40): fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 10 / 40 * 1e3
N = 2560
for M in (1, 4, 16):
    for SK in (2, 4):
        ws = torch.randn(SK, M, N, device="cuda"); h = torch.randn(M, N, device="cuda").to(bf); w = torch.randn(N, device="cuda").to(bf)
        hn = torch.empty_like(h); y = torch.empty_like(h)
        res = []
        for nw in (1, 2, 4, 8, 16):
            f = lambda: fused._reduce_add_rms_kernel[(M,)](ws, h, w, hn, y, M, N, 1e-6, SK=SK, BLOCK=4096, num_warps=nw)
            res.append((round(t(f), 2), nw))
        print(f"M={M} SK={SK}:", res)
