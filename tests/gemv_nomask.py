"""Does dropping K/N masks (+ alignment hints) speed up the GEMV? Also dumps PTX load widths.

GPU: python tests/gemv_nomask.py          (timing, cold weights, in a CUDA graph)
CPU: TRITON_INTERPRET=1 python tests/gemv_nomask.py --check   (logic only, fp32, tiny)
"""
import os, re, sys

CHECK = "--check" in sys.argv
if CHECK:
    os.environ["TRITON_INTERPRET"] = "1"  # must be set before triton is imported
import torch, triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused
from fused import _gemv_kernel


@triton.jit
def _gemv_nm_kernel(
    x_ptr, w_ptr, out_ptr, M, N, K, kps, stride_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FINAL: tl.constexpr, HINT: tl.constexpr,
):
    # same math as _gemv_kernel, but requires N % BN == 0 and kps % BK == 0 with every
    # split full: no K or N masks, so loads along K stay fully vectorised.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    mm = rm < M
    k_lo = pid_k * kps
    acc = tl.zeros([BM, BN], tl.float32)
    for k in range(k_lo, k_lo + kps, BK):
        rk = k + tl.arange(0, BK)
        if HINT:
            rk = tl.max_contiguous(tl.multiple_of(rk, BK), BK)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + rn[None, :] * K + rk[:, None])
        acc = tl.dot(x, w, acc)
    if FINAL:
        tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty), mask=mm[:, None])
    else:
        tl.store(out_ptr + pid_k * M * stride_om + rm[:, None] * stride_om + rn[None, :], acc, mask=mm[:, None])


def ptx_summary(k):
    if k is None or not hasattr(k, "asm") or "ptx" not in k.asm:
        return "n/a"
    ptx = k.asm["ptx"]
    cp = re.findall(r"cp\.async\.[\w.]+\s+\[[^\]]+\],\s*\[[^\]]+\],\s*(\d+)", ptx)
    ldg = re.findall(r"ld\.global(?:\.[\w]+)*\.(v\d\.)?[bu]\d+", ptx)
    ldv = {}
    for m in re.findall(r"ld\.global[\w.]*", ptx):
        ldv[m] = ldv.get(m, 0) + 1
    cpc = {}
    for c in cp:
        cpc[c] = cpc.get(c, 0) + 1
    return f"cp.async bytes {cpc} | ld.global {dict(sorted(ldv.items(), key=lambda x: -x[1])[:4])}"


def timeit(fn, ws):
    fn(ws[0]); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws: fn(w)
    for _ in range(2): g.replay()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(5): g.replay()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / 5 / len(ws) * 1e3


if CHECK:
    dev, dt = "cpu", torch.float32
    cases = {"tiny": (64, 256, (16, 64, 2, 2, 4))}
    M = 3
else:
    dev, dt = "cuda", torch.bfloat16
    cases = {"qkv": (6144, 2560, (64, 128, 1, 5, 4)), "o": (2560, 4096, (64, 256, 2, 4, 4)),
             "down": (2560, 9728, (32, 128, 4, 4, 4)), "gate_up*": (19456, 2560, (64, 64, 1, 4, 4))}
    M = 4

for name, (N, K, (BN, BK, SK, ST, NW)) in cases.items():
    assert N % BN == 0 and K % (SK * BK) == 0, name
    kps = K // SK
    ws = [(torch.randn(N, K, device=dev) * 0.02).to(dt) for _ in range(1 if CHECK else 16)]
    x = torch.randn(M, K, device=dev).to(dt)
    ref = torch.nn.functional.linear(x.float(), ws[0].float())
    wsp = torch.empty(SK, M, N, device=dev, dtype=torch.float32)
    out = torch.empty(M, N, device=dev, dtype=dt)

    def cur(w):
        if SK == 1:
            return _gemv_kernel[(triton.cdiv(N, BN), 1)](x, w, out, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=True, num_warps=NW, num_stages=ST)
        return _gemv_kernel[(triton.cdiv(N, BN), SK)](x, w, wsp, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=False, num_warps=NW, num_stages=ST)

    def nm(hint):
        def f(w):
            if SK == 1:
                return _gemv_nm_kernel[(N // BN, 1)](x, w, out, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=True, HINT=hint, num_warps=NW, num_stages=ST)
            return _gemv_nm_kernel[(N // BN, SK)](x, w, wsp, M, N, K, kps, N, BM=16, BN=BN, BK=BK, FINAL=False, HINT=hint, num_warps=NW, num_stages=ST)
        return f

    def result():
        return (wsp.sum(0) if SK > 1 else out).float()

    if CHECK:
        for hint in (False,):  # tl.multiple_of/max_contiguous are not interpretable on CPU
            nm(hint)(ws[0]); d = (result() - ref).abs().max().item()
            print(f"{name} hint={hint}: max|diff|={d:.2e}")
            assert d < 1e-3
        print("CHECK OK")
        continue
    k0 = cur(ws[0]); k1 = nm(False)(ws[0]); k2 = nm(True)(ws[0])
    for label, f in (("masked (current)", cur), ("no-mask", nm(False)), ("no-mask+hints", nm(True))):
        f(ws[0]); torch.cuda.synchronize()
        d = (result() - ref).abs().max().item()
        t = timeit(f, ws)
        print(f"{name:9s} {label:17s} {t:6.2f} us ({N*K*2/1e9/t*1e6:5.0f} GB/s) maxdiff {d:.3f}", flush=True)
    print(f"   PTX masked: {ptx_summary(k0)}\n   PTX nomask: {ptx_summary(k2)}", flush=True)
