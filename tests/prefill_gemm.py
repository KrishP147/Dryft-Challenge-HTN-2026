"""Prefill gate/up GEMM with fused SiLU*up epilogue (Triton) vs cuBLAS F.linear + silu_mul kernel.

out[T, I] = silu(x @ Wg^T) * (x @ Wu^T), x [T, K], W = [Wg; Wu] [2I, K]. Rounding points as the reference:
g, u -> bf16, silu(g) -> bf16, product -> bf16. Wins only if the Triton GEMM stays within ~10% of
cuBLAS's ~735 TFLOP/s, because the fusion only removes the 311 MB gu round trip (~150 us/layer).
"""
import itertools, os, sys
import torch, triton, triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused

bf = torch.bfloat16


@triton.jit
def _gu_gemm_kernel(
    a_ptr, w_ptr, out_ptr, M, I, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = I // BN
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsz
    pid_n = (pid % group) // gsz
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    acc_g = tl.zeros([BM, BN], tl.float32)
    acc_u = tl.zeros([BM, BN], tl.float32)
    a_ptrs = a_ptr + rm[:, None].to(tl.int64) * K + rk[None, :]
    g_ptrs = w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None]
    u_ptrs = w_ptr + (rn[None, :] + I).to(tl.int64) * K + rk[:, None]
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=mm[:, None], other=0.0)
        acc_g = tl.dot(a, tl.load(g_ptrs), acc_g)
        acc_u = tl.dot(a, tl.load(u_ptrs), acc_u)
        a_ptrs += BK
        g_ptrs += BK
        u_ptrs += BK
    dt = out_ptr.dtype.element_ty
    g = acc_g.to(dt).to(tl.float32)
    u = acc_u.to(dt).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(dt)
    tl.store(out_ptr + rm[:, None].to(tl.int64) * I + rn[None, :], (s.to(tl.float32) * u).to(dt), mask=mm[:, None])


def gu_gemm(x, w, cfg):
    T, K = x.shape
    I = w.shape[0] // 2
    BM, BN, BK, ST, NW, GM = cfg
    out = torch.empty((T, I), dtype=x.dtype, device=x.device)
    _gu_gemm_kernel[(triton.cdiv(T, BM) * (I // BN),)](x, w, out, T, I, K, BM=BM, BN=BN, BK=BK, GROUP_M=GM, num_warps=NW, num_stages=ST)
    return out


def timeit(fn, n=8):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n


I, K = 9728, 2560
tri = fused.TritonOps.__new__(fused.TritonOps)
for T in (8192, 2048):
    x = torch.randn(T, K, device="cuda", dtype=bf)
    w = (torch.randn(2 * I, K, device="cuda") * 0.02).to(bf)
    ref = tri.silu_mul(torch.nn.functional.linear(x, w))
    t_ref = timeit(lambda: tri.silu_mul(torch.nn.functional.linear(x, w)))
    t_gemm = timeit(lambda: torch.nn.functional.linear(x, w))
    flops = 2 * T * K * 2 * I
    print(f"T={T}: cuBLAS+silu_mul {t_ref:.3f} ms (gemm alone {t_gemm:.3f} ms = {flops/t_gemm/1e9:.0f} TFLOP/s)", flush=True)
    res = []
    for cfg in itertools.product((128, 64), (64, 128), (32, 64), (3, 4, 5), (4, 8), (8,)):
        BM, BN, BK, ST, NW, GM = cfg
        if I % BN or K % BK:
            continue
        try:
            out = gu_gemm(x, w, cfg)
            d = (out.float() - ref.float()).abs().max().item()
            if d > 0.1 * max(1.0, ref.float().abs().max().item()):
                continue
            res.append((timeit(lambda: gu_gemm(x, w, cfg)), cfg, d))
        except Exception:
            continue
    res.sort()
    for t, cfg, d in res[:3]:
        print(f"   triton fused {t:.3f} ms ({flops/t/1e9:.0f} TFLOP/s) cfg(BM,BN,BK,ST,NW,GM)={cfg} maxdiff {d:.3f}  -> {'WIN' if t < t_ref else 'loss'} {100*(t_ref/t-1):+.1f}%", flush=True)
