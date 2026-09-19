"""Triton causal GQA flash-attention forward for prefill vs torch SDPA (FA2) on the H100.

q [B, S, NH, HD] token-major (what _qkv_post_prefill_kernel writes), k/v cache [B, NKV, cap, HD],
out [B, S, NH, HD] token-major, i.e. already [T, NH*HD] with no transpose copy.

GPU: python tests/flash_prefill.py             sweep configs, report ms / TFLOP/s vs SDPA
CPU: python tests/flash_prefill.py --check     interpreter logic check (fp32, tiny)
"""
import itertools, math, os, sys

CHECK = "--check" in sys.argv
if CHECK:
    os.environ["TRITON_INTERPRET"] = "1"  # must precede the triton import
import torch, triton, triton.language as tl


@triton.jit
def _flash_prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, S, cap, qk_scale,
    NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    num_m = tl.num_programs(0)
    m_blk = num_m - 1 - pid_m                      # heaviest (longest causal prefix) blocks first
    b = bh // NH
    h = bh % NH
    kvh = h // (NH // NKV)
    rm = m_blk * BM + tl.arange(0, BM)
    d = tl.arange(0, HD)
    rmask = rm < S
    q_off = ((b * S + rm[:, None]).to(tl.int64) * NH + h) * HD + d[None, :]
    q = tl.load(q_ptr + q_off, mask=rmask[:, None], other=0.0)
    kv_base = (b * NKV + kvh).to(tl.int64) * cap * HD
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    hi = tl.minimum((m_blk + 1) * BM, S)           # causal: keys < end of this query block
    for n0 in range(0, hi, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < S
        k = tl.load(k_ptr + kv_base + rn[:, None] * HD + d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        valid = (rm[:, None] >= rn[None, :]) & nmask[None, :]
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + kv_base + rn[:, None] * HD + d[None, :], mask=nmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + q_off, o.to(o_ptr.dtype.element_ty), mask=rmask[:, None])


def flash_prefill(q, kc, vc, S, cfg=(128, 64, 8, 3)):
    """q [B, S, NH, HD]; kc/vc [B, NKV, cap, HD] (positions 0..S-1 valid) -> [B, S, NH, HD]"""
    B, _, NH, HD = q.shape
    NKV, cap = kc.shape[1], kc.shape[2]
    BM, BN, NW, ST = cfg
    out = torch.empty_like(q)
    _flash_prefill_kernel[(triton.cdiv(S, BM), B * NH)](
        q, kc, vc, out, S, cap, HD ** -0.5 * 1.4426950408889634,
        NH=NH, NKV=NKV, HD=HD, BM=BM, BN=BN, num_warps=NW, num_stages=ST,
    )
    return out


def sdpa_ref(q, kc, vc, S):
    qt = q.transpose(1, 2)                                              # [B, NH, S, HD]
    o = torch.nn.functional.scaled_dot_product_attention(qt, kc[:, :, :S], vc[:, :, :S], is_causal=True, enable_gqa=True)
    return o.transpose(1, 2)                                            # [B, S, NH, HD]


def _one(B, S, cfg):
    """time one config (own process: Triton 3.1 can abort the interpreter at compile time)"""
    bf = torch.bfloat16
    NH, NKV, HD = 32, 8, 128
    cap = S + 128
    q = torch.randn(B, S, NH, HD, device="cuda", dtype=bf)
    kc = torch.randn(B, NKV, cap, HD, device="cuda", dtype=bf); vc = torch.randn_like(kc)
    flops = 4 * B * NH * (S * (S + 1) / 2) * HD
    ref = sdpa_ref(q, kc, vc, S)

    def tm(fn, n=8):
        for _ in range(3): fn()
        torch.cuda.synchronize(); s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_.record()
        for _ in range(n): fn()
        e_.record(); torch.cuda.synchronize(); return s_.elapsed_time(e_) / n

    out = flash_prefill(q, kc, vc, S, cfg)
    d = (out.float() - ref.float()).abs().max().item()
    t_ref = tm(lambda: sdpa_ref(q, kc, vc, S))
    t = tm(lambda: flash_prefill(q, kc, vc, S, cfg))
    print(f"RESULT B{B} S{S} cfg{cfg}: triton {t:.3f} ms ({flops/t/1e9:.0f} TFLOP/s) SDPA {t_ref:.3f} ms ({flops/t_ref/1e9:.0f}) maxdiff {d:.4f} -> {100*(t_ref/t-1):+.1f}%")


if __name__ == "__main__":
    import subprocess
    if CHECK:
        torch.manual_seed(0)
        B, S, NH, NKV, HD, cap = 2, 45, 8, 2, 32, 64
        q = torch.randn(B, S, NH, HD); kc = torch.randn(B, NKV, cap, HD); vc = torch.randn(B, NKV, cap, HD)
        for cfg in ((16, 16, 4, 2), (32, 16, 4, 2)):
            d = (flash_prefill(q, kc, vc, S, cfg) - sdpa_ref(q, kc, vc, S)).abs().max().item()
            print(f"cfg {cfg}: max|diff| {d:.2e}"); assert d < 1e-4
        print("FLASH CHECK OK")
    elif "--one" in sys.argv:
        a = sys.argv[sys.argv.index("--one") + 1:]
        _one(int(a[0]), int(a[1]), tuple(int(x) for x in a[2:6]))
    else:
        shapes = [tuple(map(int, x.split(","))) for x in (sys.argv[1:] or ["4,2048"])]
        for B, S in shapes:
            for cfg in itertools.product((128, 64), (64, 128), (4, 8), (2, 3)):
                try:
                    r = subprocess.run([sys.executable, __file__, "--one", str(B), str(S), *map(str, cfg)], capture_output=True, text=True, timeout=180)
                    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
                    print(line[0] if line else f"B{B} S{S} cfg{cfg}: FAILED (rc={r.returncode})", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"B{B} S{S} cfg{cfg}: timeout", flush=True)
