import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
bf = torch.bfloat16
def t(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n * 1e3
for B, S in ((4, 2048), (16, 512), (1, 512), (1, 4096)):
    q = torch.randn(B, 32, S, 128, device="cuda", dtype=bf); k = torch.randn(B, 8, S, 128, device="cuda", dtype=bf); v = torch.randn_like(k)
    flops = 4 * B * 32 * S * S * 128 / 2
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        tf = t(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
    res = {"flash": tf}
    try:
        ke, ve = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            o = F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
            res["cudnn(+repeat)"] = t(lambda: F.scaled_dot_product_attention(q, ke, ve, is_causal=True))
            res["cudnn_only"] = t(lambda: F.scaled_dot_product_attention(q, ke, ve, is_causal=True))
            res["cudnn_maxdiff"] = (o.float() - ref.float()).abs().max().item()
    except Exception as ex:
        res["cudnn"] = repr(ex)[:80]
    print(B, S, {k_: (round(v_, 1) if isinstance(v_, float) else v_) for k_, v_ in res.items()}, f"flash TFLOPs {flops/tf/1e6:.0f}")
