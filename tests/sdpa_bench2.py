import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
bf = torch.bfloat16
def t(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n * 1e3
for B, S in ((4, 2048), (16, 512), (1, 512), (1, 2048)):
    qt = torch.randn(B, S, 32, 128, device="cuda", dtype=bf)         # token-major physical
    q = qt.transpose(1, 2)                                            # logical [B,H,S,D] view
    cap = S + 128
    kc = torch.randn(B, 8, cap, 128, device="cuda", dtype=bf); vc = torch.randn_like(kc)
    k, v = kc[:, :, :S], vc[:, :, :S]
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    def flash():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    def cudnn():
        ke, ve = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
    o = cudnn()
    print(B, S, f"flash {t(flash):.1f} us | cudnn(+repeat_interleave) {t(cudnn):.1f} us | out strides {tuple(o.transpose(1,2).stride())} contiguous={o.transpose(1,2).is_contiguous()} | maxdiff {(o.float()-ref.float()).abs().max().item():.4f}")
