"""GPU: last-query prefill attention vs the final row of causal SDPA."""
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "requires a CUDA GPU"
torch.manual_seed(0)
for B, S in ((1, 512), (4, 2048), (16, 512)):
    q = torch.randn(B, 32, S, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, 8, S, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    full = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    last = F.scaled_dot_product_attention(q[:, :, -1:, :], k, v, enable_gqa=True)
    assert torch.isfinite(last).all().item(), (B, S)
    diff = (full[:, :, -1:, :].float() - last.float()).abs().max().item()
    print(f"B{B} S{S}: last-query max|diff|={diff:.5f}")
    assert diff < 0.05, (B, S, diff)

print("last-query prefill OK")
