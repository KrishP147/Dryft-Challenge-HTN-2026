"""GPU integration check for selective fused self-test fallback."""
import os
import sys
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

assert torch.cuda.is_available(), "requires a CUDA GPU"
assert os.environ.get("TRITON_INTERPRET") != "1", "requires compiled CUDA kernels"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from engine import Engine  # noqa: E402
from fused import TritonOps  # noqa: E402

torch.manual_seed(0)
cfg = Qwen3Config(
    vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    rope_theta=5e6, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=4096,
)
model = Qwen3ForCausalLM(cfg).eval()
original_linear = TritonOps.linear


def bad_gemv(self, x, w):
    out = original_linear(self, x, w)
    # Prefill logits stay finite; only a later decode step fails.
    if self.use_gemv and x.shape[0] == 2 and w is self.e.layers[0].wqkv:
        return torch.full_like(out, float("nan"))
    return out


with tempfile.TemporaryDirectory() as model_dir:
    model.save_pretrained(model_dir)
    try:
        TritonOps.linear = bad_gemv
        engine = Engine(model_dir)
    finally:
        TritonOps.linear = original_linear
    assert isinstance(engine.ops, TritonOps)
    assert not engine.ops.use_gemv, "self-test should retain fusion and disable bad GEMV"

print("selective fused fallback OK")
