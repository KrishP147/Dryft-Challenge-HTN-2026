"""GPU: interleaved ragged groups must match separate group generation."""
import os
import sys
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

assert torch.cuda.is_available(), "requires a CUDA GPU"
assert os.environ.get("TRITON_INTERPRET") != "1", "requires compiled CUDA kernels"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from engine import CAP_GRAN, Engine  # noqa: E402

torch.manual_seed(0)
cfg = Qwen3Config(
    vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    rope_theta=5e6, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=4096,
)
model = Qwen3ForCausalLM(cfg).eval()
seqs = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15]]
n = 6

with tempfile.TemporaryDirectory() as model_dir:
    model.save_pretrained(model_dir)
    engine = Engine(model_dir)
    ragged = list(engine.generate(seqs, n))
    grouped = {
        3: list(engine.generate([seqs[0], seqs[2]], n)),
        4: list(engine.generate([seqs[1]], n)),
        5: list(engine.generate([seqs[3]], n)),
    }
    for t in range(n):
        expected = [grouped[3][t][0], grouped[4][t][0],
                    grouped[3][t][1], grouped[5][t][0]]
        assert ragged[t] == expected, (t, ragged[t], expected)
    one_token = [1, 2, 3, 4, 5, 6]
    assert len(list(engine.generate([one_token], 1))) == 1
    cap = -(-(len(one_token) + 1) // CAP_GRAN) * CAP_GRAN
    assert engine.states[(1, cap, len(one_token))].graph is None

print("ragged interleaving and single-token graph skip OK")
