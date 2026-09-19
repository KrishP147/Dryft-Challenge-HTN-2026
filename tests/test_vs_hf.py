"""CPU correctness check: engine vs HF greedy on a tiny random Qwen3 (fp32)."""
import os
import sys
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))
from engine import Engine  # noqa: E402

torch.manual_seed(0)
cfg = Qwen3Config(
    vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    rope_theta=5e6, rms_norm_eps=1e-6, tie_word_embeddings=True,
    max_position_embeddings=4096,
)
model = Qwen3ForCausalLM(cfg).eval()

with tempfile.TemporaryDirectory() as d:
    model.save_pretrained(d)
    eng = Engine(d, dtype=torch.float32)
    for B, S, n in [(1, 20, 12), (3, 37, 9), (2, 130, 5)]:
        ids = torch.randint(0, 300, (B, S))
        ref = model.generate(
            ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=n, min_new_tokens=n, eos_token_id=None, pad_token_id=0,
        )[:, S:]
        out = torch.tensor(list(eng.generate(ids.tolist(), n))).T
        ok = torch.equal(ref, out)
        print(f"B={B} S={S} n={n}: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(ref, out, sep="\n")
            sys.exit(1)
    # state reuse across calls must not leak
    ids = torch.randint(0, 300, (3, 37))
    a = list(eng.generate(ids.tolist(), 9))
    b = list(eng.generate(ids.tolist(), 9))
    assert a == b, "state leak"
    # ragged fallback
    r = list(eng.generate([[1, 2, 3], [4, 5, 6, 7]], 4))
    assert len(r) == 4 and all(len(x) == 2 for x in r)
    print("all OK")
