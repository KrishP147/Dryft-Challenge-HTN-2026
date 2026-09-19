"""Teacher-force one torch-ops token stream through fused ops on NATURAL prompts, any B; locate spikes.

usage: python tests/trace_diff2.py B S n [repeats]
Repeats the fused pass to see whether the divergence is deterministic (a numeric difference) or
changes run to run (a race). Env ENGINE_PDL=0 / ENGINE_OFF=... apply to the fused pass.
"""
import os, random, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
os.environ["ENGINE_SPEC"] = "0"
import engine as E
from transformers import AutoTokenizer

B, S, n = map(int, sys.argv[1:4])
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
eng = E.Engine("/workspace/model")
fused = eng.ops
torch_ops = E._TorchOps(eng)
import pydoc_data.topics as t
corpus = AutoTokenizer.from_pretrained("/workspace/model")(" ".join(t.topics.values()), add_special_tokens=False)["input_ids"]
r = random.Random(1 * 7919 + B * 31 + S)
ids = [corpus[o:o + S] for o in (r.randrange(0, len(corpus) - S) for _ in range(B))]


def run(ops, forced):
    eng.ops = ops
    eng.states.clear()
    st = eng._state(B, S + n + 8, graph=False)
    eng._dbg = []
    st.pos.fill_(S)
    eng._prefill(torch.tensor(ids, device="cuda"), st)
    for j in range(len(forced)):
        st.tok.copy_(forced[j])
        eng._decode_body(st)
    lg, eng._dbg = eng._dbg, None
    return lg


eng.ops = torch_ops
toks = torch.tensor([x for x in eng.generate(ids, n)], device="cuda")  # [n, B]
forced = [toks[j] for j in range(n - 1)]
lt = run(torch_ops, forced)
prev = None
for rep in range(reps):
    lf = run(fused, forced)
    diffs = torch.stack([(a - b).abs().max(-1).values for a, b in zip(lt, lf)])  # [steps, B]
    top = diffs.flatten().topk(5)
    print(f"rep {rep}: worst |dlogit| {diffs.max().item():.3f}; top5 (step,seq,val):",
          [(int(i) // B, int(i) % B, round(v, 2)) for i, v in zip(top.indices, top.values.tolist())], flush=True)
    if prev is not None:
        print("   identical to previous fused rep:", bool(torch.equal(diffs, prev)))
    prev = diffs
