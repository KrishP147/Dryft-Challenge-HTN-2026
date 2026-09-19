"""Prefill-only fused vs torch: per-layer residual error for one sequence (finds large-T kernel bugs).
usage: python tests/trace_prefill.py B S seq"""
import os, random, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
os.environ["ENGINE_SPEC"] = "0"
import engine as E
from transformers import AutoTokenizer
B, S, seq = map(int, sys.argv[1:4])
eng = E.Engine("/workspace/model"); fused = eng.ops; tor = E._TorchOps(eng)
import pydoc_data.topics as t
corpus = AutoTokenizer.from_pretrained("/workspace/model")(" ".join(t.topics.values()), add_special_tokens=False)["input_ids"]
r = random.Random(1 * 7919 + B * 31 + S)
ids = torch.tensor([corpus[o:o + S] for o in (r.randrange(0, len(corpus) - S) for _ in range(B))], device="cuda")
def run(ops):
    eng.ops = ops; eng.states.clear()
    st = eng._state(B, S + 8, graph=False); eng._dbg = []; eng._trace = []
    eng._prefill(ids, st)
    tr, eng._trace = eng._trace, None; lg, eng._dbg = eng._dbg[0], None
    return [x[seq * S:(seq + 1) * S].clone() if x.shape[0] == B * S else x[seq:seq + 1].clone() for x in tr], lg
ft, fl = run(fused); tt, tl_ = run(tor)
print("final-logit max |diff| per seq (top 5):", [(int(i), round(v, 2)) for i, v in zip(*[x.tolist() for x in (lambda d: d.topk(5))((fl - tl_).abs().max(-1).values)][::-1])] if False else (fl - tl_).abs().max(-1).values.topk(5))
for l, (a, b) in enumerate(zip(ft, tt)):
    d = (a - b).abs()
    rel = (d.norm() / b.norm()).item()
    pos = int(d.amax(-1).argmax()) if a.shape[0] > 1 else 0
    print(f"layer {l:2d}: rel L2 {rel:.4f}  max abs {d.max().item():7.3f} at token {pos:4d} of {a.shape[0]}  |h|max {b.abs().max().item():7.1f}")
