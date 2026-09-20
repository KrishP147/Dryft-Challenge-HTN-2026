"""Idea 1 probe: how often does an early-exit (logit-lens) draft agree with full greedy?

Runs a real decode loop and, at each step, also reads out argmax from the residual stream
after k layers (final RMSNorm + lm_head applied to h_k).  Prints per-k agreement with the
true next token -- i.e. the per-token acceptance rate p a self-speculative drafter would get.

usage: python tests/spec_earlyexit.py [--model /workspace/model] [--shape 4,2048,128] [--corpus pydoc]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/workspace/model")
ap.add_argument("--shape", default="4,2048,128")
ap.add_argument("--corpus", default="pydoc", choices=["pydoc", "code"])
ap.add_argument("--ks", default="")
args = ap.parse_args()

from engine import Engine  # noqa: E402

B, S, N = (int(x) for x in args.shape.split(","))
eng = Engine(args.model)
L = eng.L
KS = [int(x) for x in args.ks.split(",") if x] or [L // 6, L // 4, L // 3, L // 2, (2 * L) // 3, (3 * L) // 4, (5 * L) // 6, L - 2]
KS = sorted(set(k for k in KS if 0 < k < L))
print(f"L={L} ks={KS}", flush=True)

from transformers import AutoTokenizer  # noqa: E402

if args.corpus == "code":
    import glob

    text = chr(10).join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))[:300])
else:
    import pydoc_data.topics as t

    text = " ".join(t.topics.values())
corpus = AutoTokenizer.from_pretrained(args.model)(text, add_special_tokens=False)["input_ids"]
import random  # noqa: E402

r = random.Random(1234)
ids = [corpus[o : o + S] for o in (r.randrange(0, len(corpus) - S) for _ in range(B))]

ops = eng.ops
hit = {k: 0 for k in KS}
tot = 0


def step(st):
    """one decode step; returns (true_argmax, {k: early_argmax})"""
    h = eng.embed[st.tok]
    a = ops.rms(h, eng.layers[0].ln1)
    last = eng.L - 1
    early = {}
    for i, l in enumerate(eng.layers):
        kc, vc = st.kc[i], st.vc[i]
        q = ops.qkv_post(ops.linear(a, l.wqkv), l, kc, vc, st.B, 1, st.pos)
        o = ops.attn_decode(q, kc, vc, st.pos, 1)
        h, a2 = ops.linear_add_norm(o, l.wo, h, l.ln2)
        m = ops.gate_up_silu(a2, l.wgu)
        h, a = ops.linear_add_norm(m, l.wd, h, eng.layers[i + 1].ln1 if i < last else eng.norm)
        if (i + 1) in hit:
            early[i + 1] = ops.linear(ops.rms(h, eng.norm), eng.lm_head).argmax(-1)
    return ops.linear(a, eng.lm_head).argmax(-1), early


cap = -(-(S + N) // 128) * 128
if cap > eng.cos.shape[0]:
    eng._grow_rope(cap)
st = eng._state(B, cap, slot=S, decode_graph=False)
ids_t = torch.tensor(ids, dtype=torch.long, device=eng.dev)
eng._prefill(ids_t, st)
st.pos.fill_(S)
with torch.inference_mode():
    for t_ in range(N):
        true, early = step(st)
        for k, v in early.items():
            hit[k] += int((v == true).sum())
        tot += B
        st.tok.copy_(true)
        st.pos.add_(1)

print(f"\nshape {B},{S},{N} corpus={args.corpus}  tokens={tot}")
# byte model: draft cost = k/L of transformer weights + lm_head; verify = 1 full step
WT = 7.26  # GB of transformer-block weights (8.04 total - 0.778 lm_head)
LM = 0.778
print(f"{'k':>4} {'p(accept)':>10} {'draft cost':>11}  break-even W=2/3/4 (E[tok]/cost)")
for k in KS:
    p = hit[k] / tot
    c = (k / L * WT + LM) / (WT + LM)
    for W in (2, 3, 4):
        e = sum(p ** j for j in range(1, W + 1)) + 1.0  # accepted chain + bonus token
        cost = W * c + 1.0
        print(f"{k:>4} {p:>10.3f} {c:>11.3f}  W={W}: {e / cost:>5.2f}x")
