"""Ceiling check for multi-candidate (tree) drafting, on REAL greedy outputs.

Depth-1 tree drafting verifies C alternative next-tokens per sequence in one pass instead of 1.
All C sit at the same position and attend only to the committed prefix, so the win is bounded by
  hit(C) = P(the true next token is among the C candidates)
and the extra tokens per step is exactly hit(C).  This script measures hit(C) offline so the
kernel work (per-row KV scratch slots) is only done if the ceiling justifies it.

Generates once with speculation off, then replays the drafter over the true stream.

usage: python tests/spec_tree_sim.py [--shape 4,2048,256] [--corpus prose]
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
os.environ["ENGINE_SPEC"] = "0"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/workspace/model")
ap.add_argument("--shape", default="4,2048,256")
ap.add_argument("--corpus", default="prose")
ap.add_argument("--reps", type=int, default=3, help="independent prompt draws")
args = ap.parse_args()

from engine import Engine  # noqa: E402

B, S, N = (int(x) for x in args.shape.split(","))
eng = Engine(args.model)

from transformers import AutoTokenizer  # noqa: E402

tokzr = AutoTokenizer.from_pretrained(args.model)
if args.corpus == "prose":
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".prose_cache")
    texts = [open(os.path.join(cache, f), errors="ignore").read() for f in sorted(os.listdir(cache))]
    text = "\n".join(texts)
elif args.corpus == "code":
    import glob

    text = chr(10).join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))[:300])
else:
    import pydoc_data.topics as t

    text = " ".join(t.topics.values())
corpus = tokzr(text, add_special_tokens=False)["input_ids"]
print(f"corpus {args.corpus}: {len(corpus)} tokens", flush=True)

CS = [1, 2, 3, 4, 6, 8]
hits = {(name, c): 0 for name in ("recent", "freq", "recent_bo", "freq_bo") for c in CS}
total = 0

import random  # noqa: E402

rng = random.Random(4242)
for rep in range(args.reps):
    ids = [corpus[o : o + S] for o in (rng.randrange(0, len(corpus) - S) for _ in range(B))]
    out = [[] for _ in range(B)]
    for step in eng.generate(ids, N):
        for b, t_ in enumerate(step):
            out[b].append(t_)
    for b in range(B):
        h = list(ids[b])
        # occurrence lists per n-gram, most recent last
        occ = (defaultdict(list), defaultdict(list), defaultdict(list))
        for i in range(1, len(h)):
            for n in (1, 2, 3):
                if i >= n:
                    occ[n - 1][tuple(h[i - n : i])].append(i)

        def cands(strategy, C):
            res = []
            order = (3, 2, 1) if strategy.endswith("_bo") else (3,)
            for n in order:
                key = tuple(h[len(h) - n :])
                js = occ[n - 1].get(key)
                if not js:
                    continue
                if strategy.startswith("freq"):
                    ranked = [t for t, _ in Counter(h[j] for j in js).most_common()]
                else:
                    ranked = []
                    for j in reversed(js):
                        if h[j] not in ranked:
                            ranked.append(h[j])
                for t_ in ranked:
                    if t_ not in res:
                        res.append(t_)
                    if len(res) >= C:
                        return res
            return res

        for t_ in out[b]:
            for name in ("recent", "freq", "recent_bo", "freq_bo"):
                for C in CS:
                    if t_ in cands(name, C):
                        hits[(name, C)] += 1
            total += 1
            h.append(t_)
            for n in (1, 2, 3):
                if len(h) > n:
                    occ[n - 1][tuple(h[len(h) - 1 - n : len(h) - 1])].append(len(h) - 1)
    print(f"rep {rep + 1}/{args.reps} done ({total} tokens)", flush=True)

print(f"\ndepth-1 candidate hit rate, corpus={args.corpus}, shape {B},{S},{N}, {total} tokens")
print(f"{'C':>3} " + " ".join(f"{n:>10}" for n in ("recent", "freq", "recent_bo", "freq_bo")))
for C in CS:
    print(f"{C:>3} " + " ".join(f"{hits[(n, C)] / total:>10.3f}" for n in ("recent", "freq", "recent_bo", "freq_bo")))
print("\nextra tokens/step = hit rate. Ships today: 'recent' at C=1 (3-gram, MIN_MATCH=3).")
