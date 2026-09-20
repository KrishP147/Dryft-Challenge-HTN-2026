"""Simulate lockstep batched n-gram speculation: steps = max over sequences. Real greedy outputs."""
import os, random, statistics, sys
import numpy as np, torch
os.environ["ENGINE_SPEC"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import Engine
from transformers import AutoTokenizer
M = "/workspace/model"; tok = AutoTokenizer.from_pretrained(M); eng = Engine(M)
from huggingface_hub import hf_hub_download; import pyarrow.parquet as pq
p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset")
wiki_ids = tok(" ".join(pq.read_table(p).column("text").to_pylist()), add_special_tokens=False)["input_ids"]
import pydoc_data.topics as _t
pydoc_ids = tok(" ".join(_t.topics.values()), add_special_tokens=False)["input_ids"]
CORPORA = {"wiki": wiki_ids, "pydoc": pydoc_ids}
ids = wiki_ids  # default / backward compat for any stray reference

def draft(hist, K):
    for n in (3, 2, 1):
        if len(hist) < n + 1: continue
        key = hist[-n:]
        for i in range(len(hist) - n - 1, -1, -1):
            if hist[i:i + n] == key: return hist[i + n:i + n + K]
    return []

def steps_for(prompt, out, K):
    hist = list(prompt) + [out[0]]; i = 1; steps = 0
    while i < len(out):
        d = draft(hist, K); a = 0
        while a < len(d) and i + a < len(out) and d[a] == out[i + a]: a += 1
        adv = min(a + 1, len(out) - i); hist += out[i:i + adv]; i += adv; steps += 1
    return steps

import numpy as np

# (corpus, B, S, n, Ks). Default sweeps the SPEC_MIN_N crossover (x vs n) and the
# genre gap (wiki vs pydoc) at the one batch the hidden set actually uses (B<=16).
# Swap in other (B, n, Ks) tuples for ad hoc sweeps -- e.g. small-batch CV checks
# at (4, 2048, 32, (3,5,7)) or (16, 512, 128, (1,2,3)).
SHAPES = (
    ("wiki", 16, 512, 512, (1, 2, 3, 6)), ("wiki", 16, 512, 256, (1, 2, 3, 6)),
    ("wiki", 16, 512, 192, (6,)), ("wiki", 16, 512, 128, (6,)),
    ("wiki", 16, 512, 96, (6,)), ("wiki", 16, 512, 64, (6,)),
    ("pydoc", 16, 512, 512, (1, 2, 3, 6)), ("pydoc", 16, 512, 256, (1, 2, 3, 6)),
    ("pydoc", 16, 512, 192, (6,)), ("pydoc", 16, 512, 128, (6,)),
    ("pydoc", 16, 512, 96, (6,)), ("pydoc", 16, 512, 64, (6,)),
)
TRIALS = 10

for corpus_name, B, S, n, Ks in SHAPES:
    cids = CORPORA[corpus_name]
    res = {K: [] for K in Ks}
    for trial in range(TRIALS):
        r = random.Random(trial * 13 + B)
        prompts = [cids[o:o + S] for o in (r.randrange(0, len(cids) - S) for _ in range(B))]
        out = torch.tensor(list(eng.generate(prompts, n))).T.tolist()
        for K in Ks:
            st = [steps_for(prompts[b], out[b], K) + 1 for b in range(B)]  # +1: first token from prefill counts as a plain step? (no) -> keep as decode steps
            res[K].append(max(st) - 1)
    for K in Ks:
        vals = res[K]
        m = statistics.mean(vals); sd = statistics.pstdev(vals)
        xfac = [(n - 1) / v for v in vals]
        # (max-min)/median per group of 5, like tests/spread.py's actual 25% gate
        groups = [vals[i:i + 5] for i in range(0, len(vals) - len(vals) % 5, 5)]
        gspreads = [100 * (max(g) - min(g)) / statistics.median(g) for g in groups if statistics.median(g) > 0]
        gspread_str = ", ".join(f"{gs:.1f}%" for gs in gspreads) if gspreads else "n/a"
        print(f"corpus={corpus_name:6s} B={B:2d} n={n:4d} K={K}: decode steps {m:6.1f} (plain {n-1}) -> x{(n-1)/m:.2f} "
              f"(per-seq mean x{statistics.mean(xfac):.2f}), pstdev/mean spread {sd/m*100:.1f}%, "
              f"group(5) (max-min)/median spread: {gspread_str}", flush=True)
