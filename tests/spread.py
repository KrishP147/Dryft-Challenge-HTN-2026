"""Spread of B1 timings over many natural-text windows, in groups of 5 (like an official run)."""
import os, random, statistics, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import Engine
from transformers import AutoTokenizer
M = "/workspace/model"; tok = AutoTokenizer.from_pretrained(M); eng = Engine(M)
B, S, n = map(int, sys.argv[2:5]); which = sys.argv[1]
if which == "wiki":
    from huggingface_hub import hf_hub_download; import pyarrow.parquet as pq
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset")
    text = " ".join(pq.read_table(p).column("text").to_pylist())
elif which == "code":
    import glob
    text = "\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))[:400])
else:
    import pydoc_data.topics as t; text = " ".join(t.topics.values())
ids = tok(text, add_special_tokens=False)["input_ids"]; print(which, "corpus tokens", len(ids))
rng = random.Random(5)
def go(seed):
    r = random.Random(seed); pr = [ids[o:o + S] for o in (r.randrange(0, len(ids) - S) for _ in range(B))]
    t0 = time.perf_counter(); list(eng.generate(pr, n)); return time.perf_counter() - t0
go(0); go(1)
ts = [go(100 + i) for i in range(30)]
groups = [ts[i:i + 5] for i in range(0, 30, 5)]
print("median ms", round(statistics.median(ts) * 1e3, 1), "min", round(min(ts) * 1e3, 1), "max", round(max(ts) * 1e3, 1))
for g in groups:
    a = np.array(g); med = np.median(a)
    print(f"  group: (max-min)/med {100*(a.max()-a.min())/med:5.1f}%  (p90-p10)/p50 {100*(np.percentile(a,90)-np.percentile(a,10))/med:5.1f}%  std/mean {100*a.std()/a.mean():5.1f}%")
