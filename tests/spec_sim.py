"""Offline estimate of n-gram (prompt-lookup) speculation gain on real greedy outputs."""
import os, sys, random
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import Engine
from transformers import AutoTokenizer

M = "/workspace/model"
tok = AutoTokenizer.from_pretrained(M)
eng = Engine(M)

def corpora():
    out = {}
    import pydoc_data.topics as t
    out["pydoc"] = " ".join(t.topics.values())
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
        out["wiki"] = " ".join(pq.read_table(p).column("text").to_pylist())
    except Exception as e:
        print("no wikitext:", repr(e)[:100])
    import glob
    out["code"] = "\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/json/*.py") + glob.glob("/usr/lib/python3*/argparse.py"))[:6]) or open(__import__("json").__file__).read()
    return out

def draft(hist, K, ngrams):
    for n in ngrams:
        if len(hist) < n + 1: continue
        key = hist[-n:]
        for i in range(len(hist) - n - 1, -1, -1):
            if hist[i:i + n] == key:
                return hist[i + n:i + n + K]
    return []

VARIANTS = [(K, ng) for K in (3, 4, 6, 8) for ng in ((3, 2), (4, 3, 2), (2,), (3, 2, 1), (5, 3, 2, 1))]
for name, text in corpora().items():
    ids = tok(text, add_special_tokens=False)["input_ids"]
    rng = random.Random(1)
    runs = []
    for trial in range(10):
        s0 = rng.randrange(0, max(1, len(ids) - 700))
        prompt = ids[s0:s0 + 512]
        runs.append((prompt, [x[0] for x in eng.generate([prompt], 128)]))
    res = []
    for K, ng in VARIANTS:
        st_ = tk = 0
        for prompt, out in runs:
            hist = list(prompt); i = 0
            while i < len(out):
                d = draft(hist, K, ng); a = 0
                while a < len(d) and i + a < len(out) and d[a] == out[i + a]: a += 1
                adv = min(a + 1, len(out) - i); hist += out[i:i + adv]; i += adv; st_ += 1; tk += adv
        res.append((tk / st_, K, ng))
    res.sort(reverse=True)
    print(name, [f"{r:.2f} K={k} n={g}" for r, k, g in res[:5]], "| worst", f"{res[-1][0]:.2f}", flush=True)
