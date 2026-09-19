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
ids = tok(" ".join(pq.read_table(p).column("text").to_pylist()), add_special_tokens=False)["input_ids"]

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

for B, S, n, Ks in ((4, 2048, 32, (3, 5, 7)), (16, 512, 128, (1, 2, 3)), (16, 512, 32, (1, 2))):
    res = {K: [] for K in Ks}
    for trial in range(4):
        r = random.Random(trial * 13 + B)
        prompts = [ids[o:o + S] for o in (r.randrange(0, len(ids) - S) for _ in range(B))]
        out = torch.tensor(list(eng.generate(prompts, n))).T.tolist()
        for K in Ks:
            st = [steps_for(prompts[b], out[b], K) + 1 for b in range(B)]  # +1: first token from prefill counts as a plain step? (no) -> keep as decode steps
            res[K].append(max(st) - 1)
    for K in Ks:
        m = statistics.mean(res[K]); sd = statistics.pstdev(res[K])
        print(f"B={B:2d} n={n:3d} K={K}: decode steps {m:5.1f} (plain {n-1}) -> x{(n-1)/m:.2f}, step-count spread over samples {sd/m*100:.1f}%", flush=True)
