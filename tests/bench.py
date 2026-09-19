"""GPU bench + correctness for the engine, mimicking the platform.

usage: python tests/bench.py [--model /workspace/model] [--shapes 1,512,32 4,2048,32 16,512,128]
                             [--samples 5] [--no-check]
Prints per-shape median tok/s, TTFT, TPOT, spread, peak mem, and the teacher-forced
correctness margin vs an HF/sdpa baseline (must stay <= 2.0 logits).
"""
import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/workspace/model")
ap.add_argument("--shapes", nargs="*", default=["1,512,32", "4,2048,32", "16,512,128"])
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--no-check", action="store_true")
args = ap.parse_args()

from engine import Engine  # noqa: E402

t0 = time.time()
eng = Engine(args.model)
torch.cuda.synchronize()
print(f"load+init {time.time() - t0:.1f}s", flush=True)


def prompts(B, S, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (B, S), generator=g).tolist()


def run(ids, n):
    t0 = time.perf_counter()
    out, first = [], None
    for step in eng.generate(ids, n):
        if first is None:
            first = time.perf_counter() - t0
        out.append(step)
    return time.perf_counter() - t0, first, out


baseline = None
worst = {}
results = []
for spec in args.shapes:
    B, S, n = map(int, spec.split(","))
    run(prompts(B, S, 0), n)  # warmup, same shape
    ts, tf, outs = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for k in range(args.samples):
        ids = prompts(B, S, k + 1)
        total, first, out = run(ids, n)
        assert len(out) == n and all(len(x) == B for x in out)
        ts.append(total)
        tf.append(first)
        outs.append((ids, out))
    med = statistics.median(ts)
    spread = (max(ts) - min(ts)) / med
    tpot = (med - statistics.median(tf)) / max(n - 1, 1)
    tps = B * n / med
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"B{B} {S}->{n}: {tps:8.1f} tok/s  total {med*1e3:7.1f} ms  ttft {statistics.median(tf)*1e3:6.1f} ms  "
        f"tpot {tpot*1e3:5.2f} ms  spread {spread*100:4.1f}%  mem {mem:.1f} GiB",
        flush=True,
    )
    results.append(tps)

    if not args.no_check:
        if baseline is None:
            from transformers import AutoModelForCausalLM

            baseline = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
            ).eval().cuda()
        ids, out = outs[0]
        toks = torch.tensor(out).T.cuda()  # [B, n]
        full = torch.cat([torch.tensor(ids).cuda(), toks], 1)[:, :-1]
        worst_gap, bad = 0.0, 0
        with torch.inference_mode():
            for b in range(B):
                logits = baseline(full[b : b + 1], logits_to_keep=n).logits[0].float()  # [n, V]
                assert torch.isfinite(logits).all().item(), (spec, b, "nonfinite baseline logits")
                mx = logits.max(-1).values
                got = logits.gather(-1, toks[b].unsqueeze(-1)).squeeze(-1)
                gap = (mx - got)
                assert torch.isfinite(gap).all().item(), (spec, b, "nonfinite logit gap")
                worst_gap = max(worst_gap, gap.max().item())
                bad += (gap > 2.0).sum().item()
        print(f"   correctness: worst gap {worst_gap:.3f} logits, positions > 2.0: {bad}", flush=True)
        assert bad == 0, f"{spec}: {bad} generated tokens exceeded the 2-logit tolerance"
        worst[spec] = worst_gap

gm = 1.0
for r in results:
    gm *= r
print(f"geomean(public) {gm ** (1 / len(results)):.1f} tok/s")
