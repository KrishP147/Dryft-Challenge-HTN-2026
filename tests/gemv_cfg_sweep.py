"""A/B GEMV tile configs and PF depth inside the real decode step, with PDL on.

Isolated microbenches have twice pointed the wrong way here (attention nsplit, and the
GEMV configs pre-date the EVENK path), because PF only exists under PDL and PDL only
exists in a chain. So this sweeps by mutating fused.GEMM_CFG / fused.SILU_CFG / fused.PF,
re-capturing the decode graph, and timing graph replays -- the same number budget.py
reports, which is what the score is made of.

usage: MODEL=/workspace/model python tests/gemv_cfg_sweep.py <shape> [B,S,n ...]
       shape in {qkv, o, down, gate_up, pf}
"""
import gc
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
import fused  # noqa: E402
from engine import CAP_GRAN, Engine  # noqa: E402

WHICH = sys.argv[1] if len(sys.argv) > 1 else "o"
SHAPES = sys.argv[2:] or ["1,512,32", "16,512,128"]
REPLAYS = 40
WEIGHT = {"1,512,32": 0.14, "4,2048,32": 0.45, "16,512,128": 0.44}  # fitted score sensitivity

KEY = {  # which entry of which dict this sweep replaces
    "qkv": ("GEMM_CFG", (6144, 2560)),
    "o": ("GEMM_CFG", (2560, 4096)),
    "down": ("GEMM_CFG", (2560, 9728)),
    "gate_up": ("SILU_CFG", (9728, 2560)),
}


def grid(which):
    if which == "pf":  # tl.arange(0, PF): powers of two only
        return [("PF", pf) for pf in (0, 4, 8, 16, 32, 64)]
    dct, key = KEY[which]
    cur = getattr(fused, dct)[key]
    out = []
    if dct == "SILU_CFG":  # (BN, BK, ST, NW)
        for BN, BK, ST, NW in itertools.product((32, 64), (64, 128, 256), (3, 4, 5, 6), (4, 8)):
            if BN * BK <= 16384:
                out.append((dct, (BN, BK, ST, NW)))
    else:  # (BN, BK, SK, ST, NW) - SK is fixed: linear_add_norm only fuses when SK > 1
        SK = cur[2]
        for BN, BK, ST, NW in itertools.product((32, 64, 128), (64, 128, 256), (3, 4, 5, 6, 8), (4, 8)):
            if BN * BK <= 16384 and (key[1] // SK) % BK == 0:
                out.append((dct, (BN, BK, SK, ST, NW)))
    return [c for c in out if c[1] != cur] + [(dct, cur)]  # current config last, as the baseline


def apply(cand):
    kind, val = cand
    if kind == "PF":
        fused.PF = val
    else:
        getattr(fused, kind)[KEY[WHICH][1]] = val


eng = Engine(os.environ.get("MODEL", "/workspace/model"))
g = torch.Generator().manual_seed(0)
prompts = {s: torch.randint(1000, 100000, tuple(int(x) for x in s.split(",")[:2]), generator=g).tolist()
           for s in SHAPES}

results = {}
for cand in grid(WHICH):
    apply(cand)
    per_shape = {}
    try:
        for spec in SHAPES:
            B, S, n = map(int, spec.split(","))
            eng.states.clear()
            gc.collect()
            torch.cuda.empty_cache()
            # re-capturing into a pool whose previous graphs were just freed trips
            # "use_count > 0" in the caching allocator: give each config a fresh pool
            eng.pool = torch.cuda.graph_pool_handle()
            with torch.inference_mode():
                list(eng.generate(prompts[spec], n))  # capture the graph with this config
                cap = -(-(S + n) // CAP_GRAN) * CAP_GRAN
                # state key is (B, cap, W, slot, decode_graph) and slot varies: pick the
                # one graphed decode state for this (B, cap)
                cands = [v for k, v in eng.states.items()
                         if k[0] == B and k[1] == cap and k[2] == 1 and v.graph is not None]
                assert cands and S + REPLAYS <= cap, f"no graphed state for B{B} cap{cap}"
                st = cands[-1]
                st.pos.fill_(S)
                for _ in range(5):
                    st.graph.replay()
                torch.cuda.synchronize()
                st.pos.fill_(S)
                s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s_.record()
                for _ in range(REPLAYS):
                    st.graph.replay()
                e_.record()
                torch.cuda.synchronize()
                per_shape[spec] = s_.elapsed_time(e_) / REPLAYS * 1e3
    except Exception as ex:
        print(f"   {cand[1]}: FAILED {type(ex).__name__}: {str(ex)[:70]}", flush=True)
        continue
    # score-weighted: lower is better, weights renormalised over the shapes measured
    wt = sum(WEIGHT.get(s, 0.3) for s in SHAPES)
    score = sum(WEIGHT.get(s, 0.3) * per_shape[s] for s in SHAPES) / wt
    results[cand[1]] = (score, dict(per_shape))
    print(f"   {str(cand[1]):28s} weighted {score:8.1f} us   "
          + "  ".join(f"{s}:{v:7.1f}" for s, v in per_shape.items()), flush=True)

base = list(results.values())[-1][0] if results else 0
print(f"\n=== {WHICH}: best first (baseline = last row, {base:.1f} us)")
for cfg, (sc, per) in sorted(results.items(), key=lambda kv: kv[1][0])[:8]:
    print(f"  {str(cfg):28s} {sc:8.1f} us  {100*(base/sc-1):+5.2f}%  "
          + "  ".join(f"{s}:{v:7.1f}" for s, v in per.items()))
