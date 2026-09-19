"""Kernel-level profile of full generation, prefill, or decode graph replays.

usage: python tests/prof.py B S n [--phase full|prefill|decode]
"""
import argparse
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import CAP_GRAN, Engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("B", type=int)
ap.add_argument("S", type=int)
ap.add_argument("n", type=int)
ap.add_argument("--phase", choices=("full", "prefill", "decode"), default="full")
args = ap.parse_args()
B, S, n = args.B, args.S, args.n
if B < 1 or S < 1 or n < 1 or (args.phase == "decode" and n < 2):
    ap.error("B, S, n must be positive; decode needs n >= 2")
eng = Engine(os.environ.get("MODEL", "/workspace/model"))
g = torch.Generator().manual_seed(0)
ids = torch.randint(1000, 100000, (B, S), generator=g).tolist()
for _ in range(2):
    list(eng.generate(ids, n))

if args.phase != "full":
    cap = -(-(S + n) // CAP_GRAN) * CAP_GRAN
    st = eng._state(B, cap, graph=args.phase == "decode", slot=S)
    ids_device = torch.tensor(ids, dtype=torch.long, device=eng.dev)
    if args.phase == "decode":
        with torch.inference_mode():
            eng._prefill(ids_device, st)
            st.pos.fill_(S)

torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
    if args.phase == "full":
        list(eng.generate(ids, n))
    else:
        with torch.inference_mode():
            if args.phase == "prefill":
                eng._prefill(ids_device, st)
            else:
                for _ in range(n - 1):
                    if st.graph is not None:
                        st.graph.replay()
                    else:
                        eng._decode_body(st)
    torch.cuda.synchronize()
rows = [e for e in p.key_averages() if e.device_time_total > 0 and e.self_device_time_total > 0]
rows.sort(key=lambda e: -e.self_device_time_total)
tot = sum(e.self_device_time_total for e in rows)
if not tot:
    raise RuntimeError("profiler recorded no CUDA kernel time")
print(f"{args.phase} total device {tot/1e3:.1f} ms")
for e in rows[:22]:
    print(f"{e.self_device_time_total/1e3:8.2f} ms {100*e.self_device_time_total/tot:5.1f}%  n={e.count:5d} avg={e.self_device_time_total/e.count:7.1f} us  {e.key[:80]}")
