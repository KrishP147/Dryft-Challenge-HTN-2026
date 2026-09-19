"""Kernel-level profile of one generate() call: decode graph replays + prefill.

usage: python tests/prof.py B S n
"""
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import Engine  # noqa: E402

B, S, n = map(int, sys.argv[1:4])
eng = Engine(os.environ.get("MODEL", "/workspace/model"))
g = torch.Generator().manual_seed(0)
ids = torch.randint(1000, 100000, (B, S), generator=g).tolist()
for _ in range(2):
    list(eng.generate(ids, n))
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
    list(eng.generate(ids, n))
    torch.cuda.synchronize()
rows = [e for e in p.key_averages() if e.device_time_total > 0 and e.self_device_time_total > 0]
rows.sort(key=lambda e: -e.self_device_time_total)
tot = sum(e.self_device_time_total for e in rows)
print(f"total device {tot/1e3:.1f} ms")
for e in rows[:22]:
    print(f"{e.self_device_time_total/1e3:8.2f} ms {100*e.self_device_time_total/tot:5.1f}%  n={e.count:5d} avg={e.self_device_time_total/e.count:7.1f} us  {e.key[:80]}")
