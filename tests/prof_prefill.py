import os, sys, torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import Engine
B, S = map(int, sys.argv[1:3])
eng = Engine("/workspace/model")
g = torch.Generator().manual_seed(0)
ids = torch.randint(1000, 100000, (B, S), generator=g)
st = eng._state(B, S + 128)
for _ in range(2): eng._prefill(ids.cuda(), st)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as p:
    eng._prefill(ids.cuda(), st); torch.cuda.synchronize()
rows = sorted([e for e in p.key_averages(group_by_input_shape=True) if e.self_device_time_total > 0], key=lambda e: -e.self_device_time_total)
tot = sum(e.self_device_time_total for e in rows)
print(f"prefill device total {tot/1e3:.1f} ms")
for e in rows[:16]:
    print(f"{e.self_device_time_total/1e3:7.2f} ms n={e.count:3d} {e.key[:60]:60s} {str(e.input_shapes)[:90]}")
