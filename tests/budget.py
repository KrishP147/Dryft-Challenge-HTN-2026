"""Per-kernel time budget of one decode step, and how much of the step is launch gap.

Replays the captured decode graph on its own (no prefill, no host sync) so the numbers
are exactly what a decode step costs, then buckets kernels into the groups we can act on
and prints each as us/step and % of the measured step.

usage: MODEL=/workspace/model python tests/budget.py [B,S,n ...]
"""
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from engine import CAP_GRAN, Engine  # noqa: E402

SHAPES = sys.argv[1:] or ["1,512,32", "4,2048,32", "16,512,128"]
REPLAYS = 50

GROUPS = (  # (label, substrings) - first match wins
    ("gemv (qkv/o/down/lm_head)", ("_gemv_kernel",)),
    ("gemv+silu (gate_up)", ("_gemv_silu_kernel",)),
    ("qkv_post (norm+rope+kv)", ("_qkv_post_kernel",)),
    ("attn split", ("_attn_split_kernel",)),
    ("attn combine", ("_attn_combine_kernel",)),
    ("reduce+add+rms", ("_reduce_add_rms_kernel",)),
    ("add+rms", ("_add_rms_kernel",)),
    ("split-k reduce", ("_splitk_reduce_kernel",)),
    ("silu_mul", ("_silu_mul_kernel",)),
    ("cuBLAS / torch", ("gemm", "cutlass", "sm90", "nvjet", "elementwise", "index", "Kernel")),
)

eng = Engine(os.environ.get("MODEL", "/workspace/model"))
g = torch.Generator().manual_seed(0)

for spec in SHAPES:
    B, S, n = map(int, spec.split(","))
    ids = torch.randint(1000, 100000, (B, S), generator=g).tolist()
    list(eng.generate(ids, n))  # warm + capture
    cap = -(-(S + n) // CAP_GRAN) * CAP_GRAN
    st = eng.states[(B, cap, 1)]
    if st.graph is None:
        print(f"B{B} {S}->{n}: no decode graph, skipping")
        continue

    for _ in range(5):
        st.graph.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPLAYS):
        st.graph.replay()
    e.record()
    torch.cuda.synchronize()
    step_us = s.elapsed_time(e) / REPLAYS * 1e3

    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(REPLAYS):
            st.graph.replay()
        torch.cuda.synchronize()
    rows = [x for x in p.key_averages() if x.self_device_time_total > 0]

    buckets, counts, seen = {}, {}, set()
    for x in rows:
        label = next((lab for lab, keys in GROUPS if any(k in x.key for k in keys)), None)
        if label is None:
            label = f"? {x.key[:40]}"
        buckets[label] = buckets.get(label, 0.0) + x.self_device_time_total / REPLAYS
        counts[label] = counts.get(label, 0) + x.count / REPLAYS
        seen.add(x.key)
    busy = sum(buckets.values())

    kv_gb = 2 * eng.L * B * eng.nkv * (S + n // 2) * eng.hd * 2 / 1e9
    w_gb = (sum(l.wqkv.numel() + l.wo.numel() + l.wgu.numel() + l.wd.numel() for l in eng.layers)
            + eng.lm_head.numel()) * 2 / 1e9
    roof_us = (w_gb + kv_gb) / 3.35 * 1e3

    print(f"\nB{B} {S}->{n}: step {step_us:7.1f} us  kernels busy {busy:7.1f} us "
          f"({100*busy/step_us:4.1f}%)  gap {step_us-busy:6.1f} us")
    print(f"   roofline {roof_us:6.1f} us ({w_gb:.2f} GB weights + {kv_gb:.2f} GB kv) "
          f"-> {100*roof_us/step_us:4.1f}% of step")
    for label, us in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"   {us:8.1f} us {100*us/step_us:5.1f}%  n={counts[label]:6.1f}  {label}")
