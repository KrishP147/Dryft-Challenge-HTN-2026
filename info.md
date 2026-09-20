# info: work brief for Juan (worker-agent mode)

You = worker. Krish = owner/decider. Read `CONTEXT.md` first (rules, design, every result). This file = what to run/try/implement next. Don't redo anything in "Dead levers".

## State (Sep 20)
- Best official **1047.3** (#6/43). Leaders: 1144 / 1139 / 1138. Plateau: last 3 pushes moved score ~0.
- `main` engine == build `1e460f5` (draws 1041.1 / 1047.3 / 779.4). Decode TPOT identical to 0.01 ms across all draws; the variance is a **prefill/TTFT tail** (platform contention). Private set is prefill-heavy.
- Score: `score * metricMs / 1000 = 506.52223` (exact). Regime weights on public tok/s: B1 0.14, **B4 2048->32 0.45**, **B16 512->128 0.44**. Never tune B1. Private set likely has no B>=32, none with B*S<=4096.
- Decode step (B16, PDL off, 4192 us): small GEMVs 42.6%, gate_up+silu 31%, attn split 14.9%, reduce+add+rms 5.1%, qkv_post 2%, combine 1.5%. Launch gaps only 2.2%. Gap to roofline ~30%, ~15% of it = small GEMVs at 1.8-2.4 TB/s vs lm_head 3.08 TB/s (per-launch ramp).

## Protocol (hard rules)
1. Branch `juan/<name>` (never commit experiments to `main`).
2. **Push to `main` = official run** (serial queue, ~7-10 min, counts). Push only when: pod A/B (5 samples) shows >= +1.5% predicted score AND `python tests/bench.py --check-all` clean (worst gap <= 2.0, 0 positions > 2.0, spread <= 25%). Before push: `git fetch && git merge origin/main`.
3. Decide with pod A/B in the **real engine** (PDL on), not microbench. Isolated microbenches pointed wrong twice. Official runs can't resolve <1% and have a left tail; never trust one.
4. Don't add numeric drift (bf16 rounding points, reduction order). Massive-activation bistability: any drift flips it at large B. Selftest/fallback in `Engine._selftest` must cover any new fused op.
5. `engine/` = only thing submitted. Triton/Python source only, no binaries/network. **No NVRTC/CUDA-string code in `engine/`** until Krish confirms organizers allow it (task 0). Prototype in `experimental/`.
6. Nothing secret in git. Team token lives in `~/.dryft_token`.
7. Report each task: branch, commit, pod A/B table (before/after, 5 samples, spread), check-all result, verdict (keep/kill). Negative results get appended to CONTEXT.md "Findings" (short, with numbers).

## Setup
RunPod H100 SXM (~$3.5/h, stop when idle). Recipe in `CONTEXT.md` > Local dev. Krish's pod `6ewaafott6x3hh` is stopped; ask before starting it. Use `flock /workspace/gpu.lock` on shared pod. `export TRITON_CACHE_DIR=/workspace/triton_cache`.

## Tasks (priority order)

### T0. Rules gate (Krish does; you're blocked on CUDA tasks until answered)
Question for organizers: "is runtime-compiled CUDA C++ via NVRTC in a .py string allowed?". Starter guide says Triton/Python source only, no cubin/PTX/.so. If **no**: skip T3/T4, Triton headroom ~0-1%, stop after T1/T2.

### T1. Prefill speed (highest expected value, Triton-only, no rules risk)
Why: prefill ~48% of B4 regime (TTFT ~112 ms @ B4x2048; attention ~19.6 ms, GEMM ~740 TFLOP/s vs cuBLAS 756 peak-ish). Also shrinks exposure to the TTFT tail.
Run:
```bash
python tests/prof.py 4 2048 32 --phase prefill     # kernel table
python tests/bench.py --shapes 4,2048,32 --no-check
python tests/prof_prefill.py ; python tests/flash_prefill.py
```
Try, in order:
1. Prefill attention: tune Triton flash kernel (BLOCK_M/N, num_warps, num_stages, causal block-skip, exp2 softmax, q-token-major layout) in `engine/fused.py`. Target: attention 19.6 ms -> <= 15 ms. Sweep in-engine.
2. Non-GEMM prefill overhead (~19% cut already by token-major qkv): re-profile; look for remaining elementwise/copy kernels (add+rmsnorm at M=8192, silu*up, rope/cache write). Anything not at ~3 TB/s is a target.
3. Mid sizes (B16 512 = M 8192 too): check GEMM shapes pick best cuBLAS algo; try `torch.backends.cuda.matmul` knobs only if output stays exact.
Accept: B4 TTFT -5% or better in pod A/B, check-all clean.

### T2. Small-GEMV ramp (decode, Triton-only)
Goal: close part of the 14.7% small-GEMV gap. Ideas NOT yet tried (everything in Dead levers is excluded):
- Merge o-proj GEMV with the following reduce+add+rms so o (1761 GB/s, worst) loses one launch/ramp (careful: split-K reduce needs all CTAs; measure with `tests/budget.py`).
- Batch independent GEMVs into one launch (e.g. q/k/v already fused; try o+nothing? check for any two GEMVs with no dependency between them per layer; likely none, then skip).
- Split-K factor per shape to raise CTA count for o/down/qkv (grid 80-320 CTAs on 132 SMs; try nsplit that gives 2-4 waves).
Run: `ENGINE_PDL=0 python tests/budget.py 16,512,128`; `python tests/gemv_cfg_sweep.py {qkv,o,down,gate_up,pf}`.
Accept: >= +1.5% pod geomean.

### T3. Grid-barrier go/no-go probe (pod only, NO push; needs T0 = yes)
Measure cost of ONE grid barrier and whether a cooperative launch is capturable in a CUDA graph. Use `experimental/nvrtc/cudart.py` (`compile_cubin`, `Module`, `Kernel.__call__`; smoke test `experimental/nvrtc/test_cudart.py`).
- Kernel: 132 CTAs x N threads, loop 1000 barriers (atomic counter spin, and `cuLaunchCooperativeKernel` variant), report us/barrier.
- Go rule: <= ~1.5 us/barrier (~288 barriers/step). Otherwise megakernel can't win (persistent GEMV was -4%, atomic combine -2%). Write result to CONTEXT.md.

### T4. CUDA GEMV prototype (only if T0=yes and T3=go, or as stand-alone)
mma.sync m16n8k16 bf16 GEMV with PDL prologue issuing first W stages (cp.async) before `griddepcontrol.wait`. Design notes: K-permutation so each thread takes one 16 B W load + one 16 B x load per 32-k block; stage x in smem once; CTAs ~64 rows, 8 warps; keep split-K partials + `_reduce_add_rms_kernel`; fp32 accumulate, bf16 outputs; hook into selftest.
Falsify cheap first: 1-kernel prototype must beat Triton qkv GEMV (13.4 us) by >= 10% in `tests/gemv_bench.py`-style timing, else kill. Expect <0.5% (PF result says early weight start is exhausted).

### T5. Large-batch correctness risk (low priority, only if time)
B32 code corpus: 3.9-logit violation (seq 24 step 85); B64 natural: 3.25 (seq 0 step 99). Identical on v14/v15, native control clean (`tests/native_gap.py` gives 1.0). Needs BOTH fused attn and fused qkv (`ENGINE_OFF=attn` or `=qkv` pass). Bisect further with `tests/trace_diff2.py`, `tests/trace_prefill.py`. Only matters if a private workload has B>=32. Fix must not slow B<=16.

## Dead levers (measured, don't repeat)
PF/L2 prefetch depth (PF=4 best, 32 = -14%) | GEMV tile/stage/warp configs (255 swept, +0.0-0.1%) | mask-free GEMV (+0.7%) | FMA GEMV / FMA attention | GEMV re-tiling | persistent SM-balanced GEMV (-4 to -10%) | fused last-block attn combine (-2%) | qkv_post into GEMV epilogue | Triton fused SiLU prefill GEMM (-10%) | cuDNN SDPA prefill | n-gram spec decode (net loss on platform prompts; `ENGINE_SPEC` off) | RMSNorm-in-GEMV-prologue (slower) | nsplit==1 combine skip (0 on private set) | CUDA-graph small prefills (flat) | reruns as "improvement" (lottery, Krish decides).

## Profiling traps
- PDL on: waiting kernel gets charged for producer's time (sum reads 124-131% of step). Attribute with `ENGINE_PDL=0`.
- Graph re-capture needs fresh `torch.cuda.graph_pool_handle()`.
- `budget.py` replays `pos += 1`; rewind `st.pos` or attention reads past cap (illegal access).
- KV init `zeros` not `empty`. Tensors read in-graph must be persistent buffers.

## Unresolved (Krish)
- Organizers' answer on NVRTC?
- GPU budget for T3/T4?
- OK to spend official runs on reruns?
