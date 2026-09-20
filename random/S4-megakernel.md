# S4 — `megakernel`: probes, then a persistent whole-step decode kernel

You are a worker agent starting with zero context. Everything you need is here or in the repo files
named below. **Read `handoffs/SHARED-CONTEXT.md` first, then `CONTEXT.md`, before writing code.** They carry the
scoreboard, rules stance, protocol, correctness bar and profiling traps. Do not repeat anything in
`SHARED-CONTEXT.md` > "Dead levers".

Owner: Krish. Orchestrator: a separate Opus session that merges and pushes. You never push to `main`.

**You own the highest-ceiling and highest-risk workstream. It is the only path past ~1150. It is
expected to be the least likely to land. Both of those are fine — work the gates honestly and report
a kill as fast as you would report a win.**

---

## 1. Mission

Dryft "decode": Qwen3-4B decode on 1x H100, output byte-identical to native greedy.
Segfault leads at **1176.4** (metricMs 430.6); we are #6 at **1047.3** (483.6).
`score * metricMs = 506522` exactly, so score is exactly proportional to 1/aggregate time: **-11%
aggregate** for #1, **-15%** to hold it. Decode is ~75% of weighted time.
Deadline **Sun 08:00 ET**, feature freeze 05:00 ET.
Regime weights: B1 **0.14**, B4 2048->32 **0.45**, B16 512->128 **0.44**. Never optimise for B1.

## 2. Rules stance — settled, do not relitigate

**If the platform accepts the submission, it is allowed.** The owner decided this explicitly. Any older note saying "no NVRTC/CUDA-string code until the organizers confirm" is **superseded**. Agent S3 is pushing a small NVRTC-backed engine change to `main` in the first hour to
confirm the platform accepts it; the orchestrator will relay the result. **You do not wait for it** —
build and measure on the pod regardless; it only gates the final push.
**Do NOT run `canary()`** in `experimental/nvrtc/cudart.py`; it is obsolete.

## 3. Where the time is

Decode step B16 512->128, measured **4192 us**, against 8.04 GB of weights + 1.51 GB of KV = 9.55 GB
-> **2851 us** at 3.35 TB/s. We run at **68% of roofline**. Per-kernel budget (`tests/budget.py`,
PDL off): small GEMVs qkv/o/down/lm_head 1787 us (42.6%), gate_up+silu 1301 us (31%), attention split
626 us (14.9%), reduce+add+rms 215 us (5.1%, 72 launches), qkv_post 84 us, combine 61 us.
**Launch gaps total only 2.2%** — so "fewer launches" is not, by itself, the win.

The win is that **the memory pipeline drains at every kernel boundary**. Utilisation is the proximate
cause (see the table below), and agents S1/S2/S3 are attacking that directly with balanced grids.
Your kernel is the only design that also removes the drain: a resident CTA can issue the loads for
op *i+1* while it is still finishing op *i*.

| op | shape (N,K) | CTAs | SM util | pred TB/s | measured |
|---|---|---|---|---|---|
| qkv | 6144, 2560 | 96 | 72.7% | 2.23 | ~2.2-2.4 |
| o | 2560, 4096 | 80 | 60.6% | 1.88 | ~1.8 |
| down | 2560, 9728 | 320 | 80.8% | 2.48 | ~2.5 |
| gate_up | 9728, 2560 | 304 | 76.8% | 2.36 | 2.76 |
| **lm_head** | 151936, 2560 | 2374 | **99.9%** | 3.08 | **3.08** |

**3.08 TB/s is what a perfectly-occupied kernel achieves today. Your target is to beat it**, because
you never drain.

## 4. Gate C0 — probes FIRST. Budget 45 minutes. Report before building anything.

Use `experimental/nvrtc/cudart.py` (`compile_cubin`, `Module`, `Kernel.__call__`) — already verified
on an H100 with pinned torch 2.5.1: compile 0.03 s, eager launch correct, launch inside a CUDA graph
replays correctly. Smoke test: `experimental/nvrtc/test_cudart.py`.

Measure, and report all four numbers:

1. **Arrive/wait barrier cost at 132 CTAs.** `atomicAdd` on a counter + spin on a generation word.
   Loop 1000 barriers, report ns/barrier. Budget: ~290 barriers/step (8 ops x 36 layers); at 4192 us,
   **1% of the step = 145 ns/barrier**. Measure it **twice**: raw, and with cp.async weight loads
   already in flight across the barrier. **Only the second number matters** — see §5.
2. **Persistent streaming ceiling.** One resident kernel, 132 CTAs, reading all 8.04 GB of weights in
   the real access pattern. This, not 3.35, is your actual target number.
3. **Occupancy.** `cuOccupancyMaxActiveBlocksPerMultiprocessor` for your block shape and smem budget.
   You need **exactly 1 CTA/SM co-resident** for spin-waits to be deadlock-free. Confirm it; do not
   assume it.
4. **Graph capture of a long-running kernel.** Confirm a single `cuLaunchKernel` of a multi-ms kernel
   captures and replays. (Plain `cuLaunchKernel` capture is already verified; a *cooperative* launch
   is not — **don't use cooperative launch**, you don't need it and its capture behaviour is unknown.)

**Kill rule:** if (1)-with-prefetch is worse than ~150 ns and (2) is not above ~3.0 TB/s, say so
immediately and pivot to helping S3 on the CUDA GEMV. A bad C0 is a good outcome delivered early.

## 5. Design (the mechanism that makes it win)

One `cuLaunchKernel` per decode step; the CUDA graph becomes a single node.

- **Grid 132 CTAs, occupancy 1/SM**, all co-resident, so spin-waits cannot deadlock.
- **Instruction tape.** The op sequence is identical on every step (shapes are fixed per graph), so
  build it once at engine init as a static device array. Ops: `RMSNORM`, `GEMV_QKV`,
  `QKNORM_ROPE_KVWRITE`, `ATTN`, `GEMV_O+add+rmsnorm`, `GEMV_GATEUP_SILU`, `GEMV_DOWN+add+rmsnorm`,
  `LM_HEAD`, `ARGMAX`. ~8 per layer x 36 + 2.
- **Sync: per-op arrive/wait on `sem[op]`, not a grid barrier.** Each CTA `atomicAdd`s on finishing op
  `i` and spins on the generation word until all have arrived.
- **THE TRICK — this is the whole design.** Each CTA issues the cp.async loads for its first weight
  tiles of op `i+1` **before** it spins on op `i`. Weights never depend on activations. The barrier
  latency then hides inside HBM latency we were going to pay anyway, and the CTAs are already resident
  and warm, so there is no ramp. This is the existing PDL trick (`griddepcontrol.wait`, worth +4.5%
  today) generalised to **every** op boundary with the ramp cost removed.
- **Per-op work split:** CTA `c` owns the uneven contiguous row range `c*N/132 .. (c+1)*N/132`.
  No split-K, so `_splitk_reduce_kernel` disappears and `_reduce_add_rms_kernel` shrinks.
- **Stage `x` in smem per k-tile, never whole.** For down-proj (K=9728, M<=16), re-reading `x` from
  L2 per output row would cost ~780 MB of L2 traffic per op. Tile over K (e.g. BK=512 -> 16 KB of x),
  hold the `rows x M` fp32 accumulators in registers across the K loop, store once.
- `mma.sync.m16n8k16` bf16->fp32, with the K-permutation `k = 8q + 4s + 2h + e` (lane-group `q`, step
  `s`, register `h`, element `e`) so each thread takes one 16 B W load and one 16 B x load per 32-k
  block, with `x` permuted identically at staging time.
- Attention inside the tape: at B16 there are `B*nkv = 128` (seq, kv-head) pairs, which maps cleanly
  onto 132 CTAs. At B4 it is 32 pairs — you will need a split and a combine step inside the tape.

## 6. Gates after C0

- **C1 (the real go/no-go):** one fused layer beats the current per-op Triton chain for that layer
  (**116 us at B16**) by **>= 10%**. If it doesn't, stop and hand your probe numbers to the
  orchestrator — that is a valuable result, not a failure.
- **C2:** full step, in-engine behind `ENGINE_MK=1` (default **off**), pod A/B 5 samples, bar
  **>= +1.5% predicted score geomean**. Isolated microbenches have pointed the wrong way twice on this
  repo — **the in-engine A/B is the verdict.**
- **C3:** correctness, §7. No exceptions, even under deadline pressure.

## 7. Correctness — non-negotiable

Bistable massive activation (|h| ~ 5300 at layer ~16): *any* change in bf16 rounding points or
reduction order flips tokens at B32/B64. Keep fp32 accumulate and bf16 stores at exactly the points
the current Triton kernels use (`_gemv_kernel`, `_gemv_silu_kernel`, `_reduce_add_rms_kernel` in
`engine/fused.py` are the reference). Add the megakernel to `Engine._selftest()` so a bad kernel
degrades to slow-but-correct, never wrong. Required: `tests/bench.py --check-all` clean on the 3
public shapes **and** `--shapes 32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`,
plus `--corpus code` and `--corpus repeat`; `worst gap <= 2.0`, `positions > 2.0: 0`,
`spread <= 25%`. Pre-existing violations you must not worsen: B32 code corpus 3.9 logits (seq 24
step 85), B64 natural 3.25 (seq 0 step 99).

Also: any tensor read inside the CUDA graph must be a persistent buffer updated in place (`st.tok`,
`st.pos`); host-side Python values are baked in at capture. KV is init'd with `zeros`, not `empty`
(masked slots must be finite: `0 * NaN = NaN`). The platform runs one shape per fresh process and
capture happens during the untimed warmup generate.

## 8. Ownership — strict

Yours: `experimental/mk/` while prototyping, then a **new** `engine/mk.py`, plus one env-gated branch
in `engine/engine.py`'s decode body. **Not yours:** `_gemv_kernel`/`GEMM_CFG`/`TritonOps.linear`
(agent S1), `_gemv_silu_kernel`/attention (agent S2), `engine/cudart.py` (agent S3 — **consume it,
don't edit it**; coordinate through the orchestrator if you need a change), prefill (agent S5).

## 9. Pod setup

Create your own H100 pod (no budget ceiling; ~$3.49/h; **stop it when you hand in**). runpod MCP:
`get-capacity` / `create-pod`, GPU `NVIDIA H100 80GB HBM3`, secure cloud, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 60 GB disk, 40+ GB `/workspace`.
SSH via the **direct** `ssh root@IP -p PORT` from `get-pod` > `runtime.ports` 22 (the `ssh.runpod.io`
proxy needs a PTY). Key `~/.ssh/runpod_ed25519`. Boot ~90 s. The pinned stack matters — NVRTC is found
through torch's bundled `nvidia/cuda_nvrtc/lib/libnvrtc.so*`:
```bash
pip install torch==2.5.1 triton==3.1.0 transformers==4.51.3 safetensors==0.5.3 tokenizers==0.21.1 huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir /workspace/model
export TRITON_CACHE_DIR=/workspace/triton_cache
```
Traps: with PDL on a waiting kernel is charged its producer's time (`budget.py` sums read 124-131% of
the step) — attribute with `ENGINE_PDL=0`. `budget.py` replays `pos += 1`, so rewind `st.pos` or
attention reads past `cap` (illegal access). Graph re-capture needs a fresh
`torch.cuda.graph_pool_handle()`.

## 10. Hand-in

Branch `krish/megakernel` on your pod. Never push to `main`; never edit the Windows checkout (another
session is live in it). Push the branch to `origin` if your pod has GitHub creds, else
`git diff main > /workspace/w/megakernel.patch` and paste the diff.

Report **three times**: (a) **Gate C0 at T+45 min — all four probe numbers, unconditionally**;
(b) Gate C1 verdict with the per-layer timing; (c) final: branch + commit, in-engine pod A/B table,
`--check-all` lines, verdict, and every negative result with numbers. Negative results here are
genuinely valuable — they close out the last open question in `CONTEXT.md`.
