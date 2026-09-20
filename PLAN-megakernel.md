# Plan: persistent/megakernel decode + runtime CUDA — target #1

Companions: `CONTEXT.md` (rules, every experiment + result),
`info.md` (task brief, protocol). Read those first; this is the
build plan only.

## 1. What first place costs

| | score | metricMs |
|---|---|---|
| SSS (#1) | 1144.3 | 442.6 |
| dryfter / Silver Bullet | 1138.7 / 1137.7 | 444.8 / 445.2 |
| **krish (#6)** | **1047.3** | **483.6** |

`score * metricMs = 506522` exactly, so score is exactly proportional to 1/aggregate private time.
**Need -8.5% aggregate time to take #1**, -9.6% for margin over the 1144/1138/1137 cluster.

Where that time is (weights from `tests/score_model.py`; prefill/decode split from B4 TTFT vs
32x TPOT and the three official draws of `1e460f5`):

- decode ~= 75% of weighted time, prefill ~= 25%.
- So: **-12% decode alone = -9% aggregate = #1.** -20% prefill = -5% aggregate (not enough alone).

Decode step (B16 512->128, measured 4192 us, `tests/budget.py`): 8.04 GB weights + 1.51 GB KV
= 9.55 GB -> 2851 us at 3.35 TB/s. We are at **68% of roofline**. Reaching 85% is -20% decode =
-15% aggregate = ~1230. That is the prize, and it is all in one place.

## 2. Root cause: wave quantization, not launch overhead

The handoff blamed a 4-5 us per-kernel "ramp/tail". It is sharper than that. Every decode GEMV
launches `(N/BN) * SK` CTAs onto 132 SMs, and none of the shapes divides 132:

| op | shape (N,K) | cfg (BN,SK) | CTAs | waves | SM util | predicted TB/s | measured |
|---|---|---|---|---|---|---|---|
| qkv | 6144, 2560 | 64, 1 | 96 | 1 | **72.7%** | 2.23 | ~2.2-2.4 |
| o | 2560, 4096 | 64, 2 | 80 | 1 | **60.6%** | 1.88 | ~1.8 |
| down | 2560, 9728 | 32, 4 | 320 | 3 | **80.8%** | 2.48 | ~2.5 |
| gate_up | 9728, 2560 (x2) | 32, - | 304 | 3 | **76.8%** | 2.36 | 2.76 |
| lm_head | 151936, 2560 | 64, 1 | 2374 | 18 | **99.9%** | 3.08 | **3.08** |

`predicted = util * 3.35 * 0.93`. It reproduces the measured 1.8-2.4 TB/s band on 4 of 5 ops and
nails lm_head to 3 digits. lm_head is the control: the *only* op with ~100% SM utilisation is the
*only* op at full bandwidth. Nothing about its inner loop is special.

**The whole 30% roofline gap is idle SMs.** Fix utilisation on the four small GEMVs and the weight
traffic (8.04 GB) runs at lm_head's 3.08 TB/s: 3088 us -> 2610 us, **-478 us = -11.4% of the step**,
matching the handoff's independently-derived 14.7% estimate for the same group. Attention (626 us at
~2 TB/s) has the same disease at B4 (32*nsplit CTAs) and is worth another ~150 us.

### Why 255 Triton config sweeps did not find this
Triton block sizes are powers of two. Reachable CTA counts for qkv are 96 (BN=64), 192 (BN=32),
384 (BN=16)... **132 is not reachable.** BN=32 gives 192 tiles over 132 SMs: 60 SMs do 2 tiles, 72 do
1, makespan = 2 units = exactly the same as 96 CTAs of double-size tiles. The sweep space was closed
under the defect. This is not "Triton headroom is exhausted" — it is "Triton cannot express the fix".

### Why the peer's persistent GEMV came out 5-10% slower (`tests/gemv_persist.py`)
Its sweep is `BN in (16, 32)` only — never 64, the size that actually won. It compared a balanced
grid of *worse* tiles against an unbalanced grid of *better* tiles, and it still reset and stored the
accumulator per row-tile inside the loop. **The balance hypothesis was never tested.** The fix needs
uneven row ranges (CTA c owns rows `c*N/132 .. (c+1)*N/132`, i.e. 46 or 47 rows for qkv), which that
kernel cannot produce.

## 3. Stages, each gated on a measurement

Ordered by (expected gain) / (risk x time). **Stop at the first stage that reaches #1 and hold it.**

### Stage A — balanced-grid GEMV in *Triton* (no rules risk, ~1.5 h)
Pure falsification of section 2 using tools we already trust.
Persistent kernel, `grid = (132,)` or `(264,)`, CTA c owns a contiguous **uneven** row range; tile
shape stays BN=64 but a row mask hides the 46-vs-47 raggedness (masked loads do not fetch, so HBM
traffic is exact; wasted tensor-core lanes are free in a memory-bound op). Accumulator stays in
registers for the CTA's whole range; one store at the end. Same treatment for `_gemv_silu_kernel`.
- Bench standalone first, `tests/gemv_bench.py` style, all four shapes at M=1/4/16.
- **Gate A: >=8% on qkv + o standalone.** If yes -> wire into `TritonOps.linear` behind
  `ENGINE_BAL=1`, pod A/B 5 samples, `--check-all` clean, ship. Expected +5-9% score on its own.
- If A wins but under-delivers vs the table, the residue is real per-kernel ramp -> Stage C is worth
  more, not less.

### Stage B — CUDA GEMV via NVRTC (~3 h, gated on A)
Everything A cannot express: exact row ranges with no masking waste, `mma.sync.m16n8k16` with the
K-permutation from the handoff (one 16 B W load + one 16 B x load per thread per 32-k block),
k-tiled x staging in smem (16 KB at B16, which keeps down-proj's K=9728 out of smem entirely),
hand-tuned cp.async pipelining, PDL prologue issuing the first stages before `griddepcontrol.wait`.
- **Gate B: >=10% over the Stage-A Triton kernel on the qkv shape** (the handoff's own bar).
- If A already reached ~3.08 TB/s there is nothing here — skip to C or stop.

### Stage C — decode megakernel (~5 h, the stretch)
One `cuLaunchKernel` per decode step; the CUDA graph becomes a single node.
- **Grid 132 CTAs, 1 per SM, occupancy 1** (verify with `cuOccupancyMaxActiveBlocksPerMultiprocessor`)
  so all CTAs are co-resident and spin-waits cannot deadlock. No cooperative launch — its
  graph-capture behaviour is unverified and we do not need it.
- **Instruction tape**: the op sequence is identical every step (shapes fixed per graph), so it is a
  static device array built at engine init. Ops: `RMSNORM`, `GEMV_QKV`, `QKNORM_ROPE_KVWRITE`,
  `ATTN`, `GEMV_O+add+rmsnorm`, `GEMV_GATEUP_SILU`, `GEMV_DOWN+add+rmsnorm`, `LM_HEAD`, `ARGMAX`.
  ~8 per layer x 36 + 2.
- **Sync**: per-op arrive/wait on a `sem[op]` counter (atomicAdd + spin on a generation word), not
  `grid.sync()`. The critical trick: **each CTA issues the cp.async loads for its first weight tiles
  of op i+1 *before* spinning on op i** — weights never depend on activations. Barrier latency hides
  inside HBM latency we pay anyway. This is PDL generalised to every op boundary, with the ramp cost
  gone because the CTAs are already resident and warm.
- **This is the mechanism that gets past 3.08 TB/s**: the pipeline never drains between ops.
- **Gate C0 (before any of this, 30 min):** measure one arrive/wait barrier at 132 CTAs. Budget is
  ~290 barriers/step; at 4192 us, 1% of the step = 145 ns/barrier unhidden. Measure raw cost *and*
  cost-with-prefetch-in-flight; only the latter matters. Also measure the pure streaming ceiling
  (persistent kernel reading all 8 GB) — that is the real target number, not 3.35.
- **Gate C1:** one fused layer beats the current per-op Triton chain for that layer (116 us at B16)
  by >=10% -> build the full step. Else keep A/B and stop.
- **Gate C2:** pod A/B >=1.5% geomean over the shipped build before it goes to main.

### Stage D — prefill (only if decode stalls, or after #1 is held)
25% of weighted time; cuBLAS runs prefill GEMM at 756 of 989 TFLOP/s and FA2 attention at ~280.
A hand-written wgmma+TMA GEMM or FA3-class attention under NVRTC is several GPU-hours each with a
real chance of zero. **Do not start D before C is decided.** Note the private set is prefill-heavy
(proved by the 779 draw), which is already priced into the 75/25 split above.

## 4. Correctness and safety (non-negotiable)

The engine has a bistable massive activation (|h|~5300 at layer ~16); *any* change in rounding order
can flip a token at B32/B64. Every stage:
1. Keep reference rounding points exactly: fp32 accumulate, bf16 store at the same places as
   `_gemv_kernel` / `_reduce_add_rms_kernel`. A changed split-K count changes the reduction order —
   treat that as a numerics change, not a perf knob.
2. Every new op goes into `Engine._selftest()` so a bad kernel degrades to slow-but-correct, never wrong.
3. `tests/bench.py --check-all` clean on the 3 public shapes **plus** the audit list
   `32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`, and `--corpus code|repeat`.
   `worst gap <= 2.0`, `positions > 2.0: 0`, spread <= 25%.
4. Every new path is env-gated (`ENGINE_BAL`, `ENGINE_MK`) and **defaults off until its pod A/B
   passes**, so a bad merge is a one-line revert.
5. Verify graph capture explicitly — a capture failure silently degrades to eager and only surfaces
   as a slow official run.

## 5. Rules / compliance

Stage A is plain Triton: zero risk, ship freely. Stages B and C ship runtime-compiled CUDA C++ as a
Python string. The challenge description's own words are the strongest evidence we have: *"Vendor any
Python or Triton source in the archive... do not include weights, credentials, compiled binaries or
Docker images"* — NVRTC ships no binary, and `budgets.max_compile_seconds = 600` shows load-time
compilation is expected. The starter guide's "Triton `@triton.jit` in .py files" is narrower.
`docs/QWEN_ENGINE_CONTRACT.md` is authoritative and is not reachable through the API.

**Action: one Slack message to the organizers now** — "is runtime-compiled CUDA C++ via NVRTC from a
source string in a .py file allowed?" It runs in parallel with Stage A and gates only the *push* of
B/C, not their development. **Do not run the canary** (`engine/cudart.py:canary`): it burns a queue
slot, leaves a deliberately slow run in history, and reads as probing the grader. Keep it default-off.

## 6. Operations

- Pod `6ewaafott6x3hh` (H100 SXM 80GB, EUR-IS-3, **$3.49/h**, currently EXITED). Start via runpod MCP
  `pod-action start`; the SSH port changes every start (`get-pod` -> `runtime.ports` 22), boot ~90 s.
  `/workspace` survives: `/workspace/venv/bin/python` (torch 2.5.1 / triton 3.1.0 / transformers
  4.51.3), `/workspace/model`, `export TRITON_CACHE_DIR=/workspace/triton_cache`, work dir
  `/workspace/w`. **`flock /workspace/gpu.lock` around every GPU command** (peer protocol).
  **Stop the pod whenever idle** (standing instruction).
- Estimated cost: A 1.5 h, B 3 h, C 5.5 h, +1 h integration/bench = **~11 GPU-h ~= $38**, less if we
  stop at a gate.
- Pushes: `git fetch && git merge origin/main` **before every push** (one push was rejected for this);
  `juancavallin` pushes to main under his own Claude and every push starts a ~7-10 min official run on
  a serial per-team queue. Message him before pushing.
- Reruns (`POST /submissions/<id>/runs {"mode":"official"}`) are free draws and only the best eligible
  run counts, but the official score has a heavy left tail from platform prefill contention.
  **Judge every change by pod A/B (5 samples), never by one official run.** Ask before a rerun campaign.

## 7. Schedule (from a cold start)

| t | work | decision |
|---|---|---|
| 0:00 | Slack organizers re NVRTC; start pod | — |
| 0:10 | Stage A kernel + standalone bench | Gate A (>=8%) |
| 1:30 | Stage A in-engine A/B + `--check-all` | ship A to main if >=1.5% |
| 2:00 | Gate C0 barrier + streaming-ceiling probes | go/no-go on C |
| 2:30 | Stage B CUDA GEMV (if A left >3% on the table) | Gate B (>=10%) |
| 5:30 | Stage C one-layer megakernel | Gate C1 (>=10%) |
| 10:00 | Stage C full step + correctness + A/B | Gate C2 (>=1.5%) |
| 11:00 | push, rerun draws, stop pod | — |

Expected outcome: Stage A alone ~1100-1140 (#2-#3). A+B ~1130-1170. A+B+C ~1180-1230 (**#1 with
margin**). If Gate A fails, section 2 is wrong, the honest remaining headroom is ~0-1%, and the right
move is to stop and let the best run stand.

## 8. Unresolved questions

1. Submission deadline? (drives how many stages we attempt)
2. GPU budget ceiling? ($38 planned; ~$15 already spent today)
3. Slack to organizers re NVRTC — you send it, or is there a channel I can reach?
4. Push protocol with `juancavallin` — you coordinate, or I push and tell you?
5. Ship Stage A to main on a pod-A/B win alone, or approve each push?
