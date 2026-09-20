# SHARED-CONTEXT — read this before writing any code

Every worker agent (S1-S6) reads this file plus `CONTEXT.md` plus their own `handoffs/S*.md`.
This file holds what is true for everyone: the scoreboard, the rules stance, the protocol, what is
already dead, and the traps that have cost this team hours.

## Scoreboard (Sun 00:37 UTC) — 56 ranked teams

| rank | team | score | metricMs | achieved |
|---|---|---|---|---|
| 1 | **Segfault** | **1198.9** | **422.5** | 00:18 |
| 2 | SSS | 1144.3 | 442.6 | 21:30 |
| 3 | 0xDeadBeaf | 1142.9 | 443.2 | 00:35 |
| 4 | dryfter | 1138.7 | 444.8 | 23:35 |
| 5 | Silver Bullet | 1137.7 | 445.2 | 20:52 |
| 6 | zip | 1071.1 | 472.9 | 23:00 |
| **7** | **krish (us)** | **1055.9** | **479.7** | 00:51 |

`score * metricMs = 506522` exactly (15 digits, every run). Score is **exactly proportional to
1/aggregate private time**. Decode is ~75% of weighted time, prefill ~25%.

**Segfault is moving fast: 1064.4 -> 1176.4 -> 1198.9 in about 1.5 hours.** Extrapolating that rate to
the 08:00 ET deadline, **plan for a finish line around 1300**, not 1200. That is
`metricMs ~= 390`, i.e. **-19% aggregate from where we are**. (Our 1055.9 came from a *rerun draw* of the identical engine that previously scored 1047.3 — free, and worth +0.8%.)

## How the leader is doing it (and why it is not magic)

Work it out from their own number. `422.5 / 483.6 = 0.874` aggregate time. Prefill is ~25% of weighted
time; if theirs is unchanged, their decode step is `(0.874 - 0.25) / 0.75 = 0.832` of ours =
**~3486 us**, against the 2851 us bandwidth roofline. That is **~82% of roofline**. If they also cut
prefill, their decode is *less* impressive, around 77%.

**We are at 68%. Our own `lm_head` kernel already runs at 92%** (3.08 of 3.35 TB/s) in the build that
scored 1047. So the leader has no exotic technique — no quantization, no draft model, nothing we are
forbidden. They have kernels that do not waste SMs. **That is exactly what the wave-quantization
thesis below predicts, and exactly what S1-S4 are built to fix.** The gap is closable with the
hardware we have and the rules we must follow.

**The landing estimates that used to be here have been withdrawn.** They assumed the balanced-grid
fix would reach 3.08 TB/s across the GEMVs; S1 falsified that (see "BALANCED-GRID GEMV IS DEAD" below).
The honest position as of 01:00 UTC: the ~13% decode prize is real but only the megakernel (S4) is
still aimed at it, prefill (S5) is untouched by the falsification, and the glue kernels (S6) are
unaffected. Do not quote a projected final score; measure instead.

**No single agent's win is enough.** S1+S2 alone lands roughly where the leader already is. We need
three or four of the six to land, and S4 is what creates margin rather than a tie. Everyone finishes
their own task; nobody stops early because "the thesis is proven".

Regime weights on public tok/s, fitted to all 11 runs within 0.8%:
`log(score) = -0.266 + 0.142*log(B1 512->32) + 0.449*log(B4 2048->32) + 0.444*log(B16 512->128)`.
**1% on B1 is worth 0.14% of score. Never optimise for B1.**

**Deadline: Sun 08:00 ET. Feature freeze 05:00 ET.** After the freeze the orchestrator only
integrates, pushes and takes rerun draws.

## Rules stance — settled by the owner, do not relitigate

**If the platform accepts the submission, it is allowed.** No Slack question is pending; nobody is
waiting on an organizer. Any older note saying "no NVRTC/CUDA-string code until the organizers
confirm" is **superseded**.

Supporting evidence: the challenge description says *"Vendor any Python or Triton source in the
archive... do not include weights, credentials, compiled binaries or Docker images"*. NVRTC ships no
binary (a CUDA source string inside a `.py`, compiled at load), and `budgets.max_compile_seconds =
600` shows load-time compilation is expected.

Agent S3 confirms platform acceptance in the first hour by pushing a small, real, NVRTC-backed engine
change. **Do NOT run `canary()`** in `experimental/nvrtc/cudart.py` — it encodes a probe result as a
per-step sleep, it burns a queue slot, and S3's probe answers the same question honestly.

## Hard protocol

1. **Work on your own pod, on branch `krish/<your-name>`.** Never push to `main` — the orchestrator
   does that. Never edit the Windows checkout at `C:\Users\User\iloveayush`; another session is live
   in it and you will collide.
2. **Push to `main` = an official run** (serial per-team queue, ~7-10 min queue + ~13 min run).
3. **File-region ownership is strict.** Your handoff lists what is yours and what is not. If a change
   would touch someone else's region, route it through the orchestrator instead of editing it.
4. **The in-engine pod A/B is the verdict, never the microbench.** Isolated microbenches have pointed
   the wrong way twice on this repo. Bar to ship: **5 samples, >= +1.5% predicted score geomean**,
   with PDL on, in the real engine.
5. **Never trust one official run.** The official score has a heavy left tail from platform prefill
   contention (see below) and cannot resolve <1%.
6. **Every new path is env-gated and defaults off** until its A/B passes, so a bad merge is a one-line
   revert.
7. Nothing secret in git. The team token lives in `~/.dryft_token`.
8. Report every result, including negatives, **with numbers**. The orchestrator appends them to
   `CONTEXT.md` > Findings so nobody repeats them.

## Getting the code onto your pod (no GitHub credentials needed)

Build a tarball from `origin/main` on the orchestrator's machine and scp it — do **not** clone, and do
**not** copy the live working tree (it is on a different branch mid-edit):

```bash
cd /c/Users/User/iloveayush && git archive origin/main -o "$SCRATCH/repo.tar"   # ~287 KB
scp -i ~/.ssh/runpod_ed25519 -P <PORT> "$SCRATCH/repo.tar" root@<IP>:/workspace/
# on the pod:
mkdir -p /workspace/w && tar -xf /workspace/repo.tar -C /workspace/w && cd /workspace/w
git init -q && git add -A && git commit -qm base && git checkout -qb krish/<your-name>
```
Hand work back as a diff: `git diff base > /workspace/w/<your-name>.patch`, and paste the diff in your
report. If your pod happens to have working GitHub credentials, pushing the branch to `origin` is
fine too — but the diff is the contract.

## Pod setup

**Capacity is tight — read this before you provision.** Checked at 00:40 UTC:
`NVIDIA H100 80GB HBM3` is **Low** stock (secure $3.49/h, community $2.69/h) and **only on host CUDA
12.8** — every 12.4 host is Out. So do **not** filter for a 12.4 host; the container image is CUDA
12.4 and runs fine on a 12.8 host (the existing pod does exactly this).

**Do NOT substitute another GPU.** H200 SXM (Medium stock) has 4.8 TB/s and H100 NVL has 3.9 TB/s.
All of them have 132 SMs, so SM-occupancy conclusions would transfer, but **every bandwidth number in
this plan is against the H100 SXM's 3.35 TB/s** — tuning tile depth or CTA counts against a different
roofline would silently mistune the engine for the grader. H100 SXM or nothing.

**Claim order, because there may not be five:** S4, S1, S2 get dedicated pods first (they iterate
hardest on H100-specific perf), then S3, then S5. Try `create-pod`; if capacity refuses, **fall back
to the shared pod `6ewaafott6x3hh`** — it is already **RUNNING**, H100 SXM, EUR-IS-3, with the venv,
the model and a clone at `/workspace/w` — and wrap every GPU command in
`flock /workspace/gpu.lock -c '<cmd>'`. Tell the orchestrator immediately if you land on the shared
pod, so it can re-sequence who gets the GPU when.

Create your own H100 pod if you can. The owner set **no budget ceiling**; ~$3.49/h. **Stop your pod
when you hand in** (never stop or terminate the shared pod `6ewaafott6x3hh` — the orchestrator owns
it). runpod MCP: `get-capacity` / `create-pod`, GPU `NVIDIA H100 80GB HBM3`, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 60 GB disk, 40+ GB `/workspace`.
SSH with the **direct** `ssh root@IP -p PORT` from `get-pod` > `runtime.ports` 22 — the
`ssh.runpod.io` proxy needs a PTY and will not work. Key: `~/.ssh/runpod_ed25519` (passphrase-less).
Boot ~90 s. A stop/start resets the container but not `/workspace`.

The pinned stack matters (NVRTC is found through torch's bundled `nvidia/cuda_nvrtc/lib/libnvrtc.so*`):
```bash
# huggingface_hub MUST be pinned: unpinned pulls 1.32.0, which breaks the transformers==4.51.3
# import outright. And `huggingface-cli` is deprecated -- use `hf download`. (Verified by S2
# on a fresh pod, 2026-09-20.)
pip install torch==2.5.1 triton==3.1.0 transformers==4.51.3 safetensors==0.5.3 tokenizers==0.21.1 "huggingface_hub>=0.30.0,<1.0"
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir /workspace/model
export TRITON_CACHE_DIR=/workspace/triton_cache
```

Krish's original pod `6ewaafott6x3hh` already has a venv, the model and a clone at `/workspace/w`.
It is **shared** — if you use it, wrap every GPU command in `flock /workspace/gpu.lock`. Prefer your
own pod.

## Bench and profile commands

```bash
python tests/bench.py --model /workspace/model            # perf + correctness, 3 public shapes, 5 samples
python tests/bench.py --shapes 16,512,128 --no-check      # quick perf only
python tests/bench.py --check-all                         # teacher-forces every sample vs HF baseline
ENGINE_PDL=0 python tests/budget.py 16,512,128            # per-kernel decode-step budget
python tests/prof.py 4 2048 32 --phase prefill            # kernel table (MODEL env = model path)
python tests/gemv_bench.py                                # skinny-GEMM microbench
python tests/gemv_cfg_sweep.py qkv                        # in-engine GEMV config A/B with PDL
python tests/spread.py                                    # 5-sample spread
```
`bench.py` prints a predicted official score and % of bandwidth roofline. **Pod numbers run ~0.7-1.4%
above the platform's** — that offset is constant, so A/B deltas are trustworthy even though absolute
numbers are optimistic.

## Correctness bar — non-negotiable for everyone

The model forms a **bistable massive activation** (|h| ~ 5300 at one token in layer ~16). *Any*
rounding difference from *any* fused op flips it for that token; pure torch matches only because it is
the reference computation. Consequences:

- Keep **fp32 accumulate and bf16 stores at exactly the points the current kernels use**
  (`_gemv_kernel`, `_gemv_silu_kernel`, `_reduce_add_rms_kernel` in `engine/fused.py` are the
  reference). A changed split-K count changes the reduction order — that is a numerics change, not a
  perf knob.
- Add every new fused op to `Engine._selftest()`, which runs torch vs fused on real weights at load
  and falls back to slow-but-correct rather than shipping wrong output.
- Required before any hand-in: `tests/bench.py --check-all` clean on the 3 public shapes **and**
  `--shapes 32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`, plus `--corpus code`
  and `--corpus repeat`. Required: `worst gap <= 2.0`, `positions > 2.0: 0`, `spread <= 25%`.
- **Pre-existing violations you must not worsen** (all confirmed on `main` itself, A/B'd against the
  fused paths, native control clean; **all official runs have passed**, so the private prompt
  distribution appears safe):
  - B32 code corpus: 3.9 logits (seq 24 step 85)
  - B64 natural: 3.25 (seq 0 step 99)
  - **B4 2048->32 `--corpus repeat`: 2 violations, gaps 2.5 and 4.125 (seq 0, steps 9 and 27)** —
    re-confirmed by S3, bit-identical with the fused path disabled. Note this is the *highest-weighted
    public shape*, but `repeat` is a synthetic local stress corpus, not the platform's distribution.

Platform failure modes: `incorrect_output`, `candidate_error`, `timeout`, `latency_limit` (TTFT or
TPOT above 1.10x native), `memory_limit` (peak above 90% of 80 GB), `unstable_timing` (5-sample spread
above 25%).

## Why the official score is noisy (and what to do about it)

Three official draws of the **same** build (`1e460f5`) scored **1041.1 / 1047.3 / 779.4**. Decode TPOT
was identical to 0.01 ms across all three (3.49 / 4.02 / 3.98 at B1/B4/B16) — including the 779 run.
What moved was public-2 TTFT (105 / 102 / **151 ms**, p10-p90 spread 3 ms -> 106 ms) and `metricMs`
(+34%) against only -10% on public tok/s.

Conclusions, all load-bearing: (1) there is **no nondeterministic cliff in our code** to hunt — it is
platform contention on compute-bound prefill; (2) the private set is **prefill-heavy**; (3) the score
has a heavy **left** tail the pod cannot see (pod spread is 0.4-1.3%), so sub-1% pod wins are not
verifiable officially — **judge by pod A/B and do not let one official run talk you out of, or into,
a pod-measured change.** Reruns are free draws (only the best eligible run counts) but they are a
lottery, not an improvement.

**But draws are free and unlimited, and only the best eligible run counts** — there is no dollar cost
and no downside to a bad draw beyond the queue slot. The queue is serial per team at ~20 min per round
trip, so ~30 draws fit before the deadline. The orchestrator keeps the queue busy: land new builds
first, and fill every idle slot with another draw of the current best submission
(`POST /submissions/<id>/runs {"mode":"official"}`). Sampling the tail is worth real points on its own.
**Workers do not trigger runs** — the orchestrator owns the queue, because a worker's push would
evict a landing build.

## Forbidden shortcuts — every one of these fails the run

Generic "how to make LLM inference fast" advice does not apply here, because the judge replays your
tokens through native Qwen teacher-forced on your own prefix and every token must be the argmax of the
native logits or within 2 logits of it. That 2-logit margin absorbs BF16 near-ties, **not**
approximations. So all of the following are out, no matter how much they would help:

- **Weight quantization (INT8/INT4/FP8/FP4)** — explicitly forbidden, and it changes the logits.
- **KV-cache quantization** — same.
- **Sliding-window / sparse / approximate attention** — changes the output. The model is full-attention.
- **A smaller or distilled model** — the checkpoint is pinned to
  `Qwen/Qwen3-4B-Instruct-2507` rev `cdbee75f17c01a7cc42f958dc650907174af0554`, the engine gets a
  read-only `model_path`, and there is **no network** in the engine.
- **A draft model for speculative decoding** — there is no second checkpoint to load and nothing to
  download. *Exact* speculative decoding is legal, and we built it (`_generate_spec`, `_NG`, exact,
  1.35-1.9 tok/step on local corpora) — but the official result was **965.3 vs 976.1 for the plain
  build**. Platform prompts have far less repetition than pydoc/wiki/code windows, so drafts rarely
  hit and the verify overhead dominates. It is off (`ENGINE_SPEC=1` to enable). Do not revive it.
- **"Counting accepted tokens" to inflate the metric** — the score is
  `batch * out_tokens / median generation seconds`, i.e. wall-clock for a fixed token count. There is
  nothing to inflate; you must generate exactly `max_new_tokens` steps.
- **PagedAttention** — legal (it is exact) but pointless here: batch and cap are fixed per workload
  and the static KV cache is already faster.

Named examples that have come up and are **all out**: Liger-Kernel-style fusion (we already did it —
add+rmsnorm, silu*up, qk-norm+rope+KV-write, split-K reduce+residual+norm, shipped in v2-v5, and
fusing RMSNorm into the GEMV *prologue* was measured **slower**); TurboQuant / QJL / any 3-bit or FP8
KV-cache compression (lossy — "~0.985 cosine similarity" is nowhere near argmax-exact, and the
published 144.7 tok/s is a single-stream number on a 1.5B model on a B200, not a comparable
operating point); draft-model speculative decoding (no second checkpoint exists and there is no
network).

What *is* true from the generic advice: decode is memory-bandwidth bound, and 3.35 TB/s is the hard
ceiling. That is exactly what this plan attacks — we are at 68% of it. But note the generic advice
also blames **kernel launch overhead**, and on this engine that has been measured: **launch gaps are
only 2.2% of the step.** The loss is idle SMs (wave quantization), which is a different and much
bigger problem. Do not go hunting launch count.

KV traffic is real but secondary: 1.51 GB/step at B16 512->128 against 8.04 GB of weights, so ~16% of
the bytes. It grows with context, which is why the B4 2048 regime matters.

## Measured draw distribution (updated live)

Five official draws of the **identical** engine (`engine/` tree `fd3cdfab…`):
**779.4 / 1039.8 / 1041.1 / 1047.3 / 1055.9**. Median ~1041, best 1055.9, one severe left-tail outlier.
The median draw sits ~1.4% *below* our best, so extra draws are a free option on the upside.

**Correction to an earlier claim in `CONTEXT.md`:** it says decode TPOT is reproducible "to 0.01 ms"
across draws and that all variance is TTFT. Two fresh draws show TPOT also moves ~1.3-2.0%
(public-0 3.42 -> 3.49, public-1 3.96 -> 4.02, public-2 3.93 -> 3.98) alongside TTFT
(public-1 113.2 -> 117.3 ms). On the decode-heavy public-2 the TTFT actually *fell* while the total
rose, so that shape's variance is mostly TPOT. **Practical rule is unchanged and now better founded:
an official run cannot confirm anything below ~1.5%. Judge by pod A/B.**

## Known-broken tooling

`tests/budget.py` on `origin/main` raises `KeyError` at startup: `Engine._state` keys states as
`(B, cap, W, slot, decode_graph)` and `generate()` passes `slot=S, decode_graph=(n>1)`, but the
script looks up the old 3-tuple `(B, cap, 1)`. Fixed on the orchestrator checkout; if your pod has an
older copy, replace the lookup with a version-agnostic match:
```python
st = next((v for k, v in eng.states.items() if k[0] == B and k[1] == cap), None)
```

## MEGAKERNEL IS DEAD — killed at Gate C1, 2026-09-20 ~01:30 UTC (agent S4)

The design rested on one assumption: that an arrive/wait barrier's latency would hide behind the
cp.async weight loads a CTA issues for op i+1 before spinning on op i. **Measured false.** Grid-wide
arrive/wait at 132 CTAs costs **1056 ns raw and 1078 ns with loads in flight** — it does not overlap
with anything. Five implementations swept (atomicAdd+acquire-spin 1070, `ld.global.cv` poll 1015,
separate-flag publish 1514, nanosleep backoff 1517, two-level tree 1763); the naive one is the best
one. Budget was 145 ns.

Fused-layer measurement, real B16 byte pattern, 36 layers on cold memory, grid=132 thr=1024:

| variant | us/layer | TB/s | vs separate launches |
|---|---|---|---|
| separate launches | 68.4 | 2.33 | — |
| fused, **0** syncs | 58.5 | 2.73 | **+17.1%** |
| fused, 4 syncs | 66.1 | 2.41 | +3.5% |
| fused, **6** syncs (realistic) | 68.5 | 2.33 | **-0.0%** |
| fused, 8 syncs | 70.7 | 2.25 | -3.2% |

A real decode layer needs ~6 grid syncs, and every GEMV is an all-to-all dependency (each CTA needs
the whole activation vector), so none can be a cheaper point-to-point sync. **Break-even at best** —
and this is an *upper bound* with no mma, no smem staging, no attention and no correctness
constraints. In the real engine it is worse, because the baseline above has no PDL while production
does (+4.5%): the megakernel replaces a PDL-*overlapped* launch boundary with a hard barrier that
cannot overlap. Reproduced across 5 (grid, threads) combos.

**Arithmetic correction to the ramp model.** Fusing recovers only **~2.5 us per boundary**, not the
~5 us I estimated, while a barrier costs ~2 us — net ~0.5 us per boundary, and with 6 syncs against
4 fused ops per layer it goes negative. **The ~900 us of per-launch ramp is real but NOT addressable
by a megakernel.** Corrected per-op floor is ~3380-3630 us/step vs today's 4192, i.e. **13-19% of
remaining decode headroom, not 24%**, sitting in ops already at 70-85% of their own achievable ceiling.

**Two salvaged positives, both live:**
1. **Outstanding-load depth is worth up to 2.6x on LONG kernels.** A resident 132-CTA kernel goes
   **1.2 TB/s with 1 load in flight per thread to 3.176 TB/s (95% of roofline) with 8-16**, at
   512-1024 threads/CTA. It does nothing for the short decode GEMVs (ramp-bound, flat across depth)
   but it is why the persistent kernel reached 95%. **Check any long kernel — prefill especially —
   for load depth and `num_warps` before tuning anything else.**
2. **Ramp vs transfer size, measured** (same grid, separate launches vs one resident kernel):
   21 MB **1.53 -> 2.94 TB/s (+92%)**, 31 MB +35%, 50 MB +17%, 100 MB +5%, 200 MB ~0%. This is why
   `o` (21 MB, 11.9 us, 53% of roofline) is the worst op in the engine — it sits at the very worst
   point on that curve. The lever for it is deeper PDL overlap, not regridding.

Independent confirmation of S1's kill: qkv pure-read bandwidth is 2.21 / 2.28 / 2.25 / 2.25 TB/s at
96 / 132 / 264 / 528 CTAs — grid count does essentially nothing.

**Benchmarking traps S4 paid for:** timing a single ~20 us kernel with syncs either side is dominated
by launch overhead (loop the launches, use cold slabs — L2 is 50 MB and will hide a whole weight
matrix). Benchmarking `TritonOps.linear` standalone with `ENGINE_PDL=1` reports nonsense (7556 us of
GEMV inside a 4192 us step) because each launch stalls in `griddepcontrol.wait` with no real producer.
And `grid=264` at 1024 threads is 1 CTA/SM = 132 resident = **deadlock** for any spin-wait scheme.

## POD A/B NOISE FLOOR IS ~1% ACROSS PROCESS LAUNCHES

Agent S3, measuring `ENGINE_PREFILL_GRAPH` with **no code and no env change between the two arms**,
saw the same pair swing about 1%:

| pair | baseline | candidate | delta |
|---|---|---|---|
| 1 | 1058.1 | 1068.0 | **+0.94%** |
| 2 | 1065.2 | 1065.9 | +0.07% |
| 3 | 1065.5 | 1065.6 | +0.01% |

Mean +0.34%. **The first pair was the outlier, not the signal.** Within-process 5-sample spread is
0.4-1.3%, but *process-launch-to-process-launch* variation is about 1%, and a single A/B pair cannot
see the difference.

**Consequence: a single pod A/B pair cannot resolve a sub-1% change.** Run at least three pairs and
compare means before believing anything under ~1%. This retroactively means several "+0.5-1%"
results tonight were unresolvable as stated — prefer a direct per-kernel measurement (which has a
much smaller error bar) plus a mechanism, and treat the end-to-end A/B as corroboration rather than
proof.

## ENGINE_PREFILL_GRAPH — flat, confirmed not an artifact

`PREFILL_GRAPH_MAX` defaults to 4096 and both public shapes carrying 89% of the weight are B*S=8192,
so they were never graphed. Raising the cap to 8192 to capture them: **+0.34% mean over 3 pairs**,
under the bar. 16384 is no better (1064.3, slightly worse than 8192 — 8192 already covers both
shapes exactly). B4 TTFT 111.8 -> 111.4 ms, B16 100.8 -> 100.0 ms.
Capture was verified real (`st.pgraphs[S]` holds a `(sid, CUDAGraph)` tuple, not `False`), and peak
memory is *better* graphed, not worse (B4 17.4 -> 16.6 GiB, B16 18.8 -> 18.0 GiB) — graph-pool reuse
beats eager allocation. Audit shapes `1,8192,64` and `2,3000,32` clean at cap=16384.
**So the original "flat on public shapes" note was correct, not an artifact of the exclusion.**

## STANDING RULE (earned three times tonight)

**Nothing from a standalone Triton microbench ships without the `tests/bench.py` 5-sample in-engine
number, no matter how large the standalone delta looks.** Instances: (1) balanced-grid GEMV; (2)
attention BN=32/NW=2 measured **+7.9% standalone** and **+0.07% in-engine**, with B4 — the regime with
the biggest standalone win — actually *regressing*; (3) the historical 128-vs-256 attention target,
where the no-PDL microbench favoured 128 and the engine favoured 256.

## Measured ramp, and why it is not recoverable

Ramp per kernel, measured against the steady-state implied by lm_head's 3.08 TB/s (agent S1):

| shape | measured | steady | ramp | % of measured |
|---|---|---|---|---|
| qkv | 14.10 us | 10.21 | +3.89 | 27.6% |
| o | 12.43 us | 6.81 | **+5.62** | **45.2%** |
| gate_up | 36.14 us | 32.34 | +3.80 | 10.5% |
| down | 24.78 us | 16.17 | +8.61 | 34.7% |

**21.9 us/layer x 36 = 789 us = 18.8% of the 4192 us decode step.**

Batching R *independent* GEMVs into one launch amortises it, confirming the mechanism: qkv per-round
16.04 us at R=1 -> 11.19 us at R=32; o 18.19 -> 9.72. Linear fit qkv a=10.35 us fixed + 10.89 us/round.

**But it is not recoverable in the real engine.** Those R rounds are independent; a real decode layer's
ops are all-to-all dependent (every GEMV needs the whole activation vector), so fusing them needs
grid-wide syncs, and S4 measured a fused layer at **+17.1% with 0 syncs but -0.0% at the 6 syncs a
real layer needs**. S1's amortisation curve and S4's sync curve are the two halves of the same result.

Loose thread nobody has closed: `o` sits at only ~72% of bytes-implied throughput **even at full
amortisation** (9.40 us/round steady vs 6.81 implied), i.e. it has a tile-config inefficiency
*separate* from ramp.

## Decode attention — closed

`num_warps` 4 -> 8 -> 16 -> 32 monotonically worse: B4 18.55/19.73/28.14/41.55 us, B16
18.76/19.52/22.62/34.70. **32 warps is 2.2x slower than 4.** The opposite direction (BN=32, NW=2,
ST=3) gave +7.9% standalone on the B4 path but **+0.07% in-engine** — killed. PDL producer-timing is
blocked structurally: Q and the KV write come from the same `qkv_post` invocation, so no subset of
attention splits can skip `_gdc_wait()`; fixing it needs `qkv_post` split into Q-only and KV-write
kernels, a timing-sensitive change under graph capture.

## Dead levers — measured, do not repeat

PF / L2 prefetch depth (PF=4 best; PF>=32 is **14-24% worse** — prologue issue cost delays
`griddepcontrol.wait`) | GEMV tile/stage/warp configs (255 swept, +0.0-0.1%) | cache modifiers and
eviction policies | mask-free GEMV `EVENK` (+0.7% only; PTX already emits `cp.async ... 0x10` with or
without masks, so vectorisation was never the limiter) | FMA GEMV (5% at M=1 only) | FMA decode
attention (10.7 vs 6.6 us at B1) | GEMV weight re-tiling (<=3%) | **persistent SM-balanced GEMV
(-5 to -10%) — but see the caveat below** | fused last-block attention combine via atomic counter
(-2%: serial combine + barrier on the critical path) | qkv_post folded into the GEMV epilogue or into
attention | RMSNorm fused into the GEMV prologue (slower: serialises a pass over x before weight
loads) | Triton fused-SiLU prefill GEMM (592 vs cuBLAS 756 TFLOP/s, -10.5% net) | cuDNN SDPA prefill
(no net win once KV is expanded for GQA and the output needs a copy) | n-gram speculative decoding
(exact, 1.35-1.9 tok/step locally, but **official 965.3 vs 976.1 plain** — platform prompts have far
less repetition than local corpora; `ENGINE_SPEC` off) | nsplit==1 attention combine skip (0 on the
private set) | CUDA-graphed small prefills (flat on public shapes) | attention KV prefetch before the
PDL wait (noise) | per-site PDL prefetch sweep (+-0.5%).

**BALANCED-GRID GEMV IS DEAD — falsified 2026-09-20 ~01:00 UTC by agent S1. Do not rebuild it.**
An earlier version of this file argued that `tests/gemv_persist.py`'s "-5 to -10%" result did not
falsify the balanced-grid thesis, because it swept `BN in (16,32)` only and never BN=64. S1 ran the
corrected experiment: `grid=(132,)/(264,)`, uneven contiguous row ranges, BN swept **including 64**,
register-resident accumulator held across the whole K loop, single store at the end, SK=1, 108 configs,
H100 SXM, 32 distinct weight copies to defeat L2. Result:

| op | best balanced vs production | best TB/s |
|---|---|---|
| qkv (6144,2560) | **+1.2%** | 2.29 (target was >=3.0) |
| o (2560,4096) | **+1.0%** | <=1.75 |
| down (2560,9728) | **-7 to -9%** | 2.01-2.12 |
| lm_head | -1.1 to -1.5% | ~3.0 |

The optimiser *did* choose NCTA=132, BN=64 for qkv. Moving qkv from 73% to 100% SM utilisation should
have cut ~34% if idle SMs were the bottleneck; it cut ~1%. **So `pred = util * 3.35 * 0.93` is a
descriptive fit, not a causal mechanism — do not cite it, and do not plan against it.**

**The surviving explanation is kernel DURATION, not occupancy.** The small GEMVs run 12-25 us and lose
roughly 3-5 us each to memory-pipeline ramp and drain at the kernel boundary. lm_head runs 250 us
(18 waves) and amortises the same ramp down to ~1.5% — which is the whole reason it looks "fast and
fully occupied". Estimated prize, per layer at B16: qkv ~3.2 + o ~2.4 + gate_up ~4 + down ~5.7 =
~15 us of a 116 us layer, i.e. **~13% of the decode step (~550 us)**. Same size as before, but it lives
at the **kernel boundaries**, not in the grid shape. Agent S1 is now measuring this directly.

Corollary worth remembering: `down` got *worse* at 132 CTAs. Fewer, longer-lived CTAs means fewer
outstanding loads and less memory-level parallelism. **More CTAs can help; do not assume 1 CTA/SM is
free.**

The only design that attacks a kernel-boundary cost is the **megakernel (S4)**: resident CTAs issuing
cp.async for op i+1 before spinning on op i, so the pipeline never drains. PDL is a partial version of
the same idea and is worth +4.5% today, which is independent support for the mechanism.

## What is currently switched on

PDL (programmatic dependent launch) is on: kernels launch with the PDL attribute so each starts while
the previous drains, calling `griddepcontrol.wait` before reading its producer's output. Trigger
placement matters — `launch_dependents` at kernel *start* made GEMVs 4% slower; at the *end of the K
loop* (`ENGINE_TRIG=1`, default) it gave **+4.5% pod geomean**. It needs a launcher patch: Triton
loads its driver module under a different object than `import triton.backends.nvidia.driver`, so the
patch targets `driver.active.launcher_cls.__init__.__globals__['make_launcher']`. `_probe_pdl()`
(a chain of 64 dependent PDL launches in a graph) disables PDL on any error or miscount.
`ENGINE_PDL=0` turns it off.

Env knobs: `ENGINE_PDL`, `ENGINE_TRIG` (1), `ENGINE_PF` (4), `ENGINE_EVENK` (1), `ENGINE_ATTN_TARGET`
(256), `ENGINE_ATTN_TARGET_BIG` (128), `ENGINE_ATTN_ST` (3), `ENGINE_SPEC` (off), `ENGINE_OFF`
(disable a fused op for bisection: `attn`, `qkv`, ...).

## Profiling traps that have cost hours

- **PDL on: a waiting kernel is charged its producer's time**, so `budget.py` sums read 124-131% of
  the step. Attribute with `ENGINE_PDL=0`.
- `budget.py` replays with `pos += 1`; **rewind `st.pos`** before each block or attention reads past
  `cap` and you get an illegal access.
- Graph re-capture needs a **fresh** `torch.cuda.graph_pool_handle()`. Re-capturing into the same
  mempool trips a `CUDACachingAllocator` assert and silently drops you to eager decode.
- Any tensor read **inside** the CUDA graph must be a persistent buffer updated in place (`st.tok`,
  `st.pos`). Allocations inside capture are fine (shared graph pool); host-side Python values are
  baked in at capture.
- KV cache is init'd with `zeros`, not `empty` — masked slots must be finite (`0 * NaN = NaN`).
- The platform runs **one shape per fresh process**; graph capture happens in the untimed warmup
  generate. Engine load + warmup is untimed but capped at 300 s. Don't pre-warm every shape in
  `__init__` (that was tried and dropped in `cebcf22`).
- Memory cap is 90% of 80 GB; each `(B, cap)` state holds a full KV cache, `MAX_STATES=6`.
