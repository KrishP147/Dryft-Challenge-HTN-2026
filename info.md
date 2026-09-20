# info — S6 `glue-verify`: worker brief for Juan

You = worker agent. Krish = owner/decider. A separate Opus session = orchestrator; it merges every
worker's branch and does all pushes to `main`.

**Read `handoffs/SHARED-CONTEXT.md` first** (scoreboard, rules stance, protocol, dead levers,
correctness bar, profiling traps — all of it applies to you), then `CONTEXT.md` (design and full
experiment history). This file is only what *you* own.

> This file used to be the general task list (T0-T5). **That version is superseded.** T0 (the
> organizer rules question) is closed — the owner's stance is now "if the platform accepts the
> submission, it is allowed", and agent S3 is confirming it with a real push. T1 (prefill) is agent
> S5's. T2 (small-GEMV ramp) is agents S1/S2/S3's. T3/T4 (grid barrier, CUDA GEMV) are agents
> S4/S3's. **Do not work those** — you would duplicate five other agents' hours.

## Who is doing what (so you don't collide)

| agent | owns | do not touch |
|---|---|---|
| S1 `gemv-balance` | `_gemv_kernel`, `GEMM_CFG`, `TritonOps.linear` / `linear_add_norm` | — |
| S2 `mlp-attn-balance` | `_gemv_silu_kernel`, `SILU_CFG`, `gate_up_silu`, `_attn_split_kernel`, `_attn_combine_kernel`, `attn_decode` | — |
| S3 `nvrtc-gemv` | `engine/cudart.py`, CUDA GEMV, the platform-acceptance probe push | — |
| S4 `megakernel` | `experimental/mk/`, `engine/mk.py` | — |
| S5 `prefill` | `Engine._prefill`, `_capture_prefill`, `_qkv_post_prefill_kernel`, flash prefill | — |
| **S6 (you)** | **the decode glue kernels, the safety net, and verification — below** | everything above |

**Your region, exactly:** `_reduce_add_rms_kernel`, `_add_rms_kernel`, `_splitk_reduce_kernel`,
`_qkv_post_kernel` (the **decode** one, not the prefill one), `_silu_mul_kernel`, and their
`TritonOps` wrappers (`rms`, `add_rms`, `silu_mul`, `qkv_post`, `_rms_launch`); plus
`Engine._selftest`, `_Mixed`, `_TorchOps`, `_state`, `_host_bufs`, `MAX_STATES`.

If a change would touch someone else's region, **route it through the orchestrator** — don't edit it.

## J0. Verification service — standing duty, highest value

Five agents are optimising in parallel against a **05:00 ET feature freeze**, and each push to `main`
costs a serial queue slot. You are the only independent check between a worker's claim and a push.

When the orchestrator hands you a branch or patch: apply it to a clean tree on **your** pod and
re-run, from scratch:
- `python tests/bench.py` — 5 samples, B1/B4/B16, predicted score, **spread**, before vs after;
- `python tests/bench.py --check-all` on the 3 public shapes **and**
  `--shapes 32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`, plus `--corpus code`
  and `--corpus repeat`;
- confirm the decode graph still **captures** (a capture failure degrades silently to eager and only
  shows up as a slow official run — check `eng.states[...].graph is not None`);
- confirm peak memory stays under 90% of 80 GB at B16 and B64.

Report **pass/fail plus the numbers**, not a judgement call on whether to ship — the orchestrator
decides that. A disagreement between your A/B and the author's is itself the finding; say so plainly
rather than averaging them.

Interleave J1 between verification requests. Verification wins when they conflict.

## J1. Decode glue latency — your perf task (~300 us = 7% of the step)

From `tests/budget.py` (B16 512->128, PDL off, 4192 us step):

| kernel | us/step | % | launches/step |
|---|---|---|---|
| `_reduce_add_rms_kernel` | 215 | 5.1% | 72 |
| `_qkv_post_kernel` | 84 | 2.0% | 36 |

These are **latency-bound, not bandwidth-bound**, and that is the thing to attack. `_reduce_add_rms`
runs **one program per row** — at B16 that is a grid of **16 CTAs on 132 SMs (12% occupancy)**,
moving well under 1 MB, at ~3 us per launch. Nearly all of that 3 us is ramp, not work.

Lines of attack, cheapest first:
1. **More parallelism per launch.** One CTA per row wastes the machine. The RMSNorm needs a whole-row
   sum, so a naive column split needs a cross-CTA reduction — but the *residual add* and the `hn`
   store do not. Consider splitting the work so the add/store fan out while the norm stays per-row,
   or widening the per-row program (warps/BLOCK) so the single wave finishes sooner.
2. **Fewer launches.** 72 + 36 = 108 launches at ~2-3 us each is a hard floor of ~250 us. Look for
   two adjacent glue ops with no dependency between them that can share one launch.
3. **Re-measure after S1 lands.** S1 is moving the GEMVs to **SK=1** (perfect row balance removes the
   reason split-K existed), which **deletes `_splitk_reduce_kernel` entirely and turns
   `_reduce_add_rms` into a plain add+rms**. Some of your 215 us evaporates for free. Coordinate with
   the orchestrator on S1's landing before you invest hours here — and make whatever you build
   correct at **both** SK=1 and SK>1, because S1 might not land.

**Already dead in this area, do not repeat** (see `handoffs/SHARED-CONTEXT.md` for the full list):
RMSNorm fused into the GEMV *prologue* (slower — serialises a pass over x before weight loads);
`qkv_post` folded into the GEMV epilogue or into attention; the fused last-block attention combine via
an atomic counter (-2%). What *did* work historically was fusing the split-K reduce + residual + norm
into one kernel — that is the kernel you now own.

Ship bar: **>= +1.5% predicted score geomean** in a 5-sample in-engine pod A/B, behind an env gate
that defaults **off**, with `--check-all` clean.

## J2. Large-batch correctness — insurance

B32 on the code corpus gives a **3.9**-logit violation (seq 24 step 85); B64 natural gives **3.25**
(seq 0 step 99). Identical on v14 and v15; the native decode-vs-replay control is 1.0 at B64.
Cause: a bistable massive activation (|h| ~ 5300 at layer ~16) that any rounding difference flips.
It needs **both** the fused attention and the fused qkv path (`ENGINE_OFF=attn` or `=qkv` each pass).
Bisect further with `tests/trace_diff2.py` and `tests/trace_prefill.py`.

All official runs have passed and the private set probably has no batch >= 32 (the nsplit==1 change
that only fires at B>=32 moved the score by exactly 0). So this is **insurance, not a win** — but an
`incorrect_output` on one private workload fails the whole run. Any fix must not slow B<=16.

**Priority: below J0 and J1.** Do it if you have spare time, or immediately if the orchestrator tells
you a new lander made a large-batch gap worse.

## J3. Load-time preparation — on orchestrator request only

Engine load and warmup are **untimed** (300 s budget; `max_compile_seconds` is 600). Nobody owns this
budget. If S3 or S4 land, they may need weights repacked into a permuted or tiled layout at load time
— that is free real estate, and it is yours to build when asked. Do not start it speculatively; the
one thing already tried and dropped here is pre-warming every shape in `__init__` (`cebcf22`).

## Reporting

Branch `krish/glue-verify` on your own pod. **Never push to `main`; never edit the Windows checkout**
(another session is live in it). Hand back a diff (`git diff base > /workspace/w/glue-verify.patch`)
and paste it in your report — see `handoffs/SHARED-CONTEXT.md` > "Getting the code onto your pod" for
the credential-free setup.

Per task: branch, commit, pod A/B table (5 samples, before/after, spread), `--check-all` result per
shape, verdict keep/kill, one-line reason. Every negative result with numbers — the orchestrator
appends it to `CONTEXT.md` > Findings.

**Open questions for Krish are the orchestrator's to carry now. Send them there, not here.**
