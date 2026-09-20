# Orchestration: 6 workers, ~11.5 h to deadline

**Owner: Krish. Orchestrator: this session (Opus). Workers: 4 Sonnet subagents + 1 Opus subagent
(S4) + Juan (S6, his own Claude), one pod each.**
Deadline **Sun 08:00 ET = 12:00 UTC**. Now: Sat 20:20 ET / Sun 00:20 UTC.

## The target moved

| rank | team | score | metricMs | achieved |
|---|---|---|---|---|
| 1 | **Segfault** | **1198.9** | **422.5** | 00:18 |
| 2 | SSS | 1144.3 | 442.6 | 21:30 |
| 3 | 0xDeadBeaf | 1142.9 | 443.2 | 00:35 |
| 4 | dryfter | 1138.7 | 444.8 | 23:35 |
| 5 | Silver Bullet | 1137.7 | 445.2 | 20:52 |
| 6 | zip | 1071.1 | 472.9 | 23:00 |
| **7** | **krish** | **1047.3** | **483.6** | 23:54 |

`score * metricMs = 506522` exactly. Segfault has gone **1064.4 -> 1176.4 -> 1198.9 in ~1.5 h** and
0xDeadBeaf appeared at 1142.9 out of nowhere. **Plan for a finish line near 1300** (metricMs ~390),
i.e. **-19% aggregate**, not the -11% that would beat today's number.

**How they are doing it, from their own metric:** `422.5/483.6 = 0.874` aggregate. With prefill ~25%
and unchanged, their decode step is ~3486 us against the 2851 us roofline = **~82% of roofline**.
We are at **68%**, and our own `lm_head` already hits **92%**. So there is no exotic technique to
reverse-engineer — they simply do not waste SMs. That is precisely what the thesis below predicts and
what S1-S4 fix.

**Consequence: no single agent's win is enough.** S1+S2 landing perfectly puts us at ~1146, which is
roughly where the leader already is. We need three or four of six to land, and S4 (megakernel) is what
buys margin instead of a tie. Nobody stops early because the thesis is proven — everyone finishes
their own task.

## The thesis everything is built on

Every decode GEMV launches a CTA count that does not divide 132 SMs. Utilisation, not the inner loop,
sets the bandwidth:

| op | shape (N,K) | cfg (BN,SK) | CTAs | waves | SM util | pred TB/s | measured |
|---|---|---|---|---|---|---|---|
| qkv | 6144, 2560 | 64, 1 | 96 | 1 | 72.7% | 2.23 | ~2.2-2.4 |
| o | 2560, 4096 | 64, 2 | 80 | 1 | 60.6% | 1.88 | ~1.8 |
| down | 2560, 9728 | 32, 4 | 320 | 3 | 80.8% | 2.48 | ~2.5 |
| gate_up | 9728, 2560 (x2) | 32, - | 304 | 3 | 76.8% | 2.36 | 2.76 |
| **lm_head** | 151936, 2560 | 64, 1 | 2374 | 18 | **99.9%** | 3.08 | **3.08** |

`pred = util * 3.35 * 0.93`. Fits the measured 1.8-2.4 TB/s band on 4 of 5 ops and nails lm_head.
lm_head is the control: the only ~fully-occupied op is the only one at full bandwidth.
**The 30% roofline gap is idle SMs.** Triton block sizes are powers of two, so 132 CTAs is
unreachable — the 255-config sweep was closed under the defect, and the peer's "persistent GEMV is
5-10% slower" result swept `BN in (16,32)` only, never 64, so it compared a balanced grid of *worse*
tiles against an unbalanced grid of *better* tiles. **The hypothesis has never been tested.**

Fix: CTA `c` owns rows `c*N/132 .. (c+1)*N/132` (46 or 47 rows for qkv — uneven, which is the whole
point), tile shape stays BN=64 with a row mask. Masked loads do not fetch, so HBM traffic stays exact;
the wasted tensor-core lanes are free in a memory-bound op. Bonus: perfect row balance means
**SK=1 everywhere**, which deletes `_splitk_reduce_kernel` and shrinks `_reduce_add_rms` (215 us,
5.1% of the step, 72 launches).

## Rules stance (settled by the owner — do not relitigate)

**If the platform accepts the submission, it is allowed.** No Slack question, no waiting. We probe by
submitting a small real change rather than building everything on an unverified assumption. NVRTC
ships no binary (source string in a `.py`, compiled at load) and `budgets.max_compile_seconds = 600`
shows load-time compilation is expected. The old `info.md` rule "No NVRTC/CUDA-string code in
`engine/` until Krish confirms organizers allow it" is **superseded** — `info.md` has been rewritten
as S6's brief and no longer carries it.

**Do NOT run the canary** in `experimental/nvrtc/cudart.py` (`canary()` encodes a probe result as a
per-step sleep). It is obsolete: S3's probe push is a real feature and answers the same question.

## Worker split (6 agents, 6 pods)

| # | agent | model | owns (file regions) | first deliverable | ceiling |
|---|---|---|---|---|---|
| S1 | `gemv-balance` | Sonnet | `_gemv_kernel`, `GEMM_CFG`, `TritonOps.linear/linear_add_norm` | balanced qkv/o/down/lm_head GEMV | **-8%** step |
| S2 | `mlp-attn-balance` | Sonnet | `_gemv_silu_kernel`, `SILU_CFG`, `gate_up_silu`, decode attention | balanced gate_up (31% of step) | **-6%** step |
| S3 | `nvrtc-gemv` | Sonnet | `engine/cudart.py`, CUDA GEMV | **platform-acceptance probe push (T+60 min)** | **-5%** more |
| S4 | `megakernel` | **Opus** | `experimental/mk/` then `engine/mk.py` | C0 probes (barrier, ceiling, occupancy) | **-20%** step |
| S5 | `prefill` | Sonnet | prefill path in `engine/fused.py` + `Engine._prefill` | prefill attention + non-GEMM overhead | **-5%** aggregate |
| S6 | `glue-verify` (Juan) | his own Claude | glue kernels, safety net, **independent verification** | verification service, standing | **-4%** step |

Handoffs: `handoffs/S1..S5-*.md`, and `info.md` for S6 (Juan reads `info.md`, so his brief lives
there). Everyone also reads `handoffs/SHARED-CONTEXT.md` — scoreboard, rules stance, protocol,
correctness bar, dead levers, profiling traps — and then `CONTEXT.md`.

S1 and S2 apply the *same* technique to disjoint kernels — that is deliberate: it doubles the chance
the thesis lands and the two merge cleanly. **File-region ownership is strict**; no agent edits
another's region. Conflicts get resolved by the orchestrator, not by the workers.

**S4 runs on Opus.** It is the only path past ~1150 and the hardest thing on the board: an
instruction-tape persistent kernel with hand-rolled arrive/wait sync and cross-op cp.async prefetch.
It is also the most likely to be killed at its own Gate C0, which is fine — a fast, honest kill is
worth a pod.

**Juan is now worker S6, not a duplicate.** `info.md` was the general T0-T5 task list; it has been
rewritten as his specialised brief and the old tasks are explicitly marked superseded (T0 closed by
the owner's rules stance, T1 -> S5, T2 -> S1/S2/S3, T3/T4 -> S3/S4). He owns the decode glue kernels
(`_reduce_add_rms_kernel` 215 us / 5.1% / 72 launches, `_qkv_post_kernel` 84 us — both latency-bound
at 16 CTAs on 132 SMs, not bandwidth-bound), the safety net (`_selftest`, `_TorchOps`, state/memory),
large-batch correctness insurance, and the **standing verification duty**: he re-runs every lander's
A/B and `--check-all` on his own clean pod before the orchestrator pushes. With six parallel
workstreams and a hard freeze, that independent check is what keeps a regression off `main`.

## Timeline (ET)

| t | orchestrator | workers |
|---|---|---|
| 20:20 | spawn S1-S5, brief S6 | provision pods, scp repo tarball, warm caches |
| 21:20 | **push S3 probe to main** (NVRTC acceptance) | S1/S2 standalone kernel benches; S4 gate C0 |
| 22:00 | read probe result -> green/red light S3/S4 | S1/S2 gate A; S6 verifies first lander |
| 23:00 | merge + A/B first lander, push | S3 CUDA GEMV, S4 one-layer megakernel |
| 01:00 | merge second lander, push | S4 full step; S6 verifies |
| 03:00 | merge third, push | S5 prefill A/B |
| 05:00 | **feature freeze**; final integration A/B (S6 verifies the combined build) | hand in diffs, stop pods |
| 06:00 | push final build to main | — |
| 06:30-07:30 | rerun draws on the best submission (lottery; official score has a heavy left tail) | — |
| 07:45 | stop all pods | — |

**Feature freeze 05:00 ET is hard.** A run is ~7-10 min queue + ~13 min; we need 3-4 clean draws.

## Merge protocol

1. Workers **never push to `main`** and **never edit the Windows checkout** (another session is live
   in it). They work on their own pod, on branch `krish/<agent-name>`, and hand back a diff.
2. Orchestrator merges in landing order S1 -> S2 -> S6 -> S5 -> S3 -> S4, re-running `--check-all`
   after each merge, then `git fetch && git merge origin/main` before every push to main.
3. Ship bar per lander: **pod A/B (5 samples) >= +1.5% predicted score AND `tests/bench.py
   --check-all` clean** (worst gap <= 2.0, 0 positions > 2.0, spread <= 25%) on the 3 public shapes
   plus `32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`, **independently reproduced
   by S6** on a clean pod.
4. Every new path is env-gated and defaults **off** until its A/B passes, so a bad merge is one line.

## Decisions taken (previously open)

- **Official runs:** approved, and the budget is now *continuous*, not ~6. Runs cost nothing, only
  the best eligible run counts, and the queue is serial at ~20 min per round trip — so ~30 draws fit
  before the deadline. Land builds first; fill every idle slot with another draw of the current best.
- **No free Dryft dev GPU exists.** Verified against the API: every run is `mode: "official"`, and
  `/me`, `/sandboxes`, `/devboxes`, `/runtimes`, `/gpus` all 404. The grading harness returns three
  public tok/s numbers plus TTFT/TPOT with submission stdout suppressed — it cannot profile a kernel,
  and it is serial. RunPod is the development loop, not a testing afterthought; the 11 h is simply
  wall-clock to the deadline with agents holding a GPU while they work.
- **S4 on Opus:** yes.
- **Juan:** worker S6, scoped to glue + safety + verification. No overlap with S1-S5.
- **GitHub credentials on fresh pods (owner: "I don't know"):** resolved without them. Workers get the
  code by `git archive origin/main` (~287 KB) scp'd to their pod, then `git init` a local base commit;
  they hand work back as a diff. Nothing on a worker pod needs push rights, and no token leaves this
  machine. If a PAT turns up later it is a convenience, not a dependency.

## Cost

6 pods x ~11 h x $3.49 = **~$230**, plus the idle pod `6ewaafott6x3hh`. Owner set no ceiling. Every
agent stops its pod on hand-in; the orchestrator sweeps for stragglers at 07:45 ET.

## Orchestrator's own checklist

- Do not let a single official run overrule a pod A/B in either direction (three draws of the same
  build scored 1041.1 / 1047.3 / 779.4).
- `git fetch && git merge origin/main` before every push — one push was rejected for skipping it.
- Watch the leaderboard hourly; Segfault gained 132 points in ~2 h, so the bar may move again.
- Keep `CONTEXT.md` > Findings updated with every negative result workers report, so nothing gets
  re-tried by the next agent.
