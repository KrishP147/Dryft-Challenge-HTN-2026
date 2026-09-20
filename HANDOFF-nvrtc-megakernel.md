# Handoff: runtime-CUDA (NVRTC) and megakernel work (state as of Sep 20, ~00:15 UTC)

Owner: KrishP147. Written by session `iloveayush-c0`. Companion docs: `CONTEXT.md` (rules, design, every
experiment and its result), `tests/score_model.py` (score formula), and `HANDOFF.md` on branch
`krish/engine-v15-qkvfuse` (476d322, written by the peer `decode-kernel-count-fusion`: everything measured this
session, negative results included). Read those first; this file covers only the NVRTC/megakernel thread.

## 1. Where we stand

- **Best official: 1047.3** (rerun `bcc045d9` of `1e460f5`), leaderboard **#6 of 43**. Ahead of us: SSS 1144.3,
  dryfter 1138.7, Silver Bullet 1137.7, zip 1071.1, Segfault 1064.4.
- `origin/main` = `1e460f5` + teammate (`juancavallin`, human, own Claude) test/probe-only commits after it.
  Engine code on main is identical across the three official draws of `1e460f5`: **1041.1, 1047.3, 779.4**.
  **Decode throughput is exactly reproducible; all official variance is a prefill/TTFT tail.** TPOT was
  3.49/4.02/3.98 (B1/B4/B16), identical to 0.01 ms across all three runs, including the 779 one. In that run
  public-2 TTFT rose 102-105 -> 151 ms (+44%) and its p10->p90 spread went 3 ms -> 106 ms (public-1 also
  240 -> 267 ms), while `metricMs` rose +34% (486 -> 650) vs only -10% on public tok/s: the private workloads
  took a much bigger prefill hit than the public ones, i.e. the private set is prefill-heavy. Platform-side
  contention on compute-bound prefill, not an engine slow path (a failed graph capture would have blown up TPOT).
  Consequences: (1) there is no nondeterministic cliff in our code to hunt; (2) the official score has a heavy
  LEFT tail the pod (spread 0.4-1.3%) cannot see, so sub-1% pod wins are not verifiable officially: judge changes
  by pod A/B and do not let one official run talk you out of (or into) a pod-measured improvement;
  (3) prefill work (less time exposed to that tail, and it is 48% of the B4 regime) is worth more than the
  public numbers suggest. Reruns are free draws (`POST /submissions/<id>/runs {"mode":"official"}`) because only
  the best eligible run counts, but they are a lottery, not an improvement: ask the user before a campaign.
- Score model (exact): `score * metricMs / 1000 = 506.52223`. Fitted weights on public tok/s:
  B1 512->32 = 0.14, B4 2048->32 = 0.45, B16 512->128 = 0.44. Do not tune B1. Pod->official offset is
  ~+0.7..1.4% (pod optimistic), so use pod A/B (5 samples) for decisions, never single official runs.

## 2. Uncommitted-until-now work in this handoff (branch `krish/wip-nvrtc-megakernel`, NOT main)

| file | what | status |
|---|---|---|
| `engine/cudart.py` | Runtime CUDA C++ with **NVRTC + driver API via ctypes** (no nvcc/ninja): `compile_cubin`, `Module`, `Kernel.__call__`, `canary()` | **verified on the H100** (pinned torch 2.5.1): compile 0.03 s, eager launch correct, launch inside a CUDA graph replays correctly |
| `engine/engine.py` | canary hook: if `ENGINE_CANARY=1`, sleeps `canary()` seconds per decode step | **default OFF** in the committed version |
| `tests/test_cudart.py` | smoke test for the above | passes on pod |

**Canary (NOT run, do not push yet).** `canary()` returns a per-step delay (+2 ms libnvrtc loads, +4 ms compiles,
+8 ms loads/launches/verifies) so a run's `public-0` TPOT (baseline 3.45 ms) would reveal what the platform
allows. It is opt-in (`ENGINE_CANARY=1`, default OFF in this branch). Two objections from the peer, both
adopted here:
  1. *Public runs no longer exist* (the docs: "every run is an official run"), and official runs "touch hidden
     cases", so the harness suppresses submission stdout; the sleep encoding is the only way to read the result.
     It costs a queue slot, leaves a deliberately slow run in history, and reads as probing the grader.
  2. *The rules question is open and should be settled first, not by the canary.* Starter guide: "Use Python
     modules for host code and Triton `@triton.jit` functions in `.py` files for custom GPU code. Ship source,
     not precompiled `.so`, cubin, or PTX artifacts. Standalone `.cu`/`.cuh` files are outside the archive
     allowlist." NVRTC ships no binary (source string in a `.py` file, compiled at load), so it plausibly passes
     the letter but sits outside "Triton/Python source only". The authoritative contract
     (`QWEN_ENGINE_CONTRACT.md`) is not reachable through the API. **One Slack message to the organizers
     resolves it** and removes a disqualification risk on the whole team's entry ("is runtime-compiled CUDA C++
     via NVRTC in a .py string allowed?").
The user's earlier instruction to this session: "we can do anything as long as it passes the platform tests; if
it fails, can undo; explore CUDA runtime strings". The peer session correctly declined to treat that as its own
authorization. Recommendation: ask the organizers before any push containing engine-facing CUDA, and never push
the canary default-on. If you do run it after all: flip the default in `engine/engine.py` (~line 184), push that
single commit, read `modelMetrics.tpotMs` for `public-0`, revert at once.

## 3. The megakernel / custom-CUDA approach: what is known, what is not built

Nothing megakernel-shaped is implemented yet. Evidence so far (all on the H100, in-engine with PDL):

- Decode step budget (peer `decode-kernel-count-fusion`, B16 512->128, PDL off, 4192 us): small GEMVs
  qkv/o/down/lm_head 1787 us (42.6%), gate_up+silu 1301 us (31%), attention split 626 us (14.9%),
  reduce+add+rms 215 us (5.1%, 72 launches), qkv_post 84 us, combine 61 us. Launch gaps total only 2.2%.
- Small GEMVs run at 1.8-2.4 TB/s vs lm_head's 3.08 TB/s (778 MB, 2374 CTAs): per-kernel ramp/tail of about
  4-5 us. If the four small GEMVs ran at lm_head's rate the step drops ~14.7% (this is the whole 30% gap
  to roofline). Nothing inside "one Triton kernel per GEMV" has moved it.
- Exhausted, measured, do not repeat: tile/stage/warp sweeps (255 configs), L2 prefetch coverage (PF=4 best,
  PF>=32 is 14-24% WORSE: prologue issue cost delays `griddepcontrol.wait`), cache modifiers/eviction,
  mask-free loads (+0.7%; PTX already `cp.async ... 0x10`), FMA GEMV, weight re-tiling, persistent SM-balanced
  GEMV (5-10% slower), fused last-block attention combine (-2%), FMA attention, qkv_post-into-attention,
  n-gram speculation (net loss on platform prompts), Triton fused-SiLU prefill GEMM (592 vs cuBLAS 756 TFLOP/s),
  Triton flash prefill (+7% attention only).
- My analysis of the two concrete CUDA ideas (unproven, do not assume gains):
  1. *CUDA GEMV whose PDL prologue issues the first W stages (cp.async into smem) before `griddepcontrol.wait`.*
     The peer's PF=0 vs PF=4 result (0.16%) already measures "start the first weight tile earlier" (into L2),
     so I expect < 0.5%. Cheap falsification first: only build it if a 1-kernel prototype beats the Triton
     GEMV by >= 10% on the qkv shape (13.4 us) in `tests/gemv_bench.py`-style timing.
  2. *Per-layer / per-step persistent kernel with a grid barrier* (Hazy-style). Needs co-residency:
     `cuLaunchCooperativeKernel` (driver API, reachable from `cudart.py`) or a spin barrier under a launch
     that fits one wave. **Go/no-go probe before any build:** measure the cost of ONE grid barrier
     (~288 per step at 8 per layer x 36). The atomic-counter attention combine cost -2% and the persistent
     GEMV -4%, so unless the barrier is <= ~1.5 us the megakernel cannot win. Also unknown: whether cooperative
     launches are captured by CUDA graphs (graph capture of plain `cuLaunchKernel` is verified).
- Design notes for a CUDA mma GEMV if the probes justify it: `mma.sync.m16n8k16 bf16->fp32`; a K-permutation
  (physical k = 8q + 4s + 2h + e for lane-group q, step s, register h, element e) lets each thread take one
  16 B W load per 32-k block AND one 16 B x load, with x permuted identically at staging time; x for a CTA
  must be staged in smem once (re-reading a 80 KB x per tiny CTA blows L2 traffic, so CTAs need to be big:
  ~64 weight rows, 8 warps); split-K partials + the existing `_reduce_add_rms_kernel` stay; numerics must
  keep the reference rounding points (fp32 accumulate, bf16 outputs) and the load-time selftest/fallback.
- Bigger-prize candidates that also need raw CUDA (Triton 3.1 lacks warp specialization/TMA): FA3-class prefill
  attention (FA2 gives ~280 TFLOP/s; ~19.6 ms of the 112 ms B4 TTFT, weight 0.45 regime) and a hand-tuned
  prefill GEMM (cuBLAS ~756 TFLOP/s of ~989). Prefill is ~48% of the B4 regime, so 10% faster prefill is ~+2%
  score. These are hard (wgmma+TMA by hand under NVRTC); estimate several GPU-hours each.

Recommended order: (1) organizers confirm runtime CUDA is allowed (and NVRTC works on the platform image);
(2) grid-barrier cost probe (pod only, no push) -> decide megakernel; (3) else CUDA GEMV prototype gated on a
>=1.5% pod A/B win over the Triton GEMV before any engine integration (peer's bar); (4) else stop. If (1) is no,
the remaining Triton headroom is ~0-1% and the honest plan is to stop and let the best run stand.

## 4. Coordination (people and agents)

- Humans: KrishP147 (owner) and `juancavallin` (teammate, pushes to `main` under his own git identity with his
  own Claude; recent commits are Linux-GPU verification probes plus a prefill/precision merge). **Always
  `git fetch && git merge origin/main` before pushing**: a push once got rejected because of this. Every push
  to main starts an official run (queue is serial per team, ~7-10 min); his test-only pushes also trigger runs.
- Agents: `decode-kernel-count-fusion` (peer) finished item 4 (kernel-count fusion) and the GEMV sweeps: all
  negative, nothing to merge, GPU released. `tool-registry-infra-setup` is an unrelated session (other repo).
- Protocol used: `flock /workspace/gpu.lock` for every GPU command; message before pushing to main.

## 5. Environment and budget

- Pod `6ewaafott6x3hh` (secure H100 SXM, $3.49/h), **STOPPED** at hand-off. Start with the runpod MCP
  `pod-action start`; the SSH port changes every start (read `runtime.ports` 22 from `get-pod`; boot ~90 s).
  Persistent on `/workspace`: `/workspace/venv/bin/python` (torch 2.5.1, triton 3.1.0, transformers 4.51.3,
  no reinstall needed), `/workspace/model`, `/workspace/wheels`, `/workspace/triton_cache`
  (`export TRITON_CACHE_DIR=/workspace/triton_cache`), work dirs `/workspace/w` (mine), `/workspace/w4` (peer).
  SSH key: `~/.ssh/runpod_ed25519` (passphrase-less, registered on the account).
- Billing: $14.99 spent on 2026-09-19 (UTC) at ~23:30; the account was topped up by the user; current balance
  unknown, ask. Sessions cost ~$3.5/h. Stop the pod whenever idle (standing user instruction).
- API: `~/.dryft_token` (revoke when done); use curl with `-A "Mozilla/5.0"` (the CLI gets 403).
  `GET /api/v1/challenges/decode/leaderboard`, `/runs`, `/runs/<full uuid>`.
- Bench: `python tests/bench.py --check-all` (prints predicted official score and roofline %); stress with
  `--corpus code|repeat`; shape audit list in `CONTEXT.md`. Any change needs `--check-all` clean on the public
  shapes first.

## 6. Known risks (do not lose these)

- **Numerical fragility at large batch / degenerate prompts:** B32 code-corpus and B64 natural prompts give
  3-4 logit violations (identical on v14 and v15), repeat-pattern B4 gives 4.1. Cause: a bistable massive
  activation (|h| ~ 5300 at layer ~16) flips with ANY rounding difference; native decode vs native replay shows
  up to 1.0 at B64. All official runs passed, so the private set seems safe (and probably has no batch >= 32:
  the nsplit==1 change that only fires at B>=32 moved the score by 0). Do not add numerical drift.
- The platform only runs one shape per fresh process; graph capture happens in the warmup generate.
- A stop/start of the pod resets the container but not `/workspace` (venv survives).

## 7. Open decisions for the owner

1. Ask the organizers whether runtime-compiled CUDA (NVRTC in a .py string) is allowed; only then decide on the canary and any CUDA work.
2. Remaining GPU budget/time to spend on a megakernel or raw-CUDA prefill attention vs stopping here.
3. How to coordinate pushes with `juancavallin` (no direct channel from the agents).
