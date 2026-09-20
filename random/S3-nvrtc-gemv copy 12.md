# S3 — `nvrtc-gemv`: platform-acceptance probe, then a CUDA GEMV

You are a worker agent starting with zero context. Everything you need is here or in the repo files
named below. **Read `handoffs/SHARED-CONTEXT.md` first, then `CONTEXT.md`, before writing code.** They carry the
scoreboard, rules stance, protocol, correctness bar and profiling traps. Do not repeat anything in
`SHARED-CONTEXT.md` > "Dead levers".

Owner: Krish. Orchestrator: a separate Opus session that merges and pushes. You never push to `main`;
the orchestrator does, on your signal.

---

## 1. Mission

Dryft "decode": Qwen3-4B decode on 1x H100, output byte-identical to native greedy.
Segfault leads at **1176.4** (metricMs 430.6); we are #6 at **1047.3** (483.6).
`score * metricMs = 506522` exactly — score is exactly proportional to 1/aggregate time, so we need
**-11% aggregate**. Deadline **Sun 08:00 ET**, feature freeze 05:00 ET.
Regime weights: B1 **0.14**, B4 2048->32 **0.45**, B16 512->128 **0.44**. Never optimise for B1.

## 2. Rules stance — settled, do not relitigate

**If the platform accepts the submission, it is allowed.** The owner decided this explicitly. There is
no Slack question pending and nobody is waiting on an organizer.

Any older note saying "no NVRTC/CUDA-string code until the organizers confirm" is **superseded**;
`info.md` has been rewritten as agent S6's brief and no longer carries that rule. Supporting evidence: the challenge description says *"Vendor any Python or
Triton source in the archive... do not include weights, credentials, compiled binaries or Docker
images"*; NVRTC ships **no** binary (a source string in a `.py`, compiled at load), and
`budgets.max_compile_seconds = 600` shows load-time compilation is expected.

**Do NOT run `canary()`** in `experimental/nvrtc/cudart.py`. It encodes a probe result as a per-step
sleep; it is obsolete because your Task 0 answers the same question with a real feature instead of a
deliberately slow run. Leave it default-off, or delete it.

## 3. Task 0 — the acceptance probe. THIS IS YOUR FIRST HOUR. (deadline T+60 min)

Four other agents are sequenced off this result. Ship it before you optimise anything.

Build the smallest **real, useful, correct** engine change that makes NVRTC load-bearing:

1. Move `experimental/nvrtc/cudart.py` to `engine/cudart.py` (already verified working on an H100 with
   pinned torch 2.5.1: compile 0.03 s, eager launch correct, and a launch inside a CUDA graph replays
   correctly). Smoke test: `experimental/nvrtc/test_cudart.py`.
2. Reimplement **one small, self-contained kernel** in CUDA C++ via NVRTC — use
   `_splitk_reduce_kernel` or `_silu_mul_kernel`, whichever is simpler to match bit-for-bit. It must
   be inside the captured decode CUDA graph, so the probe also proves `cuLaunchKernel` graph capture
   works on the platform image.
3. **For this one push only, no fallback**: if NVRTC fails to compile or load, the engine must raise.
   A silent Triton fallback would make a passing run ambiguous, which defeats the probe. Pass =
   NVRTC works on the platform. Fail (`candidate_error`) = it does not, and we learn that in ~20 min
   instead of after six hours of building on it. The owner has explicitly authorised spending a run
   this way.
4. Keep the numerics bit-identical to the Triton kernel it replaces (see §6) and add it to
   `Engine._selftest()`.
5. Verify locally first: `python tests/bench.py --check-all` clean, and confirm the decode graph still
   captures (a capture failure silently degrades to eager and only shows up as a slow official run).

**Hand the diff to the orchestrator the moment it passes locally.** The orchestrator pushes it to
`main`, which starts an official run (~7-10 min queue, ~13 min run) and reads the result. Then restore
the fallback in your working branch while you wait.

## 4. Task 1 — CUDA GEMV (after the probe comes back green)

### The thesis you are exploiting
Decode step B16 512->128 = 4192 us vs a 2851 us roofline (**68%**). The gap is **wave quantization**,
not the inner loop, and not launch overhead (launch gaps measure 2.2%):

| op | shape (N,K) | cfg (BN,SK) | CTAs | waves | SM util | pred TB/s | measured |
|---|---|---|---|---|---|---|---|
| qkv | 6144, 2560 | 64, 1 | 96 | 1 | 72.7% | 2.23 | ~2.2-2.4 |
| o | 2560, 4096 | 64, 2 | 80 | 1 | 60.6% | 1.88 | ~1.8 |
| down | 2560, 9728 | 32, 4 | 320 | 3 | 80.8% | 2.48 | ~2.5 |
| **lm_head** | 151936, 2560 | 64, 1 | 2374 | 18 | **99.9%** | 3.08 | **3.08** |

`pred = util * 3.35 * 0.93`. lm_head is the control: the only ~fully-occupied op is the only one at
full bandwidth. Triton block sizes are powers of two, so **132 CTAs is unreachable** — that is exactly
what CUDA buys you. Agent S1 is attacking the same thing in Triton with a masked uneven row split
(which wastes tensor-core lanes but keeps HBM traffic exact). **You should beat S1**, because you can
give CTA `c` the exact row range `c*N/132 .. (c+1)*N/132` with no masking waste at all.

### Design (from prior analysis — unproven, do not assume the gains)
- Grid exactly 132 (or 264) CTAs, ~8 warps, one contiguous uneven row range each. No split-K, so
  `_splitk_reduce_kernel` disappears and `_reduce_add_rms_kernel` (215 us / 5.1% / 72 launches)
  shrinks.
- `mma.sync.m16n8k16` bf16->fp32. Use the K-permutation: physical `k = 8q + 4s + 2h + e` for
  lane-group `q`, step `s`, register `h`, element `e`, so each thread takes **one 16 B W load and one
  16 B x load** per 32-k block, with `x` permuted identically at staging time.
- **Stage `x` in smem per k-tile, not whole.** A CTA must not re-read `x` from L2 per output row: for
  down-proj (K=9728, M=16) that would be ~780 MB of L2 traffic per op. Tile over K (e.g. BK=512 ->
  16 KB of x in smem), hold the `rows x M` fp32 accumulators in registers across the whole K loop,
  store once.
- cp.async 16 B pipelining, hand-tuned depth. PDL prologue: issue the first W stages **before**
  `griddepcontrol.wait` (weights never depend on the previous kernel's output).
  Caution: the existing Triton `PF` sweep found PF=4 best and PF>=32 **14-24% worse** because prologue
  issue cost delays the wait — so deeper is not automatically better.
- Keep the reference rounding points: fp32 accumulate, bf16 outputs.

### Gate
**>= 10% over the best Triton GEMV on the qkv shape** (13.4 us today) in a `tests/gemv_bench.py`-style
timing, at M=1/4/16. Coordinate with the orchestrator on S1's number — if S1 already reached ~3.08
TB/s there is little left here and you should pivot to helping S4 (megakernel). Then in-engine A/B
behind `ENGINE_CUGEMV=1` (default **off**), bar **>= +1.5% predicted score geomean** — microbenches
have pointed the wrong way twice on this repo, so **the in-engine A/B is the verdict**.

## 5. Ownership — strict

Yours: `engine/cudart.py`, any new `engine/cu_*.py`, and the NVRTC kernel sources.
**Not yours:** `_gemv_kernel`/`GEMM_CFG`/`TritonOps.linear` (agent S1), `_gemv_silu_kernel`/attention
(agent S2), prefill (agent S5). Wire your kernel in through a **new** env-gated branch in
`TritonOps.linear` rather than rewriting the Triton path, and tell the orchestrator exactly which
lines you touched there.

## 6. Correctness — non-negotiable

Bistable massive activation (|h| ~ 5300 at layer ~16): *any* change in bf16 rounding points or
reduction order can flip a token at B32/B64. fp32 accumulate, bf16 store at exactly the same points as
the Triton kernel you replace. Everything new goes into `Engine._selftest()`. Required before any
hand-in: `tests/bench.py --check-all` clean on the 3 public shapes **and**
`--shapes 32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`, plus `--corpus code` and
`--corpus repeat`; `worst gap <= 2.0`, `positions > 2.0: 0`, `spread <= 25%`.
Pre-existing violations you must not worsen: B32 code corpus 3.9 logits, B64 natural 3.25.

## 7. Pod setup

Create your own H100 pod (no budget ceiling; ~$3.49/h; **stop it when you hand in**). runpod MCP:
`get-capacity` / `create-pod`, GPU `NVIDIA H100 80GB HBM3`, secure cloud, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 60 GB disk, 40+ GB `/workspace`.
SSH via the **direct** `ssh root@IP -p PORT` from `get-pod` > `runtime.ports` 22 (the `ssh.runpod.io`
proxy needs a PTY). Key `~/.ssh/runpod_ed25519`. Boot ~90 s.
The pinned stack matters — NVRTC is found through torch's bundled
`nvidia/cuda_nvrtc/lib/libnvrtc.so*`:
```bash
pip install torch==2.5.1 triton==3.1.0 transformers==4.51.3 safetensors==0.5.3 tokenizers==0.21.1 huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir /workspace/model
export TRITON_CACHE_DIR=/workspace/triton_cache
```

## 8. Hand-in

Branch `krish/nvrtc-gemv` on your pod. Never push to `main`; never edit the Windows checkout (another
session is live in it). Push the branch to `origin` if your pod has GitHub creds, else
`git diff main > /workspace/w/nvrtc-gemv.patch` and paste the diff.

Report, twice: **(a) at T+60 min, the Task 0 probe diff, whatever state it is in**; (b) at the end,
branch + commit, standalone GEMV table (TB/s vs Triton per shape, M=1/4/16), in-engine pod A/B table,
`--check-all` lines, verdict keep/kill, and every negative result with numbers.
