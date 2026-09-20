# S5 — `prefill`: cut TTFT (attention, non-GEMM overhead, GEMM)

You are a worker agent starting with zero context. Everything you need is here or in the repo files
named below. **Read `handoffs/SHARED-CONTEXT.md` first, then `CONTEXT.md`, before writing code.** They carry the
scoreboard, rules stance, protocol, correctness bar and profiling traps. Do not repeat anything in
`SHARED-CONTEXT.md` > "Dead levers".

Owner: Krish. Orchestrator: a separate Opus session that merges and pushes. You never push to `main`.

---

## 1. Mission

Dryft "decode": Qwen3-4B on 1x H100, output byte-identical to native greedy. Score is per-workload
`batch * out_tokens / median generation seconds`, **prefill included**, so TTFT counts.

| rank | team | score | metricMs |
|---|---|---|---|
| 1 | Segfault | 1176.4 | 430.6 |
| 6 | **krish (us)** | **1047.3** | **483.6** |

`score * metricMs = 506522` exactly: score is exactly proportional to 1/aggregate time. We need
**-11% aggregate**. Deadline **Sun 08:00 ET**, feature freeze 05:00 ET.
Regime weights on public tok/s: B1 512->32 = **0.14**, B4 2048->32 = **0.45**, B16 512->128 = **0.44**.
**Never optimise for B1.**

## 2. Why you matter more than the public numbers suggest

Prefill is ~**48% of the B4 2048->32 regime** (TTFT ~112 ms vs 32 decode steps x ~4.0 ms) and that
regime carries weight 0.45. Across the weighted total, prefill is ~**25% of the time** — the other
four agents are all fighting over the decode 75%.

And the private set is **prefill-heavy**, proved: three official draws of the *same* build scored
1041.1 / 1047.3 / **779.4** with decode TPOT identical to 0.01 ms (3.49/4.02/3.98 at B1/B4/B16) across
all three, while public-2 TTFT went 105 / 102 / **151 ms** and `metricMs` rose 34% against only a 10%
drop in public tok/s. So (a) there is no nondeterministic cliff in our code to hunt — it is platform
contention on compute-bound prefill; (b) **less time spent in prefill = less exposure to that tail**,
which is worth more than the arithmetic alone.

## 3. Where prefill time goes (B4 x 2048)

- GEMM-bound at ~740 TFLOP/s; cuBLAS tops out near 756 of a 989 theoretical peak.
- Attention ~19.6 ms of the ~112 ms TTFT.
- Non-GEMM overhead was ~19%, already cut by the token-major qkv kernel — **re-profile, don't assume
  that number still holds.**

Current design: prefill uses a Triton flash path with a token-major q layout; the **final** layer
normalises/rotates/stores only each sequence's *last* query while still caching every K and V, and
that one query attends to all prompt keys. Only the last token per sequence goes to `lm_head`.
Small prefills are CUDA-graphed (flat on the public shapes, helps short prompts).

## 4. Tasks, in priority order

1. **Prefill attention** (`engine/fused.py` flash kernel; see `tests/flash_prefill.py`,
   `tests/prof_prefill.py`). Tune `BLOCK_M`/`BLOCK_N`, `num_warps`, `num_stages`, causal block-skip,
   exp2-based softmax, q-token-major layout. **Target: 19.6 ms -> <= 15 ms.** Sweep in-engine, not
   only in the microbench.
2. **Non-GEMM overhead.** Re-profile and hunt anything not running near ~3 TB/s: add+rmsnorm at
   M=8192, silu*up, rope/cache write, any leftover copies. Note B16 512 is also M=8192, so wins here
   hit two of the three regimes.
3. **GEMM shape/algo.** Check cuBLAS is picking its best algo for the actual prefill shapes; try
   `torch.backends.cuda.matmul` knobs **only** if the output stays bit-exact.

**Already measured, do not repeat:** cuDNN SDPA for prefill (no net win once KV is expanded for GQA
and the output needs a copy — flash + token-major q is better); Triton prefill gate/up GEMM with a
fused SiLU epilogue (592 TFLOP/s vs cuBLAS 756, **-10.5% net**, `tests/prefill_gemm.py`, output was
exact); Triton flash prefill as written gave +7% on attention only; CUDA-graphed small prefills (flat
on public shapes).

**Stretch, only if 1-3 land early and you have >3 h left:** an FA3-class prefill attention (warp
specialisation + TMA) needs raw CUDA, because Triton 3.1 has neither. `experimental/nvrtc/cudart.py`
gives you NVRTC + the driver API through ctypes, already verified on an H100. Estimate several GPU
hours with a real chance of zero — clear it with the orchestrator before you start.

## 5. Rules stance — settled, do not relitigate

**If the platform accepts the submission, it is allowed.** The owner decided this explicitly. Any older note saying "no NVRTC/CUDA-string code until the organizers confirm" is **superseded**;
agent S3 is confirming platform acceptance with a real push in the first hour.
**Do NOT run `canary()`** in `experimental/nvrtc/cudart.py` — obsolete.

## 6. Gates

- **In-engine A/B:** behind an env gate, default **off**. Bar: **B4 TTFT -5% or better** in a 5-sample
  pod A/B, or **>= +1.5% predicted score geomean**. Isolated microbenches have pointed the wrong way
  twice on this repo — **the in-engine A/B is the verdict.**
- **Correctness:** `tests/bench.py --check-all` clean on the 3 public shapes **and**
  `--shapes 32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`, plus `--corpus code`
  and `--corpus repeat`. Required: `worst gap <= 2.0`, `positions > 2.0: 0`, `spread <= 25%`.
- **Latency rule:** a workload fails if TTFT **or** TPOT exceeds 1.10x native, or if the 5-sample
  spread exceeds 25%. You are working directly on TTFT — watch the spread, not just the median.

## 7. Correctness — non-negotiable

Bistable massive activation (|h| ~ 5300 at layer ~16): *any* change in bf16 rounding points or
reduction order can flip a token at B32/B64, and `tests/trace_prefill.py` showed the **prefill** path
can trigger it on its own. Keep fp32 accumulate and bf16 stores at exactly today's points; add any new
fused op to `Engine._selftest()`. Pre-existing violations you must not worsen: B32 code corpus 3.9
logits (seq 24 step 85), B64 natural 3.25 (seq 0 step 99), identical on v14 and v15.
Also: prompts with `cap` beyond the rope table used to blow up (fixed: `ROPE_LEN=32768` and growth via
a fresh graph pool, `Engine._grow_rope`) — don't regress that; test `1,8192,64`.

## 8. Ownership — strict

Yours: the prefill path — `Engine._prefill`, `Engine._capture_prefill`, `_qkv_post_prefill_kernel`,
the flash prefill kernel, and prefill-only branches in `engine/engine.py`.
**Not yours:** `_gemv_kernel`/`GEMM_CFG`/`TritonOps.linear` (agent S1), `_gemv_silu_kernel` and the
**decode** attention kernels `_attn_split_kernel`/`_attn_combine_kernel` (agent S2),
`engine/cudart.py` (agent S3), `engine/mk.py` (agent S4). If a change would touch a shared decode
kernel, route it through the orchestrator instead of editing it.

Note: the human teammate `juancavallin` is now worker **S6** (`info.md`), scoped to the decode glue
kernels, the safety net and independent verification. **Prefill is yours alone** — the earlier overlap
is resolved, so start task 1 without waiting.

## 9. Pod setup

Create your own H100 pod (no budget ceiling; ~$3.49/h; **stop it when you hand in**). runpod MCP:
`get-capacity` / `create-pod`, GPU `NVIDIA H100 80GB HBM3`, secure cloud, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 60 GB disk, 40+ GB `/workspace`.
SSH via the **direct** `ssh root@IP -p PORT` from `get-pod` > `runtime.ports` 22 (the `ssh.runpod.io`
proxy needs a PTY). Key `~/.ssh/runpod_ed25519`. Boot ~90 s.
```bash
pip install torch==2.5.1 triton==3.1.0 transformers==4.51.3 safetensors==0.5.3 tokenizers==0.21.1 huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir /workspace/model
export TRITON_CACHE_DIR=/workspace/triton_cache
```
```bash
python tests/prof.py 4 2048 32 --phase prefill      # prefill kernel table (MODEL env = model path)
python tests/bench.py --shapes 4,2048,32 --no-check
python tests/prof_prefill.py ; python tests/flash_prefill.py
python tests/spread.py                              # 5-sample spread check
```

## 10. Hand-in

Branch `krish/prefill` on your pod. Never push to `main`; never edit the Windows checkout (another
session is live in it). Push the branch to `origin` if your pod has GitHub creds, else
`git diff main > /workspace/w/prefill.patch` and paste the diff.

Report: branch + commit; prefill kernel table before -> after with TTFT at B4 2048 and B16 512;
in-engine pod A/B (5 samples, B1/B4/B16 tok/s + predicted score + **TTFT and spread**, before vs
after); `--check-all` line per shape; verdict keep/kill with a one-line reason; every negative result
with numbers. **Report your first profile within 45 minutes** so the orchestrator knows whether the
19.6 ms attention figure still holds.
