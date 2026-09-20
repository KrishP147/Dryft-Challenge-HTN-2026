# S2 — `mlp-attn-balance`: balanced gate_up GEMV + decode attention (Triton)

You are a worker agent starting with zero context. Everything you need is here or in the repo files
named below. **Read `handoffs/SHARED-CONTEXT.md` first, then `CONTEXT.md`, before writing code** — they carry the
scoreboard, rules stance, protocol, correctness bar, profiling traps and every experiment already run
with its result. Do not repeat anything in `SHARED-CONTEXT.md` > "Dead levers".

Owner: Krish. Orchestrator: a separate Opus session that merges and pushes. You never push to `main`.

---

## 1. Mission

Dryft "decode": make Qwen3-4B decode faster on 1x H100, output byte-identical to native greedy.

| rank | team | score | metricMs |
|---|---|---|---|
| 1 | Segfault | 1176.4 | 430.6 |
| 6 | **krish (us)** | **1047.3** | **483.6** |

`score * metricMs = 506522` exactly: score is exactly proportional to 1/aggregate time. Need **-11%
aggregate** for #1. Deadline **Sun 08:00 ET**, feature freeze 05:00 ET.
Regime weights on public tok/s: B1 512->32 = **0.14**, B4 2048->32 = **0.45**, B16 512->128 = **0.44**.
**Never optimise for B1.**

## 2. Your two targets

Decode step B16 512->128 = 4192 us vs a 2851 us roofline (**68%**). Your slice of it:

- **`gate_up` + SiLU (`_gemv_silu_kernel`): 1301 us = 31% of the step.** The single biggest kernel
  group after the plain GEMVs. 3.59 GB of weights at 2.76 TB/s.
- **attention split (`_attn_split_kernel`): 626 us = 14.9%.** 1.51 GB of KV at ~2.4 TB/s.

## 3. The thesis (task A: gate_up)

The roofline gap is **wave quantization**, not the inner loop and not launch overhead (launch gaps
measure 2.2%). Every GEMV launches a CTA count that does not divide 132 SMs:

| op | shape (N,K) | cfg (BN,SK) | CTAs | waves | SM util | pred TB/s | measured |
|---|---|---|---|---|---|---|---|
| qkv | 6144, 2560 | 64, 1 | 96 | 1 | 72.7% | 2.23 | ~2.2-2.4 |
| o | 2560, 4096 | 64, 2 | 80 | 1 | 60.6% | 1.88 | ~1.8 |
| **gate_up (yours)** | 9728, 2560 (x2 rows) | 32, - | **304** | 3 | **76.8%** | 2.36 | 2.76 |
| **lm_head** | 151936, 2560 | 64, 1 | 2374 | 18 | **99.9%** | 3.08 | **3.08** |

lm_head is the control: the only ~fully-occupied op is the only one at full bandwidth.
gate_up at 304 CTAs wastes 3 waves' worth of 23% idle SMs. **Note gate_up is the one op where the
model under-predicts the measurement (2.36 vs 2.76) — your first job is to measure its real
utilisation, not to trust the table.**

**Why prior sweeps missed this:** Triton block sizes are powers of two, so a CTA count of 132/264 is
unreachable for these N. `tests/gemv_persist.py` concluded "persistent GEMV is 5-10% slower", but its
sweep is `BN in (16,32)` only — never the BN that actually won — so it compared a balanced grid of
worse tiles against an unbalanced grid of better tiles. **That does not falsify the thesis.**

### Build
Persistent `_gemv_silu_kernel`: `grid = (NCTA,)`, NCTA in {132, 264}; CTA `c` owns the **uneven**
contiguous row range `c*I//NCTA .. (c+1)*I//NCTA` of the `I = 9728` interleaved gate/up rows; tile
shape stays a power of two with a row mask (masked loads issue no traffic, so HBM bytes stay exact and
the idle tensor-core lanes are free in a memory-bound op). Both accumulators (`accg`, `accu`) stay in
registers across the whole K loop; one store per row tile at the end. Keep the SiLU epilogue's
rounding **exactly** as it is today (`g -> bf16 -> f32`, `sigmoid` in f32, `* u`, store bf16).
Keep PDL: `TRIG=1` (trigger at end of K loop; `TRIG=0` measured -4%), `PF=4` (`PF>=32` measured
-14 to -24%), `EVENK` where the shape allows.

## 4. Task B: decode attention (start only after gate_up clears Gate A1)

`_attn_split_kernel` runs at ~2.4 TB/s on KV. Its grid is `B * nkv * nsplit`; at B16 with
`nsplit == 1` that is 128 CTAs of 132 (97% — already fine), and at B4 with `ATTN_TARGET=256` it is
256 of 264 (also fine). **So attention is probably NOT a wave-quantization problem** — measure the
real occupancy first and, if it is already >90%, say so and move on rather than forcing the analogy.
Then the remaining suspects are short-row ramp and the KV layout. Current settings live in env knobs
`ENGINE_ATTN_TARGET` (256), `ENGINE_ATTN_TARGET_BIG` (128), `ENGINE_ATTN_ST` (3).
Already measured and dead: FMA attention (10.7 vs 6.6 us at B1), fused last-block combine via atomic
counter (-2%), qkv_post folded into attention, attention KV prefetch before the PDL wait (noise),
128-CTA vs 256-CTA target (256 + stages=3 won in-engine, though a 128 microbench said otherwise).

Task B is worth ~100-150 us. **Task A is worth ~300 us. If you only finish one, finish A.**

## 5. Ownership — strict

Yours: `_gemv_silu_kernel`, `SILU_CFG`, `TritonOps.gate_up_silu`, `_attn_split_kernel`,
`_attn_combine_kernel`, `TritonOps.attn_decode`.
**Not yours:** `_gemv_kernel`, `GEMM_CFG`, `TritonOps.linear` / `linear_add_norm` — agent S1 owns
those and is applying the same technique there. Do not touch them; you would collide on the merge.

## 6. Gates

- **Gate A1 (standalone, ~45 min):** microbench the gate_up shape at M=1/4/16. Bar: **>= 8%** and
  **>= 3.0 TB/s**. Report this number the moment you have it — four other agents are sequenced off it.
- **Gate A2 (in-engine):** behind `ENGINE_BALMLP=1`, default **off**. Pod A/B, 5 samples,
  `python tests/bench.py`. Bar: **>= +1.5% predicted score geomean.** Isolated microbenches have
  pointed the wrong way twice on this repo — **the in-engine A/B is the verdict.**
- **Gate A3:** `tests/bench.py --check-all` clean on the 3 public shapes **and**
  `--shapes 32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`, plus `--corpus code`
  and `--corpus repeat`. Required: `worst gap <= 2.0`, `positions > 2.0: 0`, `spread <= 25%`.

## 7. Correctness — non-negotiable

Bistable massive activation (|h| ~ 5300 at layer ~16): *any* change in bf16 rounding points or
reduction order can flip a token at B32/B64. Keep fp32 accumulate and bf16 stores at exactly today's
points. Add your kernels to `Engine._selftest()` so a bad kernel degrades to slow-but-correct, never
wrong. Pre-existing violations you must not worsen: B32 code corpus 3.9 logits (seq 24 step 85), B64
natural 3.25 (seq 0 step 99), identical on v14 and v15.

## 8. Pod setup

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
python tests/bench.py --model /workspace/model
ENGINE_PDL=0 python tests/budget.py 16,512,128     # per-kernel budget; PDL off = clean attribution
python tests/attn_sweep.py ; python tests/attn_rules.py
```
Traps: with PDL on a waiting kernel is charged its producer's time (sums read 124-131% of the step) —
attribute with `ENGINE_PDL=0`. `budget.py` replays `pos += 1`, so rewind `st.pos` or attention reads
past `cap`. Graph re-capture needs a fresh `torch.cuda.graph_pool_handle()`. KV is init'd with
`zeros`, not `empty` (masked slots must be finite: `0 * NaN = NaN`).

## 9. Hand-in

Branch `krish/mlp-attn-balance` on your pod. Never push to `main`; never edit the Windows checkout
(another session is live in it). Push the branch to `origin` if your pod has GitHub creds, else
`git diff main > /workspace/w/mlp-attn-balance.patch` and paste the diff in your report.

Report: branch + commit; per-op table (CTAs, SM util, TB/s before -> after); in-engine pod A/B table
(5 samples, B1/B4/B16 tok/s + predicted score, before vs after, spread); `--check-all` line per shape;
verdict keep/kill with a one-line reason; every negative result with numbers.
