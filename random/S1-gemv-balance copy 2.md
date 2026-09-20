# S1 — `gemv-balance`: balanced-grid decode GEMV (Triton)

You are a worker agent. You start with zero context; everything you need is here or in the repo files
named below. **Read `handoffs/SHARED-CONTEXT.md` first, then `CONTEXT.md`, before writing code** — they carry the
scoreboard, rules stance, protocol, correctness bar, profiling traps and every experiment already run
with its result. Do not repeat anything in `SHARED-CONTEXT.md` > "Dead levers".

Owner: Krish. Orchestrator: a separate Opus session that merges your work and pushes. You report to
the orchestrator; you never push to `main`.

---

## 1. Mission

Dryft "decode" challenge: make Qwen3-4B decode faster on 1x H100, output byte-identical to native
greedy. Score = weighted throughput over 6 private workloads.

| rank | team | score | metricMs |
|---|---|---|---|
| 1 | Segfault | 1176.4 | 430.6 |
| 6 | **krish (us)** | **1047.3** | **483.6** |

`score * metricMs = 506522` exactly, so score is exactly proportional to 1/aggregate time. We need
**-11% aggregate** to take #1. Deadline **Sun 08:00 ET**; feature freeze 05:00 ET.

Regime weights on public tok/s (fitted, exact): B1 512->32 = **0.14**, B4 2048->32 = **0.45**,
B16 512->128 = **0.44**. **Never optimise for B1.** Decode is ~75% of weighted time.

## 2. Your thesis to prove (this is the whole job)

The decode step (B16 512->128) is 4192 us against a 2851 us bandwidth roofline: **68% of roofline**.
The gap is **not** the inner loop and **not** launch overhead (launch gaps measure 2.2%). It is
**wave quantization**: every GEMV launches `(N/BN)*SK` CTAs onto 132 SMs and none of the shapes
divides 132.

| op | shape (N,K) | cfg (BN,SK) | CTAs | waves | SM util | pred TB/s | measured |
|---|---|---|---|---|---|---|---|
| qkv | 6144, 2560 | 64, 1 | 96 | 1 | 72.7% | 2.23 | ~2.2-2.4 |
| o | 2560, 4096 | 64, 2 | 80 | 1 | 60.6% | 1.88 | ~1.8 |
| down | 2560, 9728 | 32, 4 | 320 | 3 | 80.8% | 2.48 | ~2.5 |
| **lm_head** | 151936, 2560 | 64, 1 | 2374 | 18 | **99.9%** | 3.08 | **3.08** |

`pred = util * 3.35 * 0.93`. It reproduces the measured band on the small GEMVs and nails lm_head to
three digits. **lm_head is the control**: the only op with ~100% SM utilisation is the only op at full
bandwidth, and nothing about its inner loop is special.

**Why 255 prior config sweeps missed it:** Triton block sizes are powers of two. Reachable CTA counts
for qkv are 96 / 192 / 384 — **132 is unreachable**. BN=32 gives 192 tiles over 132 SMs: 60 SMs do 2
tiles, 72 do 1, makespan = 2 units = identical to 96 CTAs of double-size tiles. The search space was
closed under the defect.

**Why `tests/gemv_persist.py` concluded "persistent GEMV is 5-10% slower":** its sweep is
`BN in (16, 32)` only — never 64, the size that actually won — so it compared a balanced grid of
*worse* tiles against an unbalanced grid of *better* tiles, and it reset and stored the accumulator
per row-tile inside the loop. **That result does not falsify this thesis. Do not let it stop you.**

## 3. What to build

A persistent, **row-balanced** Triton GEMV in `engine/fused.py`:

- `grid = (NCTA,)` with `NCTA` in {132, 264} (sweep both; 264 = 2 CTAs/SM may pipeline better).
- CTA `c` owns the **uneven** contiguous row range `lo = c*N//NCTA`, `hi = (c+1)*N//NCTA`.
  For qkv/132 that is 46 or 47 rows. The unevenness is the entire point — do not round it to a
  power of two.
- Tile shape stays `BN=64` (or sweep 32/64/128) with a row mask `rn < hi`. **Masked loads do not
  issue memory traffic**, so HBM bytes stay exact; the idle tensor-core lanes are free because the op
  is memory-bound.
- Loop order: outer over the CTA's row tiles, inner over K. Accumulator stays in registers across the
  whole K loop; **one store per row tile at the end** (the old persistent kernel reset+stored inside
  the loop — don't).
- **`SK=1` everywhere.** Perfect row balance removes the reason split-K existed. This deletes
  `_splitk_reduce_kernel` launches and shrinks `_reduce_add_rms_kernel` (215 us / 5.1% of the step /
  72 launches today). Treat the SK change as a **numerics change** (see §5), not a free win.
- Keep the existing PDL machinery: `_gdc_launch` / `_gdc_wait`, `TRIG=1` (trigger at the end of the K
  loop — `TRIG=0` measured -4%), `PF=4` L2 prefetch (`PF>=32` measured -14 to -24%). Keep `EVENK`
  mask-free loads on the K axis where the shape allows.

**Your shapes** (`GEMM_CFG` in `engine/fused.py`): `(6144,2560)` qkv, `(2560,4096)` o,
`(2560,9728)` down, `(151936,2560)` lm_head. **Not yours:** `(9728,2560)` gate_up / `_gemv_silu_kernel`
— agent S2 owns that. **Do not edit `_gemv_silu_kernel`, `SILU_CFG`, `gate_up_silu`, or any attention
kernel.**

## 4. Gates — stop if you fail one

- **Gate A1 (standalone, ~45 min):** in a `tests/gemv_bench.py`-style microbench, **>= 8% on qkv and
  on o** at M=1/4/16, and `>= 3.0 TB/s` on qkv. If you cannot reach 8%, report the measured TB/s per
  op and stop — the thesis is wrong and the orchestrator needs to know within the hour.
- **Gate A2 (in-engine):** wire behind `ENGINE_BAL=1` (default **off**), then pod A/B, 5 samples,
  `python tests/bench.py` on the 3 public shapes. Bar: **>= +1.5% predicted score geomean.**
  Microbenches have pointed the wrong way twice on this repo — **the in-engine A/B is the verdict.**
- **Gate A3:** `python tests/bench.py --check-all` clean on the 3 public shapes **and**
  `--shapes 32,512,128 64,512,128 8,1024,64 2,3000,32 16,2048,128 1,8192,64`, plus `--corpus code`
  and `--corpus repeat`. Required: `worst gap <= 2.0`, `positions > 2.0: 0`, `spread <= 25%`.

## 5. Correctness — non-negotiable

The model has a **bistable massive activation** (|h| ~ 5300 at layer ~16). *Any* change in bf16
rounding points or reduction order can flip a token at B32/B64. Rules:

1. fp32 accumulate, bf16 store at exactly the same points as the current `_gemv_kernel` /
   `_reduce_add_rms_kernel`.
2. Changing SK changes the reduction order. SK=1 is a *single* accumulation chain, so it should be no
   worse than today, but **prove it with `--check-all` at B32/B64**, not by argument.
3. Add your kernel to `Engine._selftest()` so a bad kernel degrades to slow-but-correct, never wrong.
4. Known pre-existing violations you are **not** causing and must not make worse: B32 code corpus
   gives 3.9 logits (seq 24 step 85), B64 natural 3.25 (seq 0 step 99), identical on v14 and v15.

## 6. Pod setup

Create your own H100 pod (owner set no budget ceiling; ~$3.49/h; **stop it when you hand in**).
Use the runpod MCP tools: `get-capacity` / `create-pod`, GPU `NVIDIA H100 80GB HBM3`, secure cloud,
image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 60 GB disk, 40+ GB `/workspace`.
SSH: read `runtime.ports` 22 from `get-pod` and use the **direct** `ssh root@IP -p PORT`
(the `ssh.runpod.io` proxy needs a PTY). Key `~/.ssh/runpod_ed25519`. Boot ~90 s.

```bash
pip install torch==2.5.1 triton==3.1.0 transformers==4.51.3 safetensors==0.5.3 tokenizers==0.21.1 huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir /workspace/model
export TRITON_CACHE_DIR=/workspace/triton_cache
```

Bench commands:
```bash
python tests/bench.py --model /workspace/model              # perf + correctness, 3 public shapes
python tests/bench.py --shapes 16,512,128 --no-check        # quick perf
ENGINE_PDL=0 python tests/budget.py 16,512,128              # per-kernel step budget (PDL off = clean attribution)
python tests/gemv_bench.py                                  # skinny-GEMM microbench
python tests/gemv_cfg_sweep.py qkv                          # in-engine GEMV config A/B with PDL
```

Profiling traps: with PDL on, a waiting kernel is charged its producer's time (sums read 124-131% of
the step) — attribute with `ENGINE_PDL=0`. `budget.py` replays `pos += 1`, so rewind `st.pos` or
attention reads past `cap`. Graph re-capture needs a fresh `torch.cuda.graph_pool_handle()`.

## 7. Hand-in

Branch `krish/gemv-balance` on your pod. Do **not** push to `main`. Do **not** edit the Windows
checkout — another session is live in it. Push the branch to `origin` if your pod has GitHub creds;
otherwise `git diff main > /workspace/w/gemv-balance.patch` and paste the diff in your report.

Report to the orchestrator, in this shape:
- branch + commit
- per-op standalone table: CTAs, SM util, TB/s before -> after
- in-engine pod A/B table: 5 samples, B1/B4/B16 tok/s + predicted score, before vs after, spread
- `--check-all` output line for every shape in Gate A3
- verdict: keep / kill, and the one-line reason
- anything negative, with numbers — the orchestrator appends it to `CONTEXT.md` > Findings

**Report Gate A1 the moment you have it (~45 min in), before doing anything else.** The orchestrator
is sequencing four other agents off that number.
