# Dryft challenge: context for teammates

Make Qwen3-4B decode faster on 1x H100, **output unchanged**. Score = geomean tok/s over 6 hidden workloads. Leaderboard: htn.dryft.ai. Top team ~1115 tok/s (as of Sep 19); we got 440.8 with v1.

## Rules (short)
- Submit `engine/` only (engine.py + imported .py). Must export `Engine` with:
  - `__init__(self, model_path)`: load weights, untimed (300s budget)
  - `generate(self, input_ids: list[list[int]], max_new_tokens)`: yield one `list[int]` (1 id/seq) per step, exactly `max_new_tokens` times. Greedy. Never stop at EOS.
- Model: `Qwen/Qwen3-4B-Instruct-2507` rev `cdbee75f17c01a7cc42f958dc650907174af0554`, BF16. Platform gives `model_path`; **no downloads, no network** in engine.
- Runtime: py3.11, CUDA 12.4, torch 2.5.1, triton 3.1.0, transformers 4.51.3. Triton/Python source only. No weights/binaries/creds in `engine/`.
- Correctness: every token = baseline greedy, or within 2 logits of it (baseline replays our output). Quant/approx forbidden. Exact spec-decode OK.
- Must pass all cases: TTFT and TPOT <= 1.10x baseline; timing spread <= 25% over 5 samples; peak mem <= 90% GPU; run <= 15 min (300s/sample).
- Score per workload = batch x out_tokens / median gen sec (incl. prefill). Public workloads (never rank): B1 512->32, B4 2048->32, B16 512->128. Hidden 6 decide rank.
- Fail codes: incorrect_output, candidate_error, timeout, latency_limit, memory_limit, unstable_timing; infra_error/harness_error: retry once then ask Slack.

## Repo layout
```
engine/engine.py   Engine: weight load, static KV, CUDA-graph decode, pipelined host sync, load-time selftest
engine/fused.py    Triton kernels + TritonOps (add+rmsnorm, qk-norm+rope+cache write, silu*up, split-KV attn, skinny split-K GEMM)
tests/bench.py     GPU bench + correctness vs HF baseline (mimics platform)
tests/prof.py      torch.profiler kernel table for one generate()
tests/test_attn.py GPU: Triton attn vs SDPA
tests/test_fused.py fused ops vs torch ops (CPU via TRITON_INTERPRET=1, or GPU)
tests/test_vs_hf.py CPU: engine vs HF greedy on tiny random Qwen3 (fp32)
tests/gemv_bench.py skinny-GEMM microbench, cuBLAS vs Triton
```
Only `engine/` is submitted. Keep notes/tools/tokens outside it.

## Engine design (what's in there)
- **v1**: torch ops, static KV cache (`B x nkv x cap x hd`, cap rounded to 128), decode captured in a CUDA graph per `(B, cap)` (max 6 cached states), D2H copy + event per step so `yield` of step t-1 overlaps GPU step t. Prefill uses SDPA causal; only last token per seq goes to lm_head. Fused wqkv and gate|up weights.
- **v2** (`fused.py`): Triton fused add+rmsnorm, qk-norm+rope+KV write, silu*up. Mirrors ref bf16 rounding points.
- **v3**: split-KV Triton decode attention (+combine kernel), reads `pos` from a device tensor (graph-safe).
- **v4**: Triton split-K skinny GEMM for decode linears when M<=16; falls to `F.linear` for unlisted shapes / M>16 (`GEMM_CFG` in fused.py).
- **Safety net**: `Engine._selftest()` runs torch ops vs fused ops on real weights at load (teacher-forced); uses fused only if max logit diff <= 0.5, else falls back to `_TorchOps`. Graph capture failure falls back to eager.
- Ragged prompt lengths -> per-sequence fallback (slow; hidden workloads are presumably equal length).

## Results
| ver | official tok/s | notes |
|---|---|---|
| v1 | **440.8** (#32) | B1 113.3, B4 233.2, B16 1464.1. TPOT 8.4-11.4 ms vs baseline 24-28 ms. TTFT B4 197 ms (~baseline). |
| v2-v4 | fill in | on `main` (HEAD 8ff6a6c = v4); add official numbers here |

Why: ~1800 launches/step unfused; prefill elementwise fp32-heavy. Next ideas: fewer launches, faster prefill (TTFT is a gate and counts in TPS), GEMM tuning, exact speculative decoding.

## Local dev

### Need
An H100 from **RunPod** (create a pod, ssh in, clone this repo). Any CUDA GPU works for correctness; perf only meaningful on H100. Model in `/workspace/model` (dev only; pinned rev):
```bash
pip install torch==2.5.1 triton==3.1.0 transformers==4.51.3 safetensors==0.5.3 tokenizers==0.21.1 huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir /workspace/model
```

### Run
```bash
# perf + correctness, 3 public shapes, 5 samples, prints geomean(public)
python tests/bench.py --model /workspace/model
python tests/bench.py --shapes 1,512,32 --no-check       # quick perf only
# other shapes: "B,S,n" e.g. 8,1024,64

python tests/prof.py 16 512 128                           # kernel profile (MODEL env = model path)
python tests/test_attn.py                                 # GPU attn vs SDPA
python tests/test_fused.py                                # GPU, or CPU w/ TRITON_INTERPRET=1 (default)
python tests/test_vs_hf.py                                # CPU, no model needed
python tests/gemv_bench.py                                # GEMM microbench
```
Bench correctness line: `worst gap` must stay <= 2.0 and `positions > 2.0: 0`. Watch `spread` <= 25%, and TPOT/TTFT vs baseline.

No GPU? Logic-check Triton on CPU: `triton-windows`/triton + `TRITON_INTERPRET=1` (fp32, slow) via `tests/test_fused.py` and `tests/test_vs_hf.py`. Kernel perf and bf16 rounding still need the H100.

## Submitting
1. GitHub App connected to this repo (Repositories page, engine folder = `engine`, auto-run on).
2. **Push to `main` = official run** (~6 min queue + ~13 min run, counts toward leaderboard). Other branches do NOT run.
3. Workflow: work on `krish/<name>` branch (prefix your own), bench on GPU, merge/push to `main` only when bench passes.
4. Follow run on submission page (logs, per-case result, run ID). Don't leave a bad push on main: it burns a queue slot.

### API / CLI
Token: team page -> API tokens. One team token, shared among teammates (share via DM, not git). Keep in `~/.dryft_token` (never commit, never in `engine/`).
`dryft.exe` CLI returned 403 for us, so use curl with a browser UA:
```bash
H=(-A "Mozilla/5.0" -H "Authorization: Bearer $(cat ~/.dryft_token)")
B=https://htn.dryft.ai/api/v1
curl "${H[@]}" $B/challenges                       # benchmark + public workloads
curl "${H[@]}" $B/submissions                      # list (id != "#7" in title)
curl "${H[@]}" $B/runs/<FULL_RUN_UUID>             # progress + results
curl "${H[@]}" "$B/runs/<RUN_ID>/logs?after=-1&limit=200"   # pass nextAfter as after
curl "${H[@]}" -X POST -H "Idempotency-Key: $(uuidgen)" -H "Content-Type: application/json" \
     -d '{"mode":"official"}' $B/submissions/<SUBMISSION_ID>/runs   # rerun existing submission
curl "${H[@]}" -X POST -H "Content-Type: application/json" -d '{}' $B/runs/<RUN_ID>/cancel
```
`./bin/dryft validate engine` (starter CLI) does the server's archive lint locally; lint failure = 400 `lint_failed` with file+line, no run created.

## Gotchas
- KV cache init with `zeros`, not `empty`: masked slots must be finite (0 * NaN = NaN).
- Any tensor read inside the CUDA graph must be a persistent buffer updated in place (`st.tok`, `st.pos`). Allocs inside capture are fine (shared graph pool), host-side Python values are baked in at capture.
- Harness warms the same shape before timing; engine load/warmup is untimed but capped at 300s. Don't pre-warm every shape in `__init__` (dropped in cebcf22).
- A fresh engine per workload: first-call graph capture cost lands in warmup, not timing.
- Selftest tolerance/fallback exists so a broken Triton kernel degrades to slow-but-correct instead of a failed run. Keep it when adding new fused ops (add them to the selftest path).
- Memory cap 90% of 80 GB: each `(B, cap)` state holds a full KV cache; `MAX_STATES=6`.
- Team-wide: any member can push, connect repos, revoke tokens, remove members. Share invite code/token only inside the team.
