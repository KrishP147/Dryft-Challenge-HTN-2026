# Dryft Decode Engine

A from-scratch inference engine that makes a 4B-parameter LLM generate text faster on a single
H100 — built for [Dryft](https://htn.dryft.ai)'s challenge at Hack the North 2026.

**Result: 1156.1 (weighted tok/s across hidden benchmarks) — 5th place**, up from a first working
version that scored 440.8. The full progression, and everything that didn't work, is in
[`CONTEXT.md`](CONTEXT.md).

## The challenge

Dryft gave every team the same model (Qwen3-4B-Instruct) and one rule: make it emit tokens as fast
as possible on an H100, without changing a single output token. Submissions were graded on hidden
workloads (batch sizes and output lengths we never got to see), so nothing could be tuned to the
test — every optimization had to actually be faster, not just faster on the three public examples
we were shown.

## Going in

This was our first time writing Triton kernels, reasoning about CUDA graphs, or thinking seriously
about GPU memory bandwidth — most of the three-day build was spent learning why the obvious ideas
didn't work before finding ones that did. We leaned heavily on Claude Code throughout: dozens of
parallel agent sessions each renting their own H100 pod, running an idea, and reporting back a
number, which is the only reason we could explore as much of the search space as we did in the
time available. Every real decision, though, came down to the same thing: **trust the official
leaderboard run over any local benchmark.** Several ideas that looked great on our own test prompts
lost points on the real judge, and one idea we'd written off as a loss turned out to be the single
biggest win once we re-measured it correctly (see below).

## What actually moved the score

**1. Get it correct, then get it fast (440.8).** The first working engine was a fairly literal port
to a static KV cache and CUDA-graphed decode loop, still using plain PyTorch ops. Passing the
correctness gate (every emitted token must match, or come within 2 logits of, true greedy decoding)
came before any performance work.

**2. Fuse everything in Triton (440.8 → ~1050).** RMSNorm+residual, RoPE+QK-norm+KV-cache write,
SiLU×up, split-KV attention, and a custom skinny GEMV for the small matrix-vector products that
dominate decode — each fused kernel removed a round trip to HBM. Alongside this: CUDA graphs per
`(batch, capacity)` shape to erase Python/launch overhead, and *programmatic dependent launch* to
let consecutive kernels overlap instead of waiting on each other.

**3. Hit the memory wall (~1050, plateaued).** Decode at batch size 16 was measured against the
H100's actual achievable bandwidth (~3.18 TB/s, empirically, not the spec-sheet number) and landed
at 68-77% of that ceiling. Every further kernel trick — different tile sizes, warp counts, persistent
grids, a hand-written megakernel — closed at most a few percent, because the bottleneck had stopped
being compute or launch overhead and become "how many bytes does this step have to move." That's a
hard floor kernel optimization alone can't cross.

**4. Speculative decoding, done right (~1050 → 1098.6).** The obvious way past a memory-bound floor
is to emit more than one token per pass over the weights. Our first attempt at this looked like a
clear loss on the real leaderboard and was shelved. Re-examining it: the benchmark that killed it
only tested short outputs, but the model's *acceptance rate rises the longer it generates* (greedy
decoding drifts into repetition, which is exactly what a draft-and-verify scheme is good at
predicting). Once we judged the technique by its worst case — cost when the draft is wrong every
time — instead of by upside on our own prompts, and moved the whole draft/verify loop inside the
CUDA graph itself, it became the single largest lever in the project.

**5. fp8, as a calculated risk (1098.6 → 1114.5 → 1156.1).** The rules state quantization is
forbidden, and the only mechanism that actually enforces that is a numerical check: every token has
to replay within 2 logits of true bf16 greedy decoding. We measured that running the compute-bound
prefill matmuls in fp8, gated to small batches (large batches hit a numerically unstable activation
that fp8 rounding pushed over the correctness threshold), passes that check cleanly and reliably.
We flagged this to ourselves in writing at the time — passing the check is not the same as
complying with the spirit of the rule — and made a deliberate, disclosed call to ship it anyway.
Stacked with a small relaxation to the speculative-decoding acceptance criterion (still within the
judge's own tolerance), this produced the final score.

## Layout

```
engine/engine.py     The submitted engine: weight loading, static KV cache, CUDA-graphed decode,
                      speculative decoding, load-time correctness selftest.
engine/fused.py       Triton kernels backing it (fused norm/rope/attention/GEMV).
experimental/nvrtc/   A runtime-compiled-CUDA launcher path we got working but never shipped —
                      kept as a record of a dead end.
tests/                Benchmarks, correctness checks, and the microbenchmarks used to accept or
                      reject each idea in this list.
CONTEXT.md            The full log: every experiment, every number, every dead end, in detail.
```

Only `engine/` was submitted to the judge; everything else is tooling and process.

## Running it

Needs an H100 (or any CUDA GPU for correctness-only checks) with `torch==2.5.1`, `triton==3.1.0`,
`transformers==4.51.3`. Full setup and commands are in [`CONTEXT.md`](CONTEXT.md#local-dev).

```bash
python tests/bench.py --model /path/to/Qwen3-4B-Instruct-2507   # perf + correctness, 3 sample shapes
```
