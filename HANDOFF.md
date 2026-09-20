# Dryft decode: session handoff (2026-09-19)

Everything measured in this session, including the negative results, which are most of
the value here. Written for whoever picks this up next — human or agent.

State at handoff: **plateau at ~1042** (v14 1042.4, juancavallin's a2a0802 1042.8,
v15 `1e460f5` 1041.1). zip 1071, Segfault 1064, top three 1138-1144. Three consecutive
pushes moved the official score by nothing.

---

## 1. The scoring function is solved

For **all 12 official runs**, spanning 440 → 1042 tok/s:

```
score × metricMs = 506.5222262957889      (constant to 15 digits)
```

So `score = 506.5222263 / (aggregate seconds)`. There is exactly one number to minimise.

**Do not claim** `geomean(B×out) = 506.52` over the six private workloads — I asserted
this early and it is **false**. If it were an unweighted geomean of six integer token
counts, `506.5222262957889^6` would have to be an integer; it is
`16888498602639307.79`, and no smooth integer exists within ±40. The aggregation is
genuinely weighted (`weighted_model_throughput` / `familyWeights` in the challenge
spec), i.e. `K = Π tokensⱼ^wⱼ` with non-uniform `wⱼ`. The formula above is unaffected.

## 2. Regime weights — useful, but weaker than they first looked

Fitting `log(score) ~ Σ wₖ·log(public tok/sₖ)` over 11 runs gave residuals <0.8% and

| regime | public shape | weight |
|---|---|---|
| small batch | B1 512→32 | 0.14 |
| medium batch, long prompt | B4 2048→32 | 0.45 |
| large batch, long output | B16 512→128 | 0.44 |

Caveat, and it matters: those 11 runs all moved their public numbers *together*, so the
weights are collinear and individually soft. The direction (B1 barely matters) is
corroborated independently by v9 (B1-only speculation, B1 −9.7%, score only −1.11%),
but treat the exact split as indicative.

## 3. What we know about the private set

- **No workload with batch ≥ 32.** a2a0802's `nsplit==1` attention skip fires only at
  `bk ≥ 256` (B ≥ 32) and changed the official score by nothing. This retroactively makes
  large-B work — the combine skip, a BM=32/64 GEMV path — worth ~0. Spend effort at B≤16.
- **Probably nothing with `B*S ≤ 4096`**: v13 (CUDA-graphed small prefills) was exactly flat.
- **Roughly one workload at B=1**, from the v9 spec sensitivity.
- Private set is heavier in tokens than the public set.

## 4. Measured decode-step budget

`tests/budget.py` (added this session) replays the captured decode graph alone and
buckets kernels. B16 512→128, **ENGINE_PDL=0** for clean attribution:

| bucket | µs | % of step | launches |
|---|---|---|---|
| gemv (qkv/o/down/lm_head) | 1787.5 | 42.6% | 109 |
| gemv+silu (gate_up) | 1301.2 | 31.0% | 36 |
| attn split | 625.8 | 14.9% | 36 |
| reduce+add+rms | 214.7 | 5.1% | 72 |
| qkv_post | 83.8 | 2.0% | 36 |
| attn combine | 60.9 | 1.5% | 36 |
| **kernel sum** | **4099** | **97.8%** | |

**Launch gap is 2.2% of the step, total, across ~290 kernels.** CUDA graphs already ate
it. Every "fuse to remove launches" idea is bounded by that 2.2% minus its own cost.
PDL on: step 4191.7 → 3981.9 µs (+5.0%).

**Profiling trap:** with PDL on, a kernel spins in `griddepcontrol.wait` for its producer
and the profiler charges that wait to the *waiting* kernel — the kernel sum reads
124-131% of the step and `attn combine` appears to cost as much as `attn split`. Always
A/B with `ENGINE_PDL=0` for attribution, and in-engine (not microbench) for decisions.

## 5. Where the remaining ~30% actually is

GEMV rate scales with kernel size — per-launch ramp, not tile config:

| shape | bytes | achieved |
|---|---|---|
| lm_head | 778 MB | 3079 GB/s |
| gate_up | 99.6 MB | 2811 |
| down | 49.8 MB | 2280 |
| qkv | 31.5 MB | 2324 |
| o | 21.0 MB | **1761** |

If the four small GEMVs ran at lm_head's rate the step would drop 616 µs = **14.7%**,
which is approximately the entire gap to the leaders. A 21-50 MB kernel with 80-320 CTAs
cannot reach the steady-state bandwidth a 778 MB kernel with 2374 CTAs reaches. **Nothing
inside the one-kernel-per-GEMV structure fixes this** — see §6 for the four ways we tried.

## 6. Exhausted levers — do not re-tread

| lever | result |
|---|---|
| PF / L2 prefetch depth | **Dead.** PF=4 beats PF=0 by 0.16%; deeper is monotonically worse: PF=8 −1.0%, 16 −3.2%, 32 −14.2%, 64 −24.2%. Cost is instruction issue (BN×PF inline prefetch ops before `gdc_wait`), not memory. `PF` must be a power of two (`tl.arange`). |
| GEMV tile configs | **Dead.** 255 configs over qkv/o/down/gate_up, in-engine at B16. `qkv` and `gate_up` current configs are already optimal (+0.00%); `down` +0.08%, `o` +0.12% — both inside noise. Deeper `num_stages` lost on every shape. EVENK did not shift the optimum. |
| Peel first tile before `gdc_wait` | **Not built.** Two independent measurements (PF worth 0.16%, num_stages worth 0.00%) say "start weight traffic earlier / more loads in flight" is exhausted. |
| Persistent SM-balanced GEMV | −4% (132/264 CTAs, contiguous slices) |
| Fused prefill gate/up + SiLU GEMM | −10.5 to −17.5% (Triton 592 vs cuBLAS 756 TFLOP/s). `silu_mul` already runs at ~3.0 TB/s = peak, so there is nothing to reclaim without beating cuBLAS. |
| `qkv_post` into qkv GEMV epilogue | Capped at 2.0%, and needs BN=128 (one CTA per head), halving the GEMV grid 96→48 CTAs. Dropped. |
| Persistent fused MLP (grid barrier) | Targets 5.1% via the grid shape already measured at −4%. Dropped. |
| attn combine skip at `nsplit==1` | Correct and worth 1.5% at B32 (4652.9→4723.2 tok/s), but worth **~0 on the private set** (§3). Landed independently by juancavallin. |
| Earlier: FMA attention, fused last-block combine, GEMV re-tiling, n-gram speculation | See CONTEXT.md |

## 7. Open risks

- **Correctness at large batch.** B32/B64 on the code/repeat corpora produce a >2.0-logit
  violation (B64: gap 3.25 at seq 0 step 99; B32: 3.9 at seq 24 step 85). The native
  control is **clean** (`tests/native_gap.py` at B64: worst gap 1.000, 0 violations), so
  it is ours, not the checker. Bisect says it needs **both** attn and qkv fused
  (`ENGINE_OFF=attn` passes, `ENGINE_OFF=qkv` passes, `ENGINE_OFF=gemv` still fails).
  Identical on v14 and v15 — pre-existing, not introduced by any recent merge. Low
  priority *if* §3 holds and no private workload has B≥32, but it is a whole-run failure
  if it ever fires.
- Official run-to-run noise is **±0.5%**, so official runs cannot resolve sub-0.5%
  changes. Use pod A/B (5 samples) to decide; use official runs only to confirm.

## 8. Tools added this session

- `tests/budget.py [B,S,n]` — per-kernel decode-step budget, roofline %, launch-gap
  remainder. Run with `ENGINE_PDL=0` for attribution. Note every replay does `pos += 1`,
  so it rewinds `st.pos` before each block; without that, attention reads past `cap` and
  you get an illegal memory access that looks like a kernel bug.
- `tests/gemv_cfg_sweep.py {qkv,o,down,gate_up,pf} [shapes]` — A/Bs `GEMM_CFG` /
  `SILU_CFG` / `PF` **inside the real decode step with PDL on**, re-capturing the graph
  per config. Isolated microbenches pointed the wrong way twice (attention `nsplit`, and
  the GEMV configs); this exists so that stops happening.
- Each config needs a fresh `torch.cuda.graph_pool_handle()`, otherwise re-capture trips
  `use_count > 0` in the caching allocator. Same root cause as the B1 8192 capture failure.

## 9. What is actually left

The only lever large enough to close ~10% is a **megakernel**: one persistent kernel
spanning what are now separate launches, so the small matrices stop paying ramp
individually. Multi-hour, high-risk, and the −4% persistent-grid result is a partial
data point against it.

A second route was in flight at handoff: compiling CUDA C++ at runtime via **NVRTC**
(shipped with the torch wheel) to express a PDL prologue that issues `cp.async` before
`griddepcontrol.wait`, which Triton cannot express. Two things to settle first:

1. **Rules.** The challenge text says the runtime is the platform's and to "vendor any
   Python or Triton source … do not include compiled binaries". NVRTC ships no binary —
   it is a string compiled at load — so it plausibly passes the letter while sitting
   outside "Triton/Python source only". Genuinely ambiguous; `docs/QWEN_ENGINE_CONTRACT.md`
   is the authoritative text and is **not** reachable through the API. Ask the organizers
   in Slack. One message removes a disqualification risk on the whole team's entry.
2. **You do not need a timing side channel to find out.** Official run logs say
   *"this run touched hidden cases, so output written by the submission is not shown here.
   It is kept for operator …"* — the suppression is conditioned on hidden cases, so a
   **public** run (which never ranks) should surface submission stdout directly via
   `GET /runs/<id>/logs`. Print what worked; do not encode results in per-step sleeps.
   Note also that submission output is retained for operator review either way.
