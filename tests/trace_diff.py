"""Teacher-force one token stream through torch ops and fused ops; report where they diverge."""
import os, random, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
os.environ["ENGINE_SPEC"] = "0"
import engine as E
B, S, n = 4, 2048, 32
eng = E.Engine("/workspace/model")
fused = eng.ops; torch_ops = E._TorchOps(eng)
def prompts(seed):
    r = random.Random(seed * 7919 + B * 31 + S); out = []
    for _ in range(B):
        pat = [r.randrange(1000, 100000) for _ in range(r.randrange(3, 40))]
        out.append((pat * (S // len(pat) + 1))[:S])
    return out
def run(ops, ids, forced, trace_step=None):
    eng.ops = ops; eng.states.clear()
    st = eng._state(B, S + 64, graph=False); eng._dbg = []; st.pos.fill_(S)
    tr = None
    eng._prefill(torch.tensor(ids, device="cuda"), st)
    for j in range(len(forced)):
        if j == trace_step: eng._trace = []
        st.tok.copy_(forced[j]); eng._decode_body(st)
        if j == trace_step: tr, eng._trace = eng._trace, None
    lg = eng._dbg; eng._dbg = None
    return lg, tr
for k in range(1, 5):
    ids = prompts(k)
    eng.ops = torch_ops; toks = torch.tensor([x for x in eng.generate(ids, n)], device="cuda")  # [n, B] natural torch-ops stream
    forced = [toks[j] for j in range(n - 1)]
    lt, _ = run(torch_ops, ids, forced)
    lf, _ = run(fused, ids, forced)
    # lt/lf: [prefill, step1..]  each [B, V]
    diffs = torch.stack([(a - b).abs().max(-1).values for a, b in zip(lt, lf)])  # [steps, B]
    top = diffs.flatten().topk(3)
    print(f"sample {k}: worst |dlogit| {diffs.max().item():.3f} at (step,seq) {[(int(i)//B, int(i)%B) for i in top.indices]} values {[round(v,3) for v in top.values.tolist()]}", flush=True)
    j, b = int(top.indices[0]) // B, int(top.indices[0]) % B
    if diffs.max() > 1.0 and j >= 1:
        _, trt = run(torch_ops, ids, forced, trace_step=j - 1); _, trf = run(fused, ids, forced, trace_step=j - 1)
        rel = [((f[b] - t[b]).norm() / t[b].norm()).item() for f, t in zip(trf, trt)]
        amax = [((f[b] - t[b]).abs().max()).item() for f, t in zip(trf, trt)]
        print("   per-layer rel L2 err (seq %d, decode step %d):" % (b, j), [round(x, 4) for x in rel])
        print("   per-layer max abs err:", [round(x, 3) for x in amax])
        print("   per-layer |h| max:", [round(t[b].abs().max().item(), 1) for t in trt])
        break
