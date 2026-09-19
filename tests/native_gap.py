"""Control: the native starter decode (HF + KV cache) vs its own teacher-forced replay on the same prompts."""
import argparse, random, sys, torch
from transformers import AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("--corpus", default="repeat"); ap.add_argument("--shape", default="4,2048,32"); ap.add_argument("--samples", type=int, default=4)
args = ap.parse_args(); B, S, n = map(int, args.shape.split(","))
M = "/workspace/model"
torch.backends.cuda.matmul.allow_tf32 = False
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).eval().cuda()
_CORPUS = None
def prompts(seed):
    global _CORPUS
    r = random.Random(seed * 7919 + B * 31 + S); out = []
    if args.corpus == "repeat":
        for _ in range(B):
            pat = [r.randrange(1000, 100000) for _ in range(r.randrange(3, 40))]
            out.append((pat * (S // len(pat) + 1))[:S])
        return out
    if _CORPUS is None:   # same windows as tests/bench.py (pydoc)
        from transformers import AutoTokenizer
        import pydoc_data.topics as t
        _CORPUS = AutoTokenizer.from_pretrained(M)(" ".join(t.topics.values()), add_special_tokens=False)["input_ids"]
    return [_CORPUS[o:o + S] for o in (r.randrange(0, len(_CORPUS) - S) for _ in range(B))]
worst, bad = 0.0, 0
for k in range(args.samples):
    ids = prompts(k + 1)
    cur = torch.tensor(ids, device="cuda"); cache = None; toks = []
    with torch.inference_mode():
        for _ in range(n):   # exactly the starter's generate loop
            o = model(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            cur = o.logits[:, -1, :].argmax(-1, keepdim=True); cache = o.past_key_values; toks.append(cur[:, 0])
        toks = torch.stack(toks, 1)                                # [B, n]
        full = torch.cat([torch.tensor(ids, device="cuda"), toks], 1)[:, :-1]
        for b in range(B):
            lg = model(full[b:b + 1], logits_to_keep=n).logits[0].float()
            gap = lg.max(-1).values - lg.gather(-1, toks[b].unsqueeze(-1)).squeeze(-1)
            worst = max(worst, gap.max().item()); bad += (gap > 2.0).sum().item()
            for p in (gap > 2.0).nonzero().flatten().tolist(): print(f"  native VIOLATION sample {k} seq {b} step {p}: gap {gap[p].item():.3f}")
print(f"NATIVE decode vs native replay: worst gap {worst:.3f}, positions > 2.0: {bad}")
