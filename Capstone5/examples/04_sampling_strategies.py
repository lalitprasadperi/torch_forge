"""
Example 04 — Sampling Strategies Side by Side

Demonstrates all sampling modes using the same prompt and logits:
  • Greedy     (temperature=0.0) — deterministic, always picks argmax
  • Top-k      — sample from the k highest-probability tokens
  • Top-p      — sample from smallest set of tokens covering p probability mass
  • Temperature — sharpen (T<1) or flatten (T>1) the distribution
  • Repetition penalty — penalise tokens that have already appeared

Run:  python examples/04_sampling_strategies.py
"""
import sys; sys.path.insert(0, ".")
import torch
from sampling.sampler import Sampler
from model.config import SamplingParams


torch.manual_seed(42)
vocab   = 200
sampler = Sampler()

# Build a skewed logit distribution so sampling differences are visible
logits_base = torch.randn(vocab) * 2.0
# Give tokens 0-4 very high scores so greedy always picks token 0
logits_base[:5] += torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

strategies = [
    ("Greedy (T=0)",           SamplingParams(temperature=0.0)),
    ("Temperature 0.5",        SamplingParams(temperature=0.5)),
    ("Temperature 1.0",        SamplingParams(temperature=1.0)),
    ("Temperature 2.0",        SamplingParams(temperature=2.0)),
    ("Top-k=5",                SamplingParams(temperature=1.0, top_k=5)),
    ("Top-p=0.9",              SamplingParams(temperature=1.0, top_p=0.9)),
    ("Top-k=5 + T=0.7",        SamplingParams(temperature=0.7, top_k=5)),
    ("Rep penalty 1.5",        SamplingParams(temperature=1.0, repetition_penalty=1.5)),
]

print("=" * 60)
print("Sampling Strategies — 10 samples each from same logits")
print("=" * 60)
print(f"  Logit distribution: token 0 has highest score")
print(f"  Top-5 token IDs by logit: {logits_base.topk(5).indices.tolist()}")
print()

for name, sp in strategies:
    samples = []
    prev_tokens = list(range(20))  # for rep penalty: pretend these appeared
    for _ in range(10):
        logits_copy = logits_base.clone()
        tok = sampler._sample_one(logits_copy, sp, prev_tokens if "Rep" in name else [])
        samples.append(tok)
    unique = len(set(samples))
    print(f"  {name:<25}  samples={samples}  unique={unique}")


# ── Show what repetition penalty actually does to logits ──────────────────────
print()
print("=" * 60)
print("Repetition Penalty — Effect on Logit Scores")
print("=" * 60)
token_ids_seen = [0, 1, 2]
logits_demo = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])

print(f"  Original logits:           {logits_demo.tolist()}")
print(f"  Tokens already generated:  {token_ids_seen}")

for penalty in [1.0, 1.3, 2.0]:
    sp_demo = SamplingParams(temperature=1.0, repetition_penalty=penalty)
    logits_copy = logits_demo.clone()
    # Apply penalty manually for display
    if penalty != 1.0:
        for tid in token_ids_seen:
            score = logits_copy[tid]
            logits_copy[tid] = score / penalty if score > 0 else score * penalty
    print(f"  penalty={penalty}:  logits={[f'{v:.2f}' for v in logits_copy.tolist()]}"
          f"  (tokens 0,1,2 penalised)")
