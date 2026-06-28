"""
Token Sampler — from logits to next tokens.

Supports:
  • Greedy decoding (argmax)
  • Temperature scaling
  • Top-k filtering
  • Top-p (nucleus) filtering
  • Repetition penalty
  • Per-sequence sampling params (different requests can have different params)
"""

from typing import List

import torch
import torch.nn.functional as F

from model.config import SamplingParams


class Sampler:
    """
    Converts model logits to next token IDs.

    Accepts a batch where each sequence may have different SamplingParams.
    Applies the sampling pipeline per-sequence:

      logits → repetition_penalty → temperature → top_k → top_p → sample
    """

    def __call__(
        self,
        logits: torch.Tensor,          # (batch, vocab_size)
        sampling_params: List[SamplingParams],
        output_token_ids: List[List[int]],   # already-generated tokens per seq (for rep penalty)
    ) -> List[int]:
        """Returns one token ID per sequence."""
        assert logits.shape[0] == len(sampling_params)
        next_tokens = []

        for i, (logit_row, sp) in enumerate(zip(logits, sampling_params)):
            token_id = self._sample_one(logit_row, sp, output_token_ids[i])
            next_tokens.append(token_id)

        return next_tokens

    def _sample_one(
        self,
        logits: torch.Tensor,   # (vocab_size,)
        sp: SamplingParams,
        prev_tokens: List[int],
    ) -> int:
        # ── Repetition penalty ─────────────────────────────────────────────────
        if sp.repetition_penalty != 1.0 and prev_tokens:
            unique = torch.tensor(list(set(prev_tokens)), dtype=torch.long, device=logits.device)
            score = logits[unique]
            # Reduce score for tokens that have appeared
            score = torch.where(score < 0, score * sp.repetition_penalty,
                                           score / sp.repetition_penalty)
            logits = logits.clone()
            logits[unique] = score

        # ── Greedy shortcut ────────────────────────────────────────────────────
        if sp.is_greedy():
            return int(logits.argmax())

        # ── Temperature ────────────────────────────────────────────────────────
        if sp.temperature != 1.0:
            logits = logits / sp.temperature

        # ── Top-k ──────────────────────────────────────────────────────────────
        if sp.top_k > 0 and sp.top_k < logits.size(-1):
            kth_value = logits.topk(sp.top_k).values[..., -1]
            logits = logits.masked_fill(logits < kth_value, float("-inf"))

        # ── Top-p (nucleus) ────────────────────────────────────────────────────
        if sp.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative prob above threshold
            # (shift right so the first token above threshold is kept)
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) > sp.top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

        # ── Sample ─────────────────────────────────────────────────────────────
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


# ── Standalone tests ───────────────────────────────────────────────────────────

def test_sampler():
    sampler = Sampler()
    vocab  = 1000
    batch  = 4

    logits = torch.randn(batch, vocab)

    params = [
        SamplingParams(temperature=0.0),           # greedy
        SamplingParams(temperature=1.0, top_k=50), # top-k
        SamplingParams(temperature=0.7, top_p=0.9),# top-p
        SamplingParams(temperature=1.0, repetition_penalty=1.3),
    ]

    prev = [[], [1, 2, 3], [], [5, 5, 5, 5]]
    tokens = sampler(logits, params, prev)

    print(f"  Sampled tokens: {tokens}")
    assert len(tokens) == batch
    assert all(0 <= t < vocab for t in tokens)

    # Greedy must be deterministic
    t1 = sampler(logits[:1], [params[0]], [[]])
    t2 = sampler(logits[:1], [params[0]], [[]])
    assert t1 == t2, "Greedy must be deterministic"
    print("  ✓ greedy deterministic")

    # Temperature=0 same as temperature=very_small
    greedy_token = sampler(logits[:1], [SamplingParams(temperature=0.0)], [[]])[0]
    argmax_token  = int(logits[0].argmax())
    assert greedy_token == argmax_token, "Greedy must equal argmax"
    print("  ✓ greedy == argmax")

    print("  All sampler tests passed.")


if __name__ == "__main__":
    test_sampler()
