"""
Text Generation — Sampling Strategies

After training, the model outputs logits (B, T, vocab_size).
To generate text we sample from these logits token by token.

SAMPLING STRATEGIES:
────────────────────

Greedy (temperature=0):
  Always pick the most probable token.
  Deterministic. Often repetitive / boring.

Temperature sampling:
  logits_scaled = logits / temperature
  probs = softmax(logits_scaled)
  next_token = sample(probs)

  temperature > 1: flatter distribution → more random
  temperature < 1: sharper distribution → more confident/repetitive
  temperature = 1: original distribution
  temperature → 0: greedy

Top-k sampling:
  Zero out all tokens except the top-k highest logits, then sample.
  Prevents the model from sampling very unlikely tokens.
  k=50 is a common default.

Top-p (nucleus) sampling:
  Find the smallest set of tokens whose cumulative probability ≥ p.
  Zero out everything outside that set, then sample.
  p=0.9 means "sample from the 90% most likely tokens".
  More adaptive than top-k: uses fewer tokens when the model is confident.

TYPICAL SETTINGS:
  Creative text:    temperature=1.0, top_p=0.95
  Chat/assistant:   temperature=0.7, top_k=50
  Code generation:  temperature=0.2, top_k=50
  Deterministic:    temperature=0,   greedy
"""

import torch
import torch.nn.functional as F
from .kv_cache import KVCache


@torch.inference_mode()
def generate(
    model,
    prompt_tokens:   torch.Tensor,    # (1, T_prompt) — single sequence
    max_new_tokens:  int = 200,
    temperature:     float = 1.0,
    top_k:           int   = 50,
    top_p:           float = 1.0,
    eos_token_id:    int   = None,
) -> torch.Tensor:
    """
    Autoregressive text generation with KV cache.

    Returns:
        tensor of shape (1, T_prompt + n_generated) with all tokens
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = prompt_tokens.to(device)

    cache  = KVCache()
    B, T   = tokens.shape

    # Prefill: process the prompt in one forward pass to populate KV cache
    logits, _ = model(tokens, kv_cache=cache.cache)
    # logits[:, -1] = distribution over next token after the prompt

    generated = []

    for step in range(max_new_tokens):
        # Sample next token from logits at last position
        next_logits = logits[:, -1, :]           # (B, vocab_size)
        next_token  = _sample(next_logits, temperature, top_k, top_p)

        generated.append(next_token)

        # Stop if EOS token generated
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # Feed single new token; KV cache handles the rest
        logits, _ = model(next_token, kv_cache=cache.cache)

    cache.clear()

    if not generated:
        return tokens

    return torch.cat([tokens, torch.cat(generated, dim=-1)], dim=-1)


def _sample(
    logits:      torch.Tensor,   # (1, vocab_size)
    temperature: float,
    top_k:       int,
    top_p:       float,
) -> torch.Tensor:
    """Sample one token from logits using temperature + top-k + top-p."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)   # greedy

    logits = logits / temperature

    # Top-k: zero out everything except top k
    if top_k > 0 and top_k < logits.size(-1):
        kth_val = logits.topk(top_k, dim=-1).values[:, -1, None]
        logits  = logits.masked_fill(logits < kth_val, -float("inf"))

    # Top-p (nucleus): keep smallest set with cumulative prob ≥ p
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens where cumulative prob ABOVE p (after the first one)
        remove = cum_probs - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits[remove] = -float("inf")
        # Scatter back to original order
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)   # (1, 1)
