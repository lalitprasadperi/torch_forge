"""
Character-Level Tokenizer

For training GPT on small text datasets (Shakespeare, etc.) we use a simple
character-level tokenizer: each character is one token.

Vocabulary = set of unique characters in the training text.
Typical vocab_size ≈ 65 for Shakespeare.

WHY CHARACTER-LEVEL?
  • No external dependencies (no tiktoken / sentencepiece)
  • Perfect for small demos — vocab fits in memory easily
  • The model must learn to spell words, not just combine word-pieces
  • Educational: makes it easy to visualise what the model learns

VS BYTE-PAIR ENCODING (BPE, used in GPT-2/3/4):
  • BPE merges frequent character pairs iteratively → sub-word tokens
  • Vocab size ~50k for GPT-2, reducing sequence length vs char-level
  • More efficient for long texts; harder to implement
  • For this capstone, char-level is sufficient to get readable output
"""

import json
from pathlib import Path


class CharTokenizer:
    """
    Minimal character-level tokenizer.
    Saves/loads vocabulary from a JSON file.
    """

    def __init__(self, vocab: list[str] | None = None):
        if vocab is not None:
            self._build(vocab)

    def _build(self, chars: list[str]):
        self.chars     = sorted(set(chars))
        self.vocab_size = len(self.chars)
        self.stoi      = {c: i for i, c in enumerate(self.chars)}
        self.itos      = {i: c for c, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        tok = cls(list(text))
        return tok

    def encode(self, text: str) -> list[int]:
        """String → list of integer token ids."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        """List of token ids → string."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.itos.get(i, "?") for i in ids)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({
            "chars": self.chars,
            "vocab_size": self.vocab_size,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(data["chars"])

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"
