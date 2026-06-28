"""
Text Dataset for Language Model Training

HOW LM TRAINING WORKS:
  We take a long text and chop it into overlapping windows of length max_len.
  Input:  tokens[i : i+max_len]
  Target: tokens[i+1 : i+max_len+1]   ← shifted by 1

  The model learns: given these T tokens, predict the next T tokens.
  This is called "next token prediction" or "causal language modelling".

  Example (max_len=5):
    text:    "hello world"
    tokens:  [7, 3, 11, 11, 14, 1, 22, 14, 17, 11, 9]
    sample:  input=[7,3,11,11,14],  target=[3,11,11,14,1]

STRIDE:
  If stride == max_len, windows are non-overlapping (faster, less data).
  If stride < max_len, windows overlap (more samples, sees more context).
  For small datasets like Shakespeare, stride = 1 maximises training data.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from .tokenizer import CharTokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        tokens:  torch.Tensor,   # 1D tensor of all token ids
        max_len: int = 256,
        stride:  int = None,     # None → stride = max_len (non-overlapping)
    ):
        self.tokens  = tokens
        self.max_len = max_len
        self.stride  = stride or max_len
        # Valid starting positions: need max_len+1 tokens ahead
        self.starts  = list(range(0, len(tokens) - max_len, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.starts[idx]
        x = self.tokens[i     : i + self.max_len]      # input
        y = self.tokens[i + 1 : i + self.max_len + 1]  # target (shifted by 1)
        return x, y


def load_shakespeare(data_dir: str = "./data", max_len: int = 256) -> tuple:
    """
    Download and prepare the tiny Shakespeare dataset.
    Returns (train_dataset, val_dataset, tokenizer).
    """
    import urllib.request

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    txt_path = data_dir / "shakespeare.txt"
    tok_path = data_dir / "shakespeare_tokenizer.json"

    # Download if not present
    if not txt_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"  Downloading tiny Shakespeare ({url})...")
        urllib.request.urlretrieve(url, txt_path)
        print(f"  Saved to {txt_path} ({txt_path.stat().st_size/1024:.0f} KB)")

    text = txt_path.read_text()
    print(f"  Text length : {len(text):,} characters")

    # Build / load tokenizer
    if tok_path.exists():
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = CharTokenizer.from_text(text)
        tokenizer.save(tok_path)
    print(f"  Vocab size  : {tokenizer.vocab_size}")

    # Tokenise full text
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"  Tokens      : {len(tokens):,}")

    # 90/10 train/val split
    split  = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]

    train_ds = TextDataset(train_tokens, max_len=max_len, stride=max_len)
    val_ds   = TextDataset(val_tokens,   max_len=max_len, stride=max_len)

    print(f"  Train samples: {len(train_ds):,}   Val samples: {len(val_ds):,}")
    return train_ds, val_ds, tokenizer
