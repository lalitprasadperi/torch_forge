"""
torch.compile Speedup Benchmark

Measures speedup across the full pipeline:
  Eager → torch.compile → torch.compile + mode='max-autotune'

Tested on:
  • MLP forward pass
  • Transformer block forward pass
  • Full GPT-style training step (forward + backward + optimizer)

Run:
  python benchmarks/compile_speedup.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def bmark(fn, *args, n_warmup=20, n_iter=100):
    """Warmup then time n_iter calls. Returns ms/iter."""
    for _ in range(n_warmup):
        fn(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Deep MLP
# ─────────────────────────────────────────────────────────────────────────────

class DeepMLP(nn.Module):
    def __init__(self, d=1024, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: Full GPT-style (for training step benchmark)
# ─────────────────────────────────────────────────────────────────────────────

class MiniGPT(nn.Module):
    def __init__(self, vocab=50257, d=512, n_heads=8, n_layers=6, ctx=256):
        super().__init__()
        self.emb    = nn.Embedding(vocab, d)
        self.pos    = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads, d*4) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d)
        self.head   = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight   # weight tying

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def benchmark_model(model_name, model, x, label_x=None, n_iter=100):
    """Benchmark eager, compile, max-autotune. Returns dict."""
    model = model.to(DEVICE).eval()
    if isinstance(x, torch.Tensor):
        x = x.to(DEVICE)

    results = {}

    # Eager
    with torch.no_grad():
        results["eager"] = bmark(model, x, n_iter=n_iter)

    # torch.compile (default = inductor)
    compiled = torch.compile(model)
    with torch.no_grad():
        results["compile"] = bmark(compiled, x, n_iter=n_iter)

    # torch.compile max-autotune (slower compile, faster runtime)
    compiled_max = torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        results["max-autotune"] = bmark(compiled_max, x, n_iter=n_iter)

    return results


def benchmark_training_step(n_iter=50):
    """Full training step: forward + backward + adam step."""
    model = MiniGPT().to(DEVICE).train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, T  = 8, 256
    idx   = torch.randint(0, 50257, (B, T), device=DEVICE)
    tgt   = torch.randint(0, 50257, (B, T), device=DEVICE)

    def step(model):
        optim.zero_grad(set_to_none=True)
        logits = model(idx)
        loss   = F.cross_entropy(logits.view(-1, 50257), tgt.view(-1))
        loss.backward()
        optim.step()

    # Eager
    t_eager = bmark(step, model, n_warmup=5, n_iter=n_iter)

    # Compiled (note: compile the model, not the step — to keep optimizer eager)
    model_c = torch.compile(model)
    t_compiled = bmark(step, model_c, n_warmup=5, n_iter=n_iter)

    return {"eager": t_eager, "compile": t_compiled}


def print_results(name, results):
    print(f"\n  {name}:")
    baseline = results.get("eager", list(results.values())[0])
    for method, ms in results.items():
        speedup = baseline / ms
        bar = "█" * int(speedup * 10)
        print(f"    {method:<16}  {ms:>7.2f} ms/iter  {speedup:.2f}×  {bar}")


def main():
    print(f"\n{'═'*60}")
    print(f"  torch.compile Speedup Benchmark  (device={DEVICE})")
    print(f"{'═'*60}")

    # MLP
    mlp = DeepMLP(d=512, n_layers=8)
    x   = torch.randn(64, 512)
    print_results("DeepMLP (inference)", benchmark_model("mlp", mlp, x))

    # Transformer block
    tf  = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
    x2  = torch.randn(32, 256, 512)   # (B, T, d)
    print_results("TransformerBlock (inference)", benchmark_model("tf", tf, x2))

    # Training step
    print("\n  MiniGPT Training Step (fwd+bwd+optim):")
    tr = benchmark_training_step()
    print(f"    eager:   {tr['eager']:.2f} ms/step")
    print(f"    compile: {tr['compile']:.2f} ms/step  "
          f"({tr['eager']/tr['compile']:.2f}×)")

    print(f"\n{'═'*60}")
    print("  Summary:")
    print("  • torch.compile gives 1.5–3× on inference for elementwise-heavy models")
    print("  • Transformer blocks get FlashAttention from SDPA (inside MHA)")
    print("  • Training step speedup depends on optimizer overhead ratio")
    print("  • max-autotune gives more speedup at the cost of longer compile time")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
