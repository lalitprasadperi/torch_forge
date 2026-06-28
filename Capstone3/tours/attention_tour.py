"""
attention_tour.py — Understand Attention from First Principles

Run with:
  python tours/attention_tour.py

Lessons:
  1 — The Problem: Permutation Invariance
  2 — Dot-Product Similarity
  3 — Scaled Dot-Product Attention (step by step)
  4 — Causal Masking
  5 — Multi-Head Attention
  6 — Visualise Attention Patterns
  7 — Complexity: Why T² Is a Problem
"""

import sys
import math
import torch
import torch.nn.functional as F

COLS = 68

def lesson(n, title):
    print()
    print("╔" + "═" * (COLS - 2) + "╗")
    print(f"║  LESSON {n}: {title:<{COLS - 14}}║")
    print("╚" + "═" * (COLS - 2) + "╝")

def explain(*lines):
    for l in lines:
        print(f"  {l}")
    print()

def show(label, value):
    print(f"  {'▶ ' + label:<32} {value}")

def heatmap(mat, row_labels=None, col_labels=None, width=6):
    """ASCII heatmap for small attention matrices."""
    SHADES = " ░▒▓█"
    if hasattr(mat, "detach"):
        mat = mat.detach().float()
    mn, mx = mat.min().item(), mat.max().item()
    if col_labels:
        print("       " + "".join(f"{l:^{width}}" for l in col_labels))
    for i, row in enumerate(mat):
        label = f"  {row_labels[i]:>4} " if row_labels else "       "
        cells = ""
        for v in row:
            norm = (v.item() - mn) / (mx - mn + 1e-9)
            shade = SHADES[min(int(norm * (len(SHADES)-1)), len(SHADES)-1)]
            cells += shade * width
        print(label + cells)
    print()


def lesson_1_permutation_invariance():
    lesson(1, "The Problem: Permutation Invariance")
    explain(
        "Consider a simple network that processes a sentence token by token:",
        "  MLP(token_0), MLP(token_1), ..., MLP(token_T)",
        "",
        "Each token is processed independently — there's NO interaction between",
        "tokens. The model can't know that 'bank' in 'river bank' differs from",
        "'bank' in 'savings bank' because those surrounding words are ignored.",
        "",
        "Also: an MLP applied to each token is permutation equivariant:",
        "  [A, B, C] → [f(A), f(B), f(C)] = same as processing [C, A, B] in order",
        "",
        "We need tokens to COMMUNICATE with each other.",
        "Attention is the mechanism that lets token i ask:",
        "  'Given my content, which other tokens should I gather info from?'",
    )

    # Demonstrate: same token, different context → should produce different output
    print("  Example: 'bank' in two contexts")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │ Context 1: 'the river bank was muddy'                       │")
    print("  │ Context 2: 'she went to the savings bank'                   │")
    print("  │                                                             │")
    print("  │ The word 'bank' is identical in both. An MLP produces the   │")
    print("  │ same representation for both — it can't tell them apart.    │")
    print("  │ Attention lets 'bank' look at 'river' vs 'savings' and      │")
    print("  │ produce context-sensitive representations.                   │")
    print("  └─────────────────────────────────────────────────────────────┘")


def lesson_2_dot_product_similarity():
    lesson(2, "Dot-Product Similarity")
    explain(
        "Before attention, we need a way to measure similarity between vectors.",
        "The dot product is the key operation:",
        "",
        "  a · b = Σ aᵢ bᵢ = |a| |b| cos(θ)",
        "",
        "  High dot product → vectors point in similar directions → high similarity",
        "  Dot product = 0 → vectors are perpendicular → unrelated",
        "  Negative → vectors point in opposite directions",
    )

    torch.manual_seed(42)
    d = 8

    # Semantic similarity example
    river = torch.tensor([1.0, 0.0, 0.5, 0.2, -0.3, 0.1, 0.8, 0.0])  # "river"
    bank_water = river + 0.1 * torch.randn(d)   # "bank" in water context
    bank_money = torch.tensor([-0.5, 1.0, -0.2, 0.8, 0.4, -0.1, -0.6, 0.9])  # "bank" in money context

    show("river · bank(water)  =", f"{river @ bank_water:.3f}  (should be HIGH)")
    show("river · bank(money)  =", f"{river @ bank_money:.3f}  (should be LOW)")

    explain("",
            "  This is what attention computes: the query ('what am I looking for?')",
            "  is dot-producted against all keys ('what do other tokens offer?').",
            "  High dot product → attend more to that token.")


def lesson_3_scaled_attention_step_by_step():
    lesson(3, "Scaled Dot-Product Attention — Step by Step")
    explain(
        "Let's implement attention manually for a tiny 3-token sequence:",
        "  tokens: ['The', 'cat', 'sat']",
        "  d_model=4, no heads (single head), d_k=4",
    )

    torch.manual_seed(0)
    d_k = 4
    T   = 3
    labels = ["The", "cat", "sat"]

    # Random embeddings (stand-in for token embeddings)
    X = torch.randn(T, d_k)

    # Learned projection matrices
    W_Q = torch.randn(d_k, d_k) * 0.5
    W_K = torch.randn(d_k, d_k) * 0.5
    W_V = torch.randn(d_k, d_k) * 0.5

    Q = X @ W_Q   # (T, d_k)
    K = X @ W_K
    V = X @ W_V

    explain("  Step 1: Project input X → Q, K, V")
    show("X shape  :", X.shape)
    show("Q shape  :", Q.shape)
    show("K shape  :", K.shape)
    show("V shape  :", V.shape)

    scores = Q @ K.T                          # (T, T)
    explain("", "  Step 2: Similarity scores = Q @ K^T")
    show("Scores (unscaled):", "")
    heatmap(scores, labels, labels)

    scaled = scores / math.sqrt(d_k)
    explain("  Step 3: Scale by 1/sqrt(d_k) to prevent softmax saturation")
    show(f"1/sqrt({d_k}) = {1/math.sqrt(d_k):.3f}", "")
    show("Scores (scaled):", "")
    heatmap(scaled, labels, labels)

    # Causal mask
    mask   = torch.tril(torch.ones(T, T, dtype=torch.bool))
    masked = scaled.masked_fill(~mask, -1e9)
    explain("  Step 4: Causal mask — future tokens → -inf")
    show("After masking:", "")
    heatmap(masked, labels, labels)

    weights = F.softmax(masked, dim=-1)
    explain("  Step 5: Softmax over keys → attention weights (sum to 1)")
    print("  Attention weights:")
    heatmap(weights, labels, labels)
    for i, row in enumerate(weights):
        print(f"    {labels[i]:<6} attends to: " +
              ", ".join(f"{labels[j]}={w:.2f}" for j, w in enumerate(row) if w > 0.01))

    output = weights @ V
    explain("", "  Step 6: Weighted sum of values → output")
    show("Output shape:", output.shape)
    show("Output[0] (The):", output[0].round(decimals=3))
    show("Output[1] (cat):", output[1].round(decimals=3))


def lesson_4_causal_masking():
    lesson(4, "Causal Masking — Preventing Future Leakage")
    explain(
        "Language models are trained to predict the NEXT token.",
        "At training time we process all T tokens at once (parallel/efficient).",
        "But: if token i can see token i+1 during training, it can just copy",
        "the answer — it's cheating. The loss would go to near-zero trivially.",
        "",
        "Solution: causal mask. Token i can only attend to tokens 0..i.",
        "",
        "The mask is a lower-triangular matrix:",
    )

    T = 6
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    tokens = ["BOS", "To", "be", "or", "not", "to"]

    print("  Causal mask (True=allowed, False=blocked):")
    print()
    print("         " + "  ".join(f"{t:>3}" for t in tokens))
    for i, row in enumerate(mask):
        cells = "  ".join("  ✓" if v else "  ✗" for v in row)
        print(f"  {tokens[i]:>5}  {cells}")
    print()
    explain(
        "  'To' can attend to: BOS, To (but NOT be, or, not, to)",
        "  'be' can attend to: BOS, To, be (but NOT future tokens)",
        "  This is the autoregressive property — each token predicts the next",
        "  using only its past context.",
    )


def lesson_5_multihead():
    lesson(5, "Multi-Head Attention")
    explain(
        "One attention head computes ONE type of relationship.",
        "Multi-head attention runs H heads in PARALLEL, each with its own",
        "Q, K, V projections → H different attention patterns simultaneously.",
        "",
        "  d_model = 512, n_heads = 8  →  d_k = d_v = 64 per head",
        "",
        "Implementation trick: instead of H separate linear layers,",
        "use ONE big linear W_QKV: (d_model → 3×d_model), then reshape:",
        "",
        "  QKV = x @ W_QKV                 (B, T, 3×d_model)",
        "  Q, K, V = split(QKV, d_model)   each (B, T, d_model)",
        "  Q = Q.view(B, T, H, d_k)        split d_model into H heads",
        "    .transpose(1, 2)               (B, H, T, d_k)",
        "",
        "  Attend all H heads simultaneously: output (B, H, T, d_k)",
        "  Concatenate: (B, T, H×d_k) = (B, T, d_model)",
        "  Project: x @ W_O → (B, T, d_model)",
    )

    B, T, d_model, H = 2, 8, 64, 4
    d_k = d_model // H
    x = torch.randn(B, T, d_model)

    # Single fused projection
    W_QKV = torch.randn(d_model, 3 * d_model) * 0.02
    QKV   = x @ W_QKV               # (B, T, 3×d_model)
    Q, K, V = QKV.split(d_model, dim=-1)

    # Reshape to heads
    Q = Q.view(B, T, H, d_k).transpose(1, 2)   # (B, H, T, d_k)
    K = K.view(B, T, H, d_k).transpose(1, 2)
    V = V.view(B, T, H, d_k).transpose(1, 2)

    show("After reshape: Q", Q.shape)
    show("Attention per head:", f"(B={B}, H={H}, T={T}, d_k={d_k})")

    # Attend (single head shown)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, H, T, T)
    mask   = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~mask, -1e9)
    weights = F.softmax(scores, dim=-1)
    out   = weights @ V   # (B, H, T, d_k)

    # Merge heads
    out = out.transpose(1, 2).contiguous().view(B, T, d_model)
    show("Output after merge :", out.shape)

    explain("",
            "  Each of the 4 heads learned a DIFFERENT attention pattern",
            "  (different W_Q, W_K, W_V initialisation leads to different patterns).",
            "  Some heads specialise in syntax, some in semantics, some in coreference.")


def lesson_6_visualise_patterns():
    lesson(6, "Visualising Attention Patterns")
    explain(
        "After training, different heads learn different patterns:",
        "",
        "  Head type 1: DIAGONAL (attend to self and immediate neighbours)",
        "  Head type 2: FIRST TOKEN (attend mainly to [BOS] or period)",
        "  Head type 3: PREVIOUS TOKEN (attend to the token just before)",
        "  Head type 4: LONG-RANGE (attend to semantically related tokens far away)",
        "",
        "Below are synthetic examples of each pattern for T=8:",
    )

    T = 8
    tok = ["BOS", "The", "cat", "sat", "on", "the", "mat", "."]

    def show_pattern(title, weights):
        print(f"  {title}:")
        print("         " + " ".join(f"{t[:3]:>4}" for t in tok))
        for i, row in enumerate(weights):
            cells = " ".join(f"{'█' if v > 0.5 else '▒' if v > 0.2 else '░' if v > 0.05 else ' ':>4}" for v in row)
            print(f"  {tok[i][:4]:>5}  {cells}")
        print()

    # Diagonal (attend to self and ±1)
    diag = torch.eye(T)
    diag = (diag + 0.3 * torch.tril(torch.ones(T,T)).roll(1, 1).tril()).clamp(0)
    diag = diag / diag.sum(-1, keepdim=True)
    show_pattern("Head 1: Local/Diagonal (attend to self + prev)", diag)

    # First-token
    first = torch.zeros(T, T)
    first[:, 0] = 1.0
    # causal: can't attend to future
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    first = first.masked_fill(~mask, 0)
    first = first / first.sum(-1, keepdim=True).clamp(min=1e-9)
    show_pattern("Head 2: First-Token (attend to BOS)", first)


def lesson_7_complexity():
    lesson(7, "Complexity: Why T² Is a Problem")
    explain(
        "The attention matrix has shape (T, T) — quadratic in sequence length.",
        "",
        "Memory: B × H × T² × 4 bytes (float32)",
    )

    headers = ["T", "H=1", "H=8", "H=32"]
    print(f"  {'':>10}", end="")
    for h in [1, 8, 32]:
        print(f"  {'H='+str(h):>12}", end="")
    print()
    print("  " + "─" * 48)

    for T in [512, 1024, 2048, 4096, 8192, 32768]:
        print(f"  T={T:<6}", end="")
        for H in [1, 8, 32]:
            B = 1
            bytes_fp32 = B * H * T * T * 4
            mb = bytes_fp32 / 1024**2
            if mb < 1024:
                s = f"{mb:.0f} MB"
            else:
                s = f"{mb/1024:.1f} GB"
            print(f"  {s:>12}", end="")
        print()

    explain("",
            "  At T=32768 (32k context), H=32: 128 GB just for attention matrices.",
            "  This is why FlashAttention (O(T) memory via tiling) is essential",
            "  for modern long-context models (GPT-4, Claude, Gemini).",
            "  See tours/flash_tour.py for the full analysis.")


if __name__ == "__main__":
    print()
    print("╔" + "═" * (COLS-2) + "╗")
    print(f"║  Attention from First Principles — 7 Lessons{' ' * (COLS-48)}║")
    print("╚" + "═" * (COLS-2) + "╝")

    lesson_1_permutation_invariance()
    lesson_2_dot_product_similarity()
    lesson_3_scaled_attention_step_by_step()
    lesson_4_causal_masking()
    lesson_5_multihead()
    lesson_6_visualise_patterns()
    lesson_7_complexity()

    print()
    print("╔" + "═" * (COLS-2) + "╗")
    print(f"║  ALL DONE  ──  Next: tours/rope_tour.py{' ' * (COLS-43)}║")
    print("╚" + "═" * (COLS-2) + "╝")
    print()
