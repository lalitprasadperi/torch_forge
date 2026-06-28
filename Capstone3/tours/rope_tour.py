"""
rope_tour.py — Rotary Position Embeddings (RoPE) from First Principles

Run with:
  python tours/rope_tour.py

Lessons:
  1 — Why absolute position encoding fails at long context
  2 — Complex number rotation — the math behind RoPE
  3 — From 2D rotation to full d_k rotation
  4 — The relative position property
  5 — Visualise RoPE frequencies
  6 — RoPE in practice: apply to Q and K
  7 — Extrapolation beyond training length
"""

import sys
import math
import torch
import torch.nn.functional as F

COLS = 68

def lesson(n, title):
    print()
    print("╔" + "═" * (COLS-2) + "╗")
    print(f"║  LESSON {n}: {title:<{COLS-14}}║")
    print("╚" + "═" * (COLS-2) + "╝")

def explain(*lines):
    for l in lines:
        print(f"  {l}")
    print()

def show(label, value):
    print(f"  {'▶ ' + label:<34} {value}")


def lesson_1_problem_with_absolute():
    lesson(1, "Why Absolute Position Encoding Fails at Long Context")
    explain(
        "Sinusoidal / Learned PE adds a position vector to the token embedding:",
        "  x_i = embed(token_i) + pe(i)",
        "",
        "This encodes ABSOLUTE position. But consider:",
        "  'John told Mary that he loved her' — 'he' refers to 'John'  (2 apart)",
        "  In a long document, same relationship but positions might be 500 apart",
        "",
        "Absolute PE: the model must learn that position 5 and position 505",
        "have the same syntactic relationship as position 5 and position 7.",
        "This generalises poorly to positions not seen during training.",
        "",
        "RELATIVE PE goal: attention(Q_i, K_j) should naturally depend on (i-j),",
        "not on i and j individually. RoPE achieves this elegantly.",
    )


def lesson_2_rotation_math():
    lesson(2, "Complex Number Rotation — The Math Behind RoPE")
    explain(
        "A 2D vector [x, y] can be written as a complex number: z = x + iy",
        "",
        "Rotating by angle θ: z' = z × e^(iθ) = z × (cosθ + i sinθ)",
        "  x' = x cosθ - y sinθ",
        "  y' = x sinθ + y cosθ",
        "",
        "KEY PROPERTY: the dot product of two rotated vectors only depends",
        "on the angle DIFFERENCE, not on the absolute angles:",
        "",
        "  rotate(q, θ_m) · rotate(k, θ_n)",
        "  = q · R(θ_m - θ_n) · k",
        "  = f(q, k, m-n)   ← only relative position m-n matters!",
        "",
        "RoPE encodes position m by rotating the query vector by m×θ:",
        "  θᵢ = pos / 10000^(2i/d_k)",
        "  (same base frequencies as sinusoidal PE, but applied as rotation to Q,K)",
    )

    θ = math.pi / 4   # 45 degrees

    def rotate_2d(v, angle):
        cos, sin = math.cos(angle), math.sin(angle)
        return torch.tensor([v[0]*cos - v[1]*sin, v[0]*sin + v[1]*cos])

    q = torch.tensor([1.0, 0.0])
    k = torch.tensor([0.8, 0.6])

    # Without rotation
    dot_orig = (q @ k).item()

    # Rotate by same angle
    q_rot = rotate_2d(q, θ)
    k_rot = rotate_2d(k, θ)
    dot_same = (q_rot @ k_rot).item()

    # Rotate by different angles
    q_rot2 = rotate_2d(q, θ)
    k_rot2 = rotate_2d(k, 2*θ)   # relative angle = θ
    dot_diff = (q_rot2 @ k_rot2).item()

    show("q · k (no rotation)           :", f"{dot_orig:.4f}")
    show("rotate(q,θ) · rotate(k,θ)     :", f"{dot_same:.4f}  ← same as no rotation (Δangle=0)")
    show("rotate(q,θ) · rotate(k,2θ)    :", f"{dot_diff:.4f}  ← depends on Δangle=θ only")
    explain("",
            "  Same relative angle (Δ=0) → same dot product as un-rotated.",
            "  Different absolute positions, same relative angle → same score.",
            "  This is the relative position property.")


def lesson_3_full_dimension_rotation():
    lesson(3, "From 2D to Full d_k Rotation")
    explain(
        "For a d_k-dimensional vector, treat it as d_k/2 pairs:",
        "  [x₀, x₁, x₂, x₃, ..., x_{d-2}, x_{d-1}]",
        "   └──pair 0──┘  └──pair 1──┘   └───pair d/2-1───┘",
        "",
        "Each pair (x_{2i}, x_{2i+1}) gets rotated by a DIFFERENT frequency:",
        "  θᵢ = pos / 10000^(2i/d_k)",
        "",
        "  Pair 0 (i=0): θ = pos / 10000^0      = pos         (fast oscillation)",
        "  Pair 1 (i=1): θ = pos / 10000^(2/d)               (slower)",
        "  ...                                                  ...",
        "  Pair d/2 (last): θ = pos / 10000^1   ≈ pos/10000  (very slow)",
        "",
        "The rotation matrix R_pos is block-diagonal:",
        "  R_pos = diag(R(θ₀), R(θ₁), ..., R(θ_{d/2-1}))",
        "  Each 2×2 block is a standard 2D rotation matrix.",
        "",
        "Efficient implementation (no explicit matrix construction):",
        "  x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)",
        "  where rotate_half(x) = [-x_{d/2:}, x_{:d/2}]",
    )

    d_k = 8
    pos = 5

    # Compute frequencies for d_k=8 (4 pairs)
    freqs = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))
    angles = pos * freqs   # one angle per pair

    explain(f"  Rotation angles for position {pos}, d_k={d_k}:")
    for i, θ in enumerate(angles):
        print(f"    pair {i}: θ = {pos} / 10000^({2*i}/{d_k}) = {θ.item():.4f} rad "
              f"({math.degrees(θ.item()):.1f}°)")
    print()
    explain("  Low-index pairs rotate fast (sensitive to nearby positions).",
            "  High-index pairs rotate slowly (sensitive to distant positions).",
            "  Together they provide multi-scale position sensitivity.")


def lesson_4_relative_position_property():
    lesson(4, "The Relative Position Property — Verified")
    explain(
        "Let's verify numerically that RoPE dot products depend only on",
        "the RELATIVE distance between positions, not absolute positions.",
    )

    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from transformer.modern.rope import RotaryEmbedding, apply_rope

    torch.manual_seed(42)
    d_k     = 32
    n_heads = 1
    rope    = RotaryEmbedding(d_k, max_len=128)

    q = torch.randn(1, n_heads, 1, d_k)
    k = torch.randn(1, n_heads, 1, d_k)

    results = []
    pairs = [(0, 5), (3, 8), (10, 15), (50, 55)]   # all have distance 5

    for pos_q, pos_k in pairs:
        cos_q, sin_q = rope(1, torch.device("cpu"), offset=pos_q)
        cos_k, sin_k = rope(1, torch.device("cpu"), offset=pos_k)
        q_rot = apply_rope(q, cos_q, sin_q)
        k_rot = apply_rope(k, cos_k, sin_k)
        dot = (q_rot * k_rot).sum(-1).item()
        results.append((pos_q, pos_k, dot))

    explain("  dot product(rotate(q, m), rotate(k, n)) for various (m, n) with n-m=5:")
    for pos_q, pos_k, dot in results:
        print(f"    pos_q={pos_q:>3}, pos_k={pos_k:>3}  (distance=5): dot = {dot:.4f}")

    print()
    explain("  All dots are IDENTICAL — only the relative distance (5) matters,",
            "  not the absolute positions. This is the relative position property.")


def lesson_5_visualise_frequencies():
    lesson(5, "Visualising RoPE Frequencies")
    explain(
        "Each dimension pair of Q and K oscillates at a different frequency.",
        "Lower pairs oscillate fast (useful for nearby position discrimination).",
        "Higher pairs oscillate slowly (useful for long-range relationships).",
    )

    d_k = 16
    max_pos = 50

    print("  Rotation angle per dimension pair across positions 0..49:")
    print()
    print(f"  {'Pos':>4} ", end="")
    for i in range(0, d_k, 2):
        print(f"  dim{i:02d}", end="")
    print()
    print("  " + "─" * (6 + (d_k//2) * 7))

    for pos in [0, 1, 5, 10, 25, 50]:
        freqs = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))
        angles = pos * freqs
        print(f"  {pos:>4} ", end="")
        for a in angles:
            a_norm = (a.item() % (2 * math.pi)) / (2 * math.pi)
            bar = "█" if a_norm > 0.75 else "▓" if a_norm > 0.5 else "▒" if a_norm > 0.25 else "░"
            print(f"  {bar:>5}{a.item():.1f}"[:7], end="")
        print()

    print()
    explain(
        "  dim00 (fastest): completes many cycles even at pos=10",
        "  dim14 (slowest): barely moves even at pos=50",
        "  This multi-scale encoding is similar to sinusoidal PE but applied",
        "  as a rotation to Q,K — not as an additive offset to the embedding.",
    )


def lesson_6_apply_rope():
    lesson(6, "RoPE in Practice: Apply to Q and K")
    explain(
        "RoPE is applied inside the attention layer, AFTER the Q,K,V projections",
        "and BEFORE computing attention scores.",
        "",
        "  x → W_Q → Q → rotate(Q, pos) → Q_rope",
        "  x → W_K → K → rotate(K, pos) → K_rope",
        "  scores = Q_rope @ K_rope^T / sqrt(d_k)",
        "",
        "Note: V is NOT rotated — only Q and K.",
        "The output of attention is a weighted sum of values, so it doesn't",
        "need position encoding (the weights already have position awareness).",
    )

    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from transformer.modern.rope import RotaryEmbedding, apply_rope

    B, H, T, d_k = 2, 4, 8, 32
    rope = RotaryEmbedding(d_k, max_len=64)

    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)

    cos, sin = rope(T, device=torch.device("cpu"))

    Q_rope = apply_rope(Q, cos, sin)
    K_rope = apply_rope(K, cos, sin)

    show("Q before RoPE  shape :", Q.shape)
    show("Q after  RoPE  shape :", Q_rope.shape)
    show("Norm preserved?       :", f"before={Q[0,0,0].norm():.4f}  "
         f"after={Q_rope[0,0,0].norm():.4f}  (rotation preserves norm)")

    # Compute attention with and without RoPE
    scores_plain = (Q @ K.transpose(-2,-1)) / math.sqrt(d_k)
    scores_rope  = (Q_rope @ K_rope.transpose(-2,-1)) / math.sqrt(d_k)

    show("Scores differ?        :", f"max diff = {(scores_plain - scores_rope).abs().max():.4f}")
    explain("  Scores differ → RoPE changes attention patterns by encoding position.")


def lesson_7_extrapolation():
    lesson(7, "Extrapolation Beyond Training Length")
    explain(
        "A key motivation for RoPE was length extrapolation:",
        "  Train on max_len=2048 → can it attend to token 3000 at inference?",
        "",
        "LEARNT PE: impossible — position 3000 was never seen, embedding is random.",
        "",
        "ROPE: frequencies are mathematically defined, not learned.",
        "  Position 3000 gets a valid rotation angle → the model CAN extrapolate,",
        "  though quality degrades beyond ≈2× training length.",
        "",
        "YARN (Yet Another RoPE Extension, NTK-aware):",
        "  Scale the base frequency: 10000 → 10000 × (new_len/train_len)^(d/(d-2))",
        "  This spreads the frequency range to cover the new longer context.",
        "  Used in LLaMA-3 to extend 8k → 128k context.",
        "",
        "LONGROPE / ROPE SCALING (LLaMA-3.1):",
        "  Per-dimension scaling factors tuned empirically.",
        "  Allows 8k → 1M context with minimal quality loss.",
    )

    # Show that RoPE angles at out-of-training positions are valid
    d_k = 8
    freqs = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))

    print("  RoPE angles at various positions (all valid, even beyond training):")
    for pos in [0, 100, 1000, 2048, 5000, 10000]:
        angles = (pos * freqs) % (2 * math.pi)
        angle_str = "  ".join(f"{a:.2f}" for a in angles)
        print(f"  pos={pos:>6}: [{angle_str}]  (all in [0, 2π])")
    print()
    explain("  Every position produces a valid rotation angle.",
            "  The mathematical formula never 'runs out' — unlike a lookup table.")


if __name__ == "__main__":
    print()
    print("╔" + "═" * (COLS-2) + "╗")
    print(f"║  Rotary Position Embeddings (RoPE) — 7 Lessons{' ' * (COLS-50)}║")
    print("╚" + "═" * (COLS-2) + "╝")

    lesson_1_problem_with_absolute()
    lesson_2_rotation_math()
    lesson_3_full_dimension_rotation()
    lesson_4_relative_position_property()
    lesson_5_visualise_frequencies()
    lesson_6_apply_rope()
    lesson_7_extrapolation()

    print()
    print("╔" + "═" * (COLS-2) + "╗")
    print(f"║  ALL DONE  ──  Next: tours/flash_tour.py{' ' * (COLS-44)}║")
    print("╚" + "═" * (COLS-2) + "╝")
    print()
