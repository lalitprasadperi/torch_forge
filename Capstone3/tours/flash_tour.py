"""
flash_tour.py — FlashAttention: Memory Analysis and Benchmarks

Run with:
  python tours/flash_tour.py

Lessons:
  1 — GPU Memory Hierarchy (HBM vs SRAM)
  2 — Why Naive Attention Is Memory-Bound
  3 — The Tiling Algorithm: Online Softmax
  4 — Memory Complexity: O(T²) vs O(T)
  5 — Live Benchmark: Naive vs Flash
  6 — When Does Flash Matter?
"""

import sys
import math
import time
import torch
import torch.nn.functional as F
from pathlib import Path

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
    print(f"  {'▶ ' + label:<36} {value}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lesson_1_memory_hierarchy():
    lesson(1, "GPU Memory Hierarchy (HBM vs SRAM)")
    explain(
        "A modern GPU has two memory levels:",
        "",
        "  HBM (High Bandwidth Memory / VRAM):",
        "    • On-board GPU RAM — 'GPU memory' in nvidia-smi",
        "    • RTX 4090: 24 GB,  BW ≈ 1 TB/s",
        "    • RTX PRO 2000: 16 GB,  BW ≈ 288 GB/s",
        "    • All model weights, activations, KV cache live here",
        "    • SLOW relative to compute — memory transfers dominate runtime",
        "",
        "  SRAM (Static RAM / L2 cache / shared memory):",
        "    • Tiny, on-chip, extremely fast",
        "    • RTX 4090: ~50 MB L2 cache",
        "    • RTX PRO 2000: ~16 MB L2 cache",
        "    • GPU cores read/write this during computation",
        "",
        "  Bandwidth ratio:  SRAM >> HBM",
        "  Capacity ratio:   HBM  >> SRAM",
        "",
        "  The goal of FlashAttention is to keep the attention computation",
        "  inside SRAM (fast), avoiding unnecessary HBM reads/writes.",
    )

    print("  Memory hierarchy diagram:")
    print()
    print("  ┌────────────────────────────────────────────────────────┐")
    print("  │  GPU Chip                                              │")
    print("  │  ┌──────────────┐   ┌──────────────┐                 │")
    print("  │  │ CUDA Cores   │   │  SRAM / L2   │ ~16-50 MB fast │")
    print("  │  │ (compute)    │ ↔ │  shared mem  │                 │")
    print("  │  └──────────────┘   └──────┬───────┘                 │")
    print("  │                            │  slow (1 TB/s on A100)  │")
    print("  │  ┌─────────────────────────┴────────────────────────┐ │")
    print("  │  │  HBM (VRAM) — 16-80 GB                          │ │")
    print("  │  │  weights, activations, KV cache                  │ │")
    print("  │  └──────────────────────────────────────────────────┘ │")
    print("  └────────────────────────────────────────────────────────┘")
    print()


def lesson_2_why_naive_is_memory_bound():
    lesson(2, "Why Naive Attention Is Memory-Bound")
    explain(
        "Naive attention materialises the full (T, T) score matrix in HBM:",
        "",
        "  1. Compute S = Q @ K^T / sqrt(d_k)   → write S to HBM  (T² floats)",
        "  2. Read S from HBM → apply mask → write masked S to HBM",
        "  3. Read S → compute softmax → write P to HBM             (T² floats)",
        "  4. Read P → compute P @ V → write output O to HBM",
        "",
        "  4 HBM reads + 4 HBM writes of O(T²) tensors!",
        "",
        "  The actual FLOP count is O(T² × d_k) — GPU can compute this fast.",
        "  But it's bottlenecked by reading/writing O(T²) to slow HBM.",
        "  This is called MEMORY BOUND: roofline analysis shows bandwidth < compute.",
    )

    print("  HBM traffic for naive attention:")
    print(f"  {'T':>6}  {'H':>4}  {'d_k':>4}  {'Attention matrix':>18}  {'Total HBM r/w':>16}")
    print("  " + "─" * 58)
    for T in [512, 1024, 2048, 4096, 8192]:
        for H in [32]:
            d_k = 64
            B   = 1
            mat_bytes = B * H * T * T * 2   # float16
            total_rw  = 8 * mat_bytes        # 4 reads + 4 writes (rough)
            print(f"  {T:>6}  {H:>4}  {d_k:>4}  "
                  f"  {mat_bytes/1e6:>12.0f} MB  "
                  f"  {total_rw/1e9:>10.1f} GB")
    print()
    explain("  At T=8192, 32 heads: ~2 GB of HBM traffic just for the attention op.",
            "  On a 288 GB/s GPU that's 7 milliseconds of bandwidth time alone.")


def lesson_3_tiling_algorithm():
    lesson(3, "The Tiling Algorithm: Online Softmax")
    explain(
        "FlashAttention's key insight: can we compute softmax WITHOUT materialising",
        "the full attention matrix?",
        "",
        "ONLINE SOFTMAX (Milakov & Gimelshein, 2018):",
        "  Normal softmax(x) needs two passes: one for max, one for sum.",
        "  Online softmax combines them into ONE pass with running statistics.",
        "",
        "  For each new tile of K,V:",
        "    1. Compute local scores S_tile = Q_block @ K_tile^T",
        "    2. Find tile max: m_new = max(m_current, max(S_tile))",
        "    3. Rescale previous accumulator: O = O × exp(m_current - m_new)",
        "    4. Add new contribution: O += exp(S_tile - m_new) @ V_tile",
        "    5. Update normaliser: l = l × exp(m_current - m_new) + sum(exp(S_tile - m_new))",
        "    6. Update m_current = m_new",
        "  Final: O = O / l",
        "",
        "  At NO point is the full T×T matrix stored. Only O(T) is written to HBM.",
    )

    print("  Tiling diagram:")
    print()
    print("    Q blocks (T_q × d_k):        K/V blocks (T_k × d_k):")
    print("    ┌──────────┐                 ┌──┐┌──┐┌──┐┌──┐")
    print("    │ Q_block1 │ ──────────────→ │K1││K2││K3││K4│   Tiles of K")
    print("    ├──────────┤      ↓          └──┘└──┘└──┘└──┘")
    print("    │ Q_block2 │   S_tile stays  ┌──┐┌──┐┌──┐┌──┐")
    print("    ├──────────┤   in SRAM       │V1││V2││V3││V4│   Tiles of V")
    print("    │ Q_block3 │   (never HBM)   └──┘└──┘└──┘└──┘")
    print("    └──────────┘")
    print("          │")
    print("          ↓")
    print("    ┌──────────┐")
    print("    │ Output O │   Written to HBM only ONCE per Q block")
    print("    └──────────┘")
    print()


def lesson_4_memory_complexity():
    lesson(4, "Memory Complexity: O(T²) vs O(T)")
    explain(
        "What actually needs to be stored in HBM?",
        "",
        "  NAIVE ATTENTION:      FLASHATTENTION:",
        "    Q, K, V  → O(Td)      Q, K, V  → O(Td)  (same)",
        "    S = QK^T → O(T²)      S        → 0       (never stored!)",
        "    P = soft → O(T²)      P        → 0       (computed tile by tile)",
        "    O        → O(Td)      O        → O(Td)   (same)",
        "",
        "  HBM peak:  O(T²)      HBM peak:  O(T)",
        "",
        "  FlashAttention also saves recompute during backward (checkpointing):",
        "  Store only (O, softmax_lse) → recompute attention on the fly.",
    )

    print("  Memory comparison (float16, B=1, H=32, d_k=64):")
    print(f"  {'T':>6}  {'QKV (both)':>14}  {'Naive attn':>14}  {'Flash attn':>14}  {'Speedup':>8}")
    print("  " + "─" * 64)
    for T in [512, 1024, 2048, 4096, 8192, 32768]:
        H, d_k = 32, 64
        B = 1
        qkv_mb    = 3 * B * T * H * d_k * 2 / 1e6          # Q+K+V each
        naive_mb  = B * H * T * T * 2 / 1e6                 # attention matrix (float16)
        flash_mb  = B * H * T * d_k * 2 / 1e6               # just output (approx)
        ratio = naive_mb / max(flash_mb, 0.001)
        print(f"  {T:>6}  {qkv_mb:>12.1f}M  {naive_mb:>12.1f}M  "
              f"{flash_mb:>12.1f}M  {ratio:>7.0f}×")
    print()
    explain("  At T=32768: naive needs 128 GB per layer — impossible!",
            "  Flash: 8 MB per layer — perfectly feasible.")


def lesson_5_live_benchmark():
    lesson(5, f"Live Benchmark: Naive vs Flash  [{DEVICE}]")
    explain(
        "Let's measure the actual speedup on your GPU.",
        "We compare our manual implementation vs torch SDPA (uses FlashAttn on CUDA).",
    )

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from transformer.modern.flash_compare import compare_attention_implementations

    results_by_seqlen = {}
    for T in [256, 512, 1024, 2048]:
        r = compare_attention_implementations(
            seq_len=T, d_model=512, n_heads=8, n_repeat=20, device=DEVICE
        )
        results_by_seqlen[T] = r

    print(f"  {'T':>6}  {'Naive (ms)':>12}  {'Flash (ms)':>12}  {'Speedup':>10}  {'Naive mem MB':>14}  {'Flash mem MB':>14}")
    print("  " + "─" * 76)
    for T, r in results_by_seqlen.items():
        naive_t  = r["naive"]["time_ms"]
        flash_t  = r["flash"]["time_ms"]
        speedup  = naive_t / max(flash_t, 0.001)
        naive_m  = r["naive"]["peak_mem_mb"]
        flash_m  = r["flash"]["peak_mem_mb"]
        theory_m = r["theoretical_attn_matrix_mb"]
        print(f"  {T:>6}  {naive_t:>12.2f}  {flash_t:>12.2f}  "
              f"{speedup:>10.2f}×  {naive_m:>14.1f}  {flash_m:>14.1f}")

    print()
    if DEVICE == "cuda":
        explain(
            "  FlashAttention is fastest when T is large (more memory to save).",
            "  Small T: overhead from tiling can make Flash similar to naive.",
            "  Large T: Flash wins dramatically on both speed and memory.",
        )
    else:
        explain(
            "  Running on CPU — FlashAttention's CUDA Tensor Core path is inactive.",
            "  For real speedup numbers, run on a CUDA GPU.",
        )


def lesson_6_when_does_flash_matter():
    lesson(6, "When Does FlashAttention Matter?")
    explain(
        "FlashAttention shines when sequence length T is large.",
        "For T=256 (GPT-nano on Shakespeare), the difference is minimal.",
        "For T=2048+ (modern LLMs), it's essential.",
        "",
        "RULE OF THUMB: use Flash when T > 512. For T < 256, overhead dominates.",
        "",
        "WHERE IT'S USED TODAY:",
        "  • ALL major open-source LLMs (LLaMA 2/3, Mistral, Falcon)",
        "  • Stable Diffusion (for image attention layers)",
        "  • GPT-4, Claude (internally, presumably)",
        "",
        "PYTORCH INTEGRATION (PyTorch >= 2.0):",
        "  torch.nn.functional.scaled_dot_product_attention() automatically",
        "  dispatches to FlashAttention v2 when:",
        "    • Running on CUDA",
        "    • No custom attention mask (or only causal mask)",
        "    • dtype is float16 or bfloat16",
        "",
        "  You don't need to install flash-attn separately — PyTorch includes it!",
        "  Our GPT uses use_flash=True by default, which calls torch SDPA.",
        "",
        "FLASHATTENTION v2 (Dao 2023) improvements over v1:",
        "  • Better GPU work distribution → higher utilisation",
        "  • 2× the throughput of v1",
        "  • Support for attention head dimensions up to 256",
        "",
        "FLASHATTENTION v3 (in development, 2024):",
        "  • Targets Hopper (H100) architecture",
        "  • Overlaps softmax and matmul",
        "  • ~3× faster than v2 on H100",
    )

    print("  Performance scaling with T (theoretical, H=32, d_k=64, A100 GPU):")
    print()
    print(f"  {'T':>6}  {'Naive GB/s':>12}  {'Flash GB/s':>12}  {'Flash advantage':>16}")
    print("  " + "─" * 54)
    peak_bw = 2000   # GB/s (A100)
    for T in [256, 512, 1024, 2048, 4096, 8192]:
        H, d_k = 32, 64
        B = 1
        naive_hbm  = 8 * B * H * T * T * 2 / 1e9   # GB read/write
        naive_gbs  = min(peak_bw * 0.8, peak_bw)    # HBM-bound
        flash_hbm  = 4 * B * H * T * d_k * 2 / 1e9  # only read Q,K,V, write O
        flash_gbs  = min(peak_bw * 0.7, peak_bw)
        advantage  = naive_hbm / max(flash_hbm, 1e-9)
        print(f"  {T:>6}  {'HBM-bound':>12}  {'compute-bound':>12}  {advantage:>14.1f}×  HBM savings")

    print()
    explain("  Larger T → more HBM traffic saved → larger Flash advantage.")


if __name__ == "__main__":
    print()
    print("╔" + "═" * (COLS-2) + "╗")
    print(f"║  FlashAttention: Memory Analysis & Benchmarks{' ' * (COLS-49)}║")
    print(f"║  6 lessons: HBM, tiling, O(T²)→O(T), live benchmark{' ' * (COLS-55)}║")
    print("╚" + "═" * (COLS-2) + "╝")

    lesson_1_memory_hierarchy()
    lesson_2_why_naive_is_memory_bound()
    lesson_3_tiling_algorithm()
    lesson_4_memory_complexity()
    lesson_5_live_benchmark()
    lesson_6_when_does_flash_matter()

    print()
    print("╔" + "═" * (COLS-2) + "╗")
    print(f"║  ALL DONE  ──  Train your GPT: python train_gpt.py --config gpt_nano{' ' * (COLS-73)}║")
    print("╚" + "═" * (COLS-2) + "╝")
    print()
