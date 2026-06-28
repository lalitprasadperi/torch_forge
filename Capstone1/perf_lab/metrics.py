"""
Derived performance metrics: TFLOPS, bandwidth, arithmetic intensity, roofline.

Arithmetic Intensity (AI) = FLOPs / bytes
  AI < ridge_point  -> memory-bandwidth bound
  AI > ridge_point  -> compute bound (ALU is the bottleneck)

Ridge point = peak_TFLOPS / peak_BW_GB_s (in FLOPs per byte).
  Example: 40 TFLOPS / 288 GB/s ≈ 139 FLOPs/byte (RTX PRO 2000 rough estimate)
  Ops with AI well below this (LayerNorm, Softmax, GELU) saturate memory bus.
  Ops with AI well above this (large MatMul) saturate compute.
"""


def tflops(flop_count: int, latency_ms: float) -> float:
    """Achieved TFLOPS = FLOPs / (latency_s * 1e12)."""
    return flop_count / (latency_ms / 1000.0 * 1e12)


def bandwidth_gb_s(byte_count: int, latency_ms: float) -> float:
    """Achieved memory bandwidth in GB/s = bytes / (latency_s * 1e9)."""
    return byte_count / (latency_ms / 1000.0 * 1e9)


def arithmetic_intensity(flop_count: int, byte_count: int) -> float:
    """FLOPs per byte — the x-axis of the roofline model."""
    return flop_count / byte_count if byte_count > 0 else 0.0


def roofline(
    achieved_tflops: float,
    achieved_bw_gb_s: float,
    ai: float,
    peak_tflops: float,
    peak_bw_gb_s: float,
) -> dict:
    """
    Roofline model analysis.

    Roofline ceiling = min(peak_tflops, ai * peak_bw_gb_s)
    Efficiency = achieved / ceiling
    """
    ridge = peak_tflops / peak_bw_gb_s  # FLOPs/byte at the ridge point
    ceiling_tflops = min(peak_tflops, ai * peak_bw_gb_s)
    efficiency = achieved_tflops / ceiling_tflops if ceiling_tflops > 0 else 0.0
    return {
        "ridge_intensity": ridge,
        "bound": "compute" if ai >= ridge else "memory",
        "ceiling_tflops": ceiling_tflops,
        "efficiency_pct": efficiency * 100,
    }
