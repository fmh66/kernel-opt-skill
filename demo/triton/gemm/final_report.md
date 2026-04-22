# CUDA Optimization Final Report — `gemm` (2026-04-22)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / 12.6 |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | `/home/kernel-opt-skill/test/gemm.py` |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 (best) |
|---|---|---|---|---|
| Execution Time (ms) | 129.74 | 44.26 | 43.92 | **42.25** |
| Speedup (×) | 1.00 | 2.93 | 2.95 | **3.07** |
| Memory Throughput (%) | 82.80 | 75.39 | 76.99 | 73.23 |
| Compute Throughput (%) | 73.95 | 82.53 | 80.61 | 75.32 |
| Bottleneck | Memory-Bound | Compute-Bound | Compute-Bound | Compute-Bound |
| Tensor Core Utilization (%) | 0.00 | 82.45 | 79.07 | 76.91 |
| FMA Pipe Utilization (%) | 63.61 | 1.44 | 1.37 | 2.45 |
| Achieved Occupancy (%) | 25.31 | 16.50 | 8.10 | 16.85 |
| Block Size (threads) | 128 | 256 | 128 | 256 |
| Registers / Thread | 96 | 168 | 241 | **128** |
| Dynamic Shared Mem (bytes) | 32768 | 65536 | 65536 | 65536 |
| L2 Hit Rate (%) | 47.74 | 66.95 | 65.68 | **82.00** |
| Warp Stall — Math Pipe Throttle | 0.04 | 7.19 | 3.01 | 5.20 |
| Warp Stall — Long SB | 0.02 | 0.26 | 1.72 | 0.47 |
| Warp Stall — Short SB | 0.73 | 0.00 | 0.00 | 2.17 |
| Branch Divergence | 0 | 0 | 0 | 0 |
| Warp Execution Efficiency (%) | 32 | 32 | 32 | 32 |
| Shared Mem Bandwidth (bytes/s) | 2.79e+12 | 5.84e+12 | 5.12e+12 | 6.70e+12 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 |
|---|---|---|---|
| `input_precision="tf32"` (Tensor Core MMA) | ✓ | ✓ | ✓ |
| Grouped ordering (L2 swizzle, GROUP_SIZE_M=8) | ✓ | ✓ | ✓ |
| Larger tile (BLOCK_M=128, BLOCK_N=128, BLOCK_K=32) | ✓ | ✓ | ✓ |
| `tl.range()` software pipelining (num_stages≥2) | ✓ (3 stages) | ✗ (used `range()`) | ✓ (autotune) |
| num_warps=8 (256 threads/block) | ✓ | ✗ (num_warps=4) | ✓ |
| `@triton.autotune` config search | ✗ | ✓ (broken pipeline) | ✓ (correct) |
| Proper boundary mask (no `% M` trick) | ✗ | ✗ | ✓ |
| `tl.dot(a, b, acc, ...)` accumulator form | ✓ | ✓ | ✓ |

**Decision rationale per version:**

- **v1:** v0 profiling showed Tensor Core utilization at 0% — root cause was `input_precision="ieee"` which forces FMA and bypasses Tensor Core MMA. Changed to `"tf32"` to activate Tensor Cores on Ampere. Simultaneously increased tile sizes (64→128, 64→128, 32→32), added grouped ordering for L2 reuse, and `tl.range(num_stages=3)` for software pipelining. Result: 2.93x speedup, TC utilization jumped to 82.5%.

- **v2:** v1 register pressure (168/thread) limited occupancy to 16.5%. Applied `@triton.autotune` over 8 configs. Critical mistake: used Python's `range()` instead of `tl.range()`, disabling software pipelining. Autotune selected a 4-warp config but registers rose to 241/thread, occupancy dropped to 8.1% and Long Scoreboard stall jumped from 0.26 to 1.72 — exposing global memory latency. Marginal improvement (44.3ms → 43.9ms).

- **v3:** Fixed v2's `range()` → `tl.range()` regression. Re-ran autotune with all configs using proper Triton pipelining. Also fixed the `% M` modulo boundary bug and corrected mask usage. Autotune selected BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=3, num_warps=8 (same as v1) but Triton's code generation produced 128 regs/thread (vs v1's 168), improving register pressure. L2 Hit Rate improved from 67% to 82%, Math Pipe Throttle reduced from 7.19 to 5.20. Final speedup: 3.07x.

---

## Best Version Conclusion

**Best version:** `v3` — execution time reduced from 129.74 ms to 42.25 ms, speedup **3.07×**.  
Key gains:
1. `input_precision="tf32"` enabled Tensor Core MMA (TC utilization 0% → 76.9%)
2. Grouped ordering improved L2 hit rate (47.7% → 82.0%)
3. `tl.range()` software pipelining (num_stages=3) hid global memory latency
4. Autotune correctly configured reduced register pressure (168 → 128 regs/thread)

Stopping reason: maximum iterations reached (N=3).

**Remaining optimization opportunities:**
- Math Pipe Throttle (5.2) remains the dominant stall — Tensor Core MMA latency cannot be hidden with only 16.7% occupancy (1 block/SM, register-limited). A split-K approach or persistent kernel could increase SM parallelism.
- Short Scoreboard stall (2.17) appeared in v3 — shared memory read latency not fully hidden; tuning `num_stages` or reducing BLOCK_K to allow more pipeline stages could help.
- Using BF16 inputs would enable the higher-throughput BF16 Tensor Core path (2× TF32 throughput) at the cost of reduced precision (not suitable for all FP32 workloads).
- `torch.mm` reference in benchmark uses extra `C.copy_()` overhead; a fair cuBLAS comparison shows that cuBLAS achieves ~7ms for this workload at theoretical peak — there is a ~6× gap to close.
