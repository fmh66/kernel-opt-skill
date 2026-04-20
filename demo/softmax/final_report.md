# CUDA Optimization Final Report — `softmax` (`2026-04-20`)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 (Build cuda_12.6.r12.6/compiler.35059454_0) |
| ncu | 2024.3.2.0 (build 34861637) |
| nsight-python | 0.9.6 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | `/home/kernel-opt-skill/test/softmax.cu` |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | best |
|---|---|---|---|---|---|
| Execution Time (ns) | 891,936 | 124,896 | 131,424 | 127,808 | **v1** |
| Speedup (×) | 1.00 | **7.14** | 6.79 | 6.98 | 7.14× |
| Memory Throughput (%) | 32.3 | **92.3** | 90.1 | 90.4 | v1 |
| Compute Throughput (SM %) | 2.9 | 33.0 | 31.1 | 21.2 | v1 |
| DRAM Total BW (GB/s) | 235 | **673** | 657 | 659 | v1 |
| Bottleneck | Latency-Bound | Memory-Bound | Memory-Bound | Memory-Bound | — |
| Achieved Occupancy (%) | 16.6 | 94.8 | 85.4 | **96.9** | v3 (but v1 is fastest) |
| Waves / SM | 0.08 | 20.3 | 20.3 | 20.3 | — |
| Registers / Thread | 38 | 40 | 32 | 40 | — |
| Static Shared Mem (bytes) | 0 | 32 | 64 | 32 | — |
| Warp Stall — Long SB (%) | 46.3 | 18.0 | 20.3 | 24.5 | v1 |
| Warp Stall — Barrier (%) | 0.0 | 6.5 | 5.9 | 11.3 | v0/v2 |
| Branch Divergence (%) | 0.0 | 0.0 | 0.0 | 0.0 | — |
| Global Load Efficiency (%) | 12.5 | **100** | **100** | **100** | tied |
| L1 Hit Rate (%) | 91.3 | 54.0 | 21.1 | 53.3 | — |
| Issue Slot Utilization (%) | 3.9 | 28.8 | 33.0 | 22.5 | v2 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 |
|---|---|---|---|
| Block-per-row (coalesced global access) | ✓ | ✓ | ✓ |
| Warp shuffle reduction | ✓ | ✓ | ✓ |
| Shared memory for warp-level broadcast | ✓ | ✓ | ✓ |
| `__ldg` / `__restrict__` hints | ✓ | ✓ | ✓ |
| `__launch_bounds__` | ✗ | ✗ | ✓ |
| Vectorized loads (`float4`) | ✗ | ✓ | ✓ |
| Online softmax algorithm (2-pass) | ✗ | ✓ | ✗ |
| `__expf()` fast math | ✗ | ✗ | ✓ |
| `__fdividef()` | ✓ | ✓ | ✓ |

**Decision rationale per version:**

- **v1:** Root cause of v0 was one-thread-per-row: each warp's threads accessed different rows (stride D=4096 bytes), yielding 12.5% global load efficiency and only 40 blocks for 108 SMs (0.08 waves/SM, 16.6% occupancy). Fix: block-per-row with 10,240 blocks. All threads in a warp now access consecutive floats in one row (100% coalesced). Warp shuffle + shared memory reduction eliminates the need for global atomics. Result: 7.14× speedup, Memory-Bound at 92.3%.

- **v2:** Hypothesis: online softmax (Milakov & Gimelshein) eliminates pass 3 (output read-back) and saves one read of input, reducing logical data movement from 5 passes to 3. Combined with `float4` to reduce instruction count. Reality: v1's pass 3 reads the output from L1 cache (54% L1 hit rate), so actual DRAM savings were minimal. Meanwhile, the online algorithm adds two `expf()` calls per element in pass 1 (running max update), increasing SM compute overhead. The 2× shared memory arrays (smem_m + smem_s) also reduced register budget and occupancy (85.4% vs 94.8%). Net result: 5% slower than v1.

- **v3:** Returned to 3-pass approach to preserve v1's L1 cache reuse. Added `float4` vectorized I/O and `__expf()` (fast SFU approximation). However, float4 reduces per-thread loop iterations from 4 to 1 (with D=1024, BLOCK_SIZE=256: D4=256=BLOCK_SIZE → 1 float4 per thread). Fewer iterations mean fewer independent memory requests in flight, reducing the GPU's ability to hide DRAM latency → Long Scoreboard stall rose from 18% to 24.5%. `__expf()` uses SFU units (no FMA), so FMA throughput dropped to 0% and instructions-per-cycle fell. Net result: 2.3% slower than v1.

---

## Best Version Conclusion

**Best version: `v1`** — execution time reduced from **891,936 ns** to **124,896 ns**, speedup **7.14×** over the naive baseline.

Key gains:
1. **Block-per-row parallelism**: 40 → 10,240 blocks; 0.08 → 20.3 waves/SM; occupancy 16.6% → 94.8%
2. **Coalesced global memory access**: load efficiency 12.5% → 100%; DRAM utilization 32% → 92.3%
3. **Warp shuffle reduction**: eliminates shared memory for max/sum reduction (only 32 bytes of smem for warp-level broadcast)
4. **`__ldg` / `__restrict__`**: read-only L1 cache hints reduce L2 pressure

Stopping reason: maximum iterations (N=3) reached.

**Remaining optimization opportunities:**

- **Larger D values (D ≥ 2048)**: with more elements per row, each thread would iterate more in the inner loops, providing better ILP for latency hiding. The scalar 3-pass approach would benefit even more, and `float4` would then provide a meaningful speedup by reducing instruction count without hurting ILP.
- **Warp-level softmax (D ≤ 32)**: for very short rows, use a single warp per row with only warp-shuffle reductions and no shared memory.
- **`cp.async` prefetching** (Ampere-native): issue async loads for the next tile while computing on the current tile to better overlap computation and memory access.
- **Reducing __syncthreads() count**: fuse the two-level reduction into a single barrier using Cooperative Groups or minimize sync points with a tuned block size where all warp sums fit in a single warp's shuffle.
- **Half-precision (FP16) output**: if the downstream consumer accepts FP16, converting during the final write halves write bandwidth.
