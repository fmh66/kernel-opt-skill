# CUDA Optimization Final Report — `mha` (2026-04-21)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 (V12.6.85) |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | `/home/kernel-opt-skill/test/mha.cu` |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 (best) |
|---|---|---|---|---|---|---|
| Execution Time (ms) | 15.45 | 4.15 | 1.86 | 1.91 | 1.99 | **1.51** |
| Speedup (×) | 1.00 | 3.72 | 8.31 | 8.09 | 7.76 | **10.23** |
| Memory Throughput (%) | 0.09 | 0.32 | 0.76 | 0.74 | 0.72 | 0.95 |
| SM Throughput (%) | 87.0 | 17.8 | 33.4 | 32.4 | 87.9 | **96.1** |
| Bottleneck | Latency (Barrier) | Latency (Short SB) | Latency (Long SB) | Latency (Long SB) | Compute (low occ.) | Compute |
| Achieved Occupancy (%) | 62.7 | 63.9 | 64.8 | 81.0 | 20.5 | 40.3 |
| Active Warps / SM | 30.1 | 30.7 | 31.1 | 38.9 | 9.8 | 19.4 |
| Registers / Thread | 48 | 40 | 52 | 40 | 64 | 40 |
| Warp Stall — Long SB | 4.85 | 6.73 | 21.39 | 18.12 | 2.02 | 1.56 |
| Warp Stall — Short SB | 0.20 | 8.48 | 2.00 | 4.43 | 0.85 | 1.84 |
| Warp Stall — Barrier | 14.51 | 2.06 | 0.73 | 1.82 | 0.53 | 1.37 |
| Warp Execution Efficiency (%) | 1.72 | 31.93 | 31.91 | 31.93 | 31.58 | 31.45 |
| Global Load Efficiency (%) | 17.6 | 22.2 | 66.7 | 66.7 | **100.0** | **100.0** |
| L1 Bank Conflicts (total) | 76K | 368M | 76M | 62M | 7.6M | 3.7M |
| Smem Bandwidth (TB/s) | 0.02 | 0.08 | 0.20 | 0.20 | 4.81 | 4.81 |
| Divergent Branch Targets | 8K | 16K | 25K | 16K | 262K | 262K |
| Dynamic Smem (bytes) | 4096 | 4480 | 4480 | 8832 | 17280 | 17792 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Coalesced global memory access | ✗ | ✗ | ✗ | ✓ | ✓ |
| Shared memory tiling (K/V) | ✗ | ✗ | ✗ | ✓ | ✓ |
| `__ldg` read-only cache hints | ✗ | ✓ | ✓ | ✗ | ✗ |
| Vectorized loads (`float4`) | ✗ | ✓ | ✓ | ✗ | ✗ |
| Tensor Core (`wmma` / `mma`) | ✗ | ✗ | ✗ | ✗ | ✗ |
| ILP (4-accumulator unrolling) | ✗ | ✓ | ✓ | ✓ | ✓ |
| Online softmax (Flash Attention 2) | ✗ | ✗ | ✗ | ✓ | ✓ |
| Smem bank-conflict padding (+1) | ✗ | ✗ | ✗ | ✓ | ✓ |
| Larger block size / `__launch_bounds__` | ✗ | ✓ | ✓ | ✓ | ✓ |
| BLOCKQ > 1 (multi-query per block) | ✗ | ✗ | ✓ | ✗ | ✓ |

**Decision rationale per version:**

- **v1:** The baseline kernel serializes all QK^T dot products to a single thread (`if (t == 0)` inner loop), leaving 63/64 threads idle. v1 parallelises the dot product across all `dk` threads, eliminating the dominant Stall: Barrier (14.5) and achieving a 3.7× speedup. Warp Execution Efficiency jumped from 1.7% to 31.9%.

- **v2:** With N=1024 scores computed per query, v1 still had poor memory access quality: each thread loaded a full K row (stride = d_model = 2048 bytes between float4 loads), and had 368M L1 bank conflicts. v2 introduced `float4` loads for both Q (loaded to smem) and K (via `__ldg`), and used 4 independent FMA accumulators (a0–a3) to break the FMA dependency chain. Global Load Efficiency improved from 22% to 66.7%, bank conflicts fell 5×, and execution dropped to 1.86ms (8.3×). The dominant new bottleneck was Stall: Long Scoreboard (21.4) from K's non-coalesced column-major access pattern (stride d_model between different threads).

- **v3:** Observed that v2's Long Scoreboard stall was linked to non-coalesced K/V accesses. v3 tried to improve L1 reuse by processing 2 queries per block (BLOCKQ=2, 128 threads). Both groups access identical K/V addresses, creating temporal L1 reuse and increasing occupancy to 81%. However, Long Scoreboard stalls only fell from 21.4 to 18.1 (K is still loaded scattered from DRAM), and the grid size halved (less parallelism), leaving execution at 1.91ms — marginally slower than v2 in practice.

- **v4:** The root cause of Long Scoreboard stalls was identified: K/V were accessed with stride d_model=512 floats between consecutive threads — 2KB strides, requiring one cache line per thread. v4 introduced Flash Attention 2's online softmax (`m_t`, `l_t`, `o_t` recurrence) combined with smem tiling: a `kv_sm[dk][dk+1]` tile is loaded with ALL threads contributing one column each (coalesced), then kept in smem (20-cycle latency) for dot product computation. The +1-float row padding eliminates bank conflicts. Result: Global Load Efficiency reached 100%, Long Scoreboard stalls fell to 2.0, smem bandwidth rose 24×. SM Throughput jumped to 87.9%. However, the 17280-byte smem tile limits occupancy to 20.5% (5 blocks/SM × 2 warps = 10 warps vs 48 max), partially cancelling the per-block efficiency gains. Execution time was 1.99ms — similar to v2.

  **Bug fixed:** A race condition was identified where warp 0 could overwrite `warp_buf[0]` (which stores the tile max after the cross-warp max reduction) with the tile sum before warp 1 read the tile max value. Fixed by inserting one extra `__syncthreads()` after `float tile_max = warp_buf[0]` to guarantee all warps have captured the value before any warp overwrites the slot for the sum reduction.

- **v5:** v4's bottleneck was low occupancy (20.5%, 2 warps/block) preventing latency hiding. v5 applies BLOCKQ=2 to the v4 Flash Attention framework: 128 threads (4 warps) process 2 queries simultaneously, sharing a single `kv_sm` K/V tile. Each group loads half the tile rows (32 iterations instead of 64), halving per-group load latency. The total smem grows to 17792 bytes, still fitting 5 blocks/SM, but with 4 warps/block → 20 resident warps/SM → **40.3% occupancy** (2×). Eligible warps/cycle rose from 0.38 to 0.65. Result: SM Throughput 96.1%, FFMA throughput 440 ops/cycle, execution time **1.51ms = 10.2× speedup over baseline**.

---

## Best Version Conclusion

**Best version:** `v5` — execution time reduced from **15.45 ms** to **1.51 ms**, speedup **10.23×**.

Key gains:
1. **v1:** Eliminate serial `if (t==0)` bottleneck → 3.7× speedup
2. **v2:** Float4 vectorized loads + 4 FMA accumulators for ILP → 8.3× total
3. **v4:** Flash Attention 2 online softmax + coalesced smem K/V tiling + +1-padding for bank-conflict elimination → Global Load Efficiency 100%, Long Scoreboard stalls reduced 10×
4. **v5:** BLOCKQ=2 doubles active warps/SM (40% occupancy) → SM Throughput 96%, final 10.2× total

Stopping reason: Maximum iteration count (N=5) reached.

**Remaining optimization opportunities:**
- **Async prefetch (`cp.async`):** Ampere supports pipelining global→smem loads. Overlapping the next tile's K load with the current tile's dot product could further hide memory latency and push SM utilization toward 100%.
- **Tensor Cores:** Converting Q, K, V to FP16 and using `mma` would enable 8× FP32 compute density. At 96% SM throughput in FP32, the FP16 kernel would be compute-bound at a much lower time budget.
- **Further occupancy tuning:** Reducing smem to <16384 bytes (e.g., TILE_K=32 instead of dk=64) would fit 6 blocks/SM → 50% occupancy without changing the BLOCKQ approach.
- **Warp Execution Efficiency (31.5%):** The `if (lt==0)` serial reduction serializes 1/64 threads per sync. A warp-shuffle-based tree reduction could reduce the number of idle warp cycles.
