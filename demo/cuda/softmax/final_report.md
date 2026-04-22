# CUDA Optimization Final Report — `softmax` (2026-04-20)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / V12.6.85 |
| ncu | 2024.3.2.0 (build 34861637) |
| nsight-python | 0.9.6 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | `demo/softmax/v3/v3.cu` |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 (best) |
|---|---|---|---|---|
| Execution Time (ms) | 0.8992 | 0.2219 | 0.1515 | **0.1424** |
| Speedup (×) | 1.00 | 4.05× | 5.93× | **6.32×** |
| Memory Throughput (%) | 33.2 | 81.9 | 90.0 | **91.4** |
| Compute Throughput (%) | 3.0 | 16.8 | 28.4 | 30.9 |
| Bottleneck | Latency-Bound | Memory-Bound | Memory-Bound | Memory-Bound |
| Achieved Occupancy (%) | 16.6 | 87.5 | 88.0 | **97.0** |
| Registers / Thread | 38 | 21 | 28 | 24 |
| Warp Stall — Long SB (%) | 46.2 | 52.5 | 23.2 | **19.4** |
| Warp Stall — Short SB (%) | 1.3 | 1.2 | 0.7 | 2.1 |
| Warp Stall — Barrier (%) | 0.0 | 7.4 | 5.7 | 7.3 |
| Global Load Efficiency (%) | 12.5 | 100.0 | 100.0 | 100.0 |
| Global Store Efficiency (%) | 12.5 | 100.0 | 100.0 | 100.0 |
| L1 Hit Rate (%) | 91.8 | 43.7 | 21.5 | 0.0 |
| L2 Hit Rate (%) | 84.2 | 62.2 | 54.7 | 50.1 |
| Shared Memory BW (bytes/s) | 0 | 5.92e9 | 1.56e10 | **1.02e12** |
| Branch Divergence (%) | 0 | 0 | 0 | 0 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 |
|---|---|---|---|
| One block per row (vs one thread per row) | ✓ | ✓ | ✓ |
| Coalesced global memory access | ✓ | ✓ | ✓ |
| Warp shuffle reduction (`__shfl_xor_sync`) | ✓ | ✓ | ✓ |
| Shared memory for warp → block reduction | ✓ | ✓ | ✓ |
| Vectorized loads (`float4`) | ✗ | ✓ | ✓ |
| Online softmax (fused max+sum in one pass) | ✗ | ✓ | ✓ |
| `__ldg` read-only cache | ✗ | ✓ | ✓ |
| Shared memory row cache (eliminate 2nd DRAM read) | ✗ | ✗ | ✓ |
| `__launch_bounds__` register hint | ✗ | ✗ | ✓ |
| Larger block size (256 vs 128) | ✗ | ✓ | ✓ |

**Decision rationale per version:**

- **v1:** v0 assigns one thread per row, causing stride-D strided accesses (12.5% Global Load Efficiency) and only 40 blocks for 84 SMs (16.6% Achieved Occupancy). Switching to one block per row coalesces access to 100%, increases grid to N=10240 blocks (occupancy 87.5%), and uses warp-shuffle + shared-memory reduction. Result: **4.05× speedup** (Latency-Bound → Memory-Bound).

- **v2:** v1 still makes 3 DRAM passes (max pass → exp+sum pass → normalize pass). Online softmax fuses max+sum into a single pass using the online merge rule `(m,s)+(m',s') → (max(m,m'), s·exp(m−M)+s'·exp(m'−M))`, reducing to 2 DRAM reads + 1 write. float4 loads reduce L1 transaction count by 4×, and 256-thread blocks improve ILP. Stall Long Scoreboard drops from 52.5% to 23.2%. Result: **1.46× speedup over v1, 5.93× over v0**.

- **v3:** v2 still reads DRAM twice (online pass + normalize pass). Caching the entire row (4 KB) into shared memory eliminates the second DRAM read entirely. After a single float4 DRAM→shmem load, all compute accesses shared memory (latency ~20 cycles vs ~400 cycles for DRAM). `__launch_bounds__(256, 4)` reduces register pressure from 28 to 24 per thread, improving occupancy to 97.0%. Shared memory bandwidth rises to 1.02 TB/s. Result: **1.06× speedup over v2, 6.32× over v0**.

---

## Best Version Conclusion

**Best version: `v3`** — execution time reduced from **0.8992 ms** to **0.1424 ms**, total speedup **6.32×**.

Key gains:
1. **Parallelism fix** (v0→v1): one block per row restored coalesced access and enabled full GPU occupancy — single largest gain (4×)
2. **Online softmax + float4** (v1→v2): fused reduction pass eliminated one DRAM read, vectorized loads reduced transaction count — 1.46×
3. **Shared memory row cache** (v2→v3): eliminated the second DRAM read entirely, achieving minimum possible DRAM traffic (1 read + 1 write = 80 MB) — 1.06×

Stopping reason: maximum iterations (N=3) reached. Memory throughput is at 91.4% of hardware peak; the kernel is effectively saturating the DRAM bandwidth of the RTX A6000.

**vs PyTorch (benchmark):** v3 runs in 0.1469 ms vs PyTorch's 0.2721 ms — **1.85× faster** than the framework's optimized implementation at equal memory throughput (~92%), confirming we achieve optimal DRAM utilization.

**Remaining optimization opportunities:**
- Use `cp.async` (Ampere) to pipeline the shmem load phase and overlap it with computation from the previous block's work (persistent kernel pattern)
- Explore `__half2` (FP16) for reduced memory bandwidth if precision allows
- For variable D at runtime, add a code path for D < 128 using pure warp-shuffle without shared memory
- Grid-stride persistent kernel to reduce launch overhead across batches

