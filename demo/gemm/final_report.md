# CUDA Optimization Final Report — `gemm` (2026-04-20)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / V12.6.85 |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | `/home/kernel-opt-skill/test/gemm.cu` |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | best |
|---|---|---|---|---|---|
| Execution Time (ms) | 64.2265 | 47.8810 | 11.6437 | 9.4263 | **v3** |
| Speedup (×) | 1.00 | 1.34 | 5.52 | 6.81 | 6.81 |
| Memory Throughput (%) | 39.64 | 25.66 | 53.60 | 73.93 | 73.93 |
| SM Throughput (%) | 98.45 | 79.27 | 55.00 | 72.56 | 72.56 |
| Bottleneck | Latency-Bound (Long SB) | Latency-Bound (Barrier+Not Selected) | Balanced (Short SB + Long SB) | Balanced (Not Selected) | — |
| Achieved Occupancy (%) | 99.76 | 66.68 | 65.25 | 32.69 | — |
| Active Warps / SM | ~48 | ~32 | ~32 | ~16 | — |
| Registers / Thread | 40 | 38 | 56 | 122 | — |
| Global Load Efficiency (%) | 56.25 | 100.00 | 100.00 | 100.00 | — |
| L1 Bank Conflicts (total) | 2.11e+09 | 3.23e+07 | 3.37e+08 | 6.73e+07 | — |
| FMA Pipe Utilization (%) | 19.28 | 7.93 | 38.58 | 55.81 | — |
| IPC | 0.2858 | 0.2154 | 0.5246 | 0.7174 | — |
| Issue Slot Utilization (%) | 28.58 | 21.54 | 52.46 | 71.74 | — |
| Warp Stall — Long SB | 12.75 | 5.66 | 3.01 | 0.88 | — |
| Warp Stall — Short SB | 0.003 | 0.086 | 2.97 | 0.15 | — |
| Warp Stall — Barrier | 0.00 | 3.01 | 1.99 | 0.43 | — |
| Warp Stall — Not Selected | 3.63 | 3.59 | 1.73 | 1.67 | — |
| FFMA Throughput (per cycle) | 661 | 774 | 3937 | 5425 | — |
| Branch Divergence (%) | 0 | 0 | 0 | 0 | — |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 |
|---|---|---|---|
| Coalesced global memory access (128B aligned) | ✓ | ✓ | ✓ |
| Shared memory tiling | ✓ | ✓ | ✓ |
| 2D thread-level register tiling | ✗ | ✓ | ✓ |
| Larger K-tile (BK) | ✗ | ✗ | ✓ |
| Shared memory bank-conflict padding | ✗ | ✗ | ✓ |
| Float4 vectorized C stores | ✗ | ✗ | ✓ |
| `__ldg` / L2 persistence | ✗ | ✗ | ✗ |
| `cp.async` async prefetch | ✗ | ✗ | ✗ |
| Vectorized float4 tile loads | ✗ | ✗ | ✗ |
| Tensor Core (`wmma` / `mma`) | ✗ | ✗ | ✗ |
| `#pragma unroll` (inner K loop) | ✓ | ✓ | ✓ |
| Mixed precision (FP16 / BF16) | ✗ | ✗ | ✗ |

**Decision rationale per version:**

- **v1 — Shared Memory Tiling (TILE=32):** v0's dominant stall was Long Scoreboard (12.75) from global memory latency, caused by the absence of data reuse — every element of A and B was loaded from DRAM for every K step. Global Load Efficiency was 56.25% due to broadcast access patterns in the 16×16 block (same A row accessed by 16 threads, same B columns accessed twice per warp). Introducing a 32×32 shared memory tile eliminated repeated DRAM loads, reducing the Long Scoreboard stall from 12.75 → 5.66 and achieving 100% global load efficiency. Speedup: 1.34×.

- **v2 — 2D Thread-Level Register Tiling (BM=64, BN=64, BK=8, TM=4, TN=4):** v1's new bottleneck was low warp parallelism: Eligible Warps/Cycle = 0.99 (barely 1 warp per cycle), Stall Barrier = 3.01 (syncthreads overhead), and Stall Not Selected = 3.59. The root cause was the large 1024-thread block (TILE=32) limiting to 1 block/SM. By reducing block size to 256 threads (16×16) and having each thread compute a 4×4 tile of 16 outputs, arithmetic intensity per tile increased 16×. This jumped FMA utilization from 7.93% → 38.58%, IPC from 0.22 → 0.52, Issue Slot from 21.54% → 52.46%, and FFMA throughput from 774 → 3937 per cycle. Speedup: 5.52× over v0.

- **v3 — Larger K-tile + Deeper Thread Tile + Bank-Conflict Padding + Float4 Stores:** v2's dominant stalls were Short Scoreboard (2.97 — shared memory latency from reading regA/regB) and Long Scoreboard (3.01). The strategy was to increase compute per shared memory synchronization: doubled BK from 8→16 (half the number of __syncthreads()) and increased TM from 4→8 (32 FMAs per smem load vs 16). Shared memory padding (sA[BK][BM+1]) shifted bank addresses by stride 1 per K row, reducing 2-way conflicts. Float4 stores for the C output improved Global Store Efficiency from 25% → 100%. Result: all stall categories dropped sharply (Long SB: 3.01→0.88, Short SB: 2.97→0.15, Barrier: 1.99→0.43), FMA utilization reached 55.81%, Issue Slot reached 71.74%. Speedup: 6.81× over v0.

---

## Best Version Conclusion

**Best version:** `v3` — execution time reduced from 64.23 ms to 9.43 ms, **speedup 6.81×**.

Key gains:
1. **Shared memory tiling** eliminated repeated DRAM loads and fixed non-coalesced access patterns.
2. **2D thread-level register tiling (TM=8, TN=4)** dramatically increased arithmetic intensity per smem synchronization (BK×TM×TN = 16×8×4 = 512 FMAs per tile).
3. **K-tile doubling (BK=8→16)** halved barrier frequency.
4. **Bank-conflict padding** on sA reduced L1 bank conflicts by 5× vs v2.
5. **Float4 C stores** boosted Global Store Efficiency from 25% → 100%.

Stopping reason: maximum 3 iterations reached.

**Remaining optimization opportunities:**
- **Tensor Cores (WMMA/MMA PTX):** RTX A6000 has Ampere 3rd-gen Tensor Cores with 77.7 TFLOPS FP16/BF16 and 19.5 TFLOPS TF32 peak. Enabling Tensor Core accumulation (e.g., `mma.sync.aligned.m16n8k8`) could yield another 2–4× over v3.
- **Double Buffering (`cp.async`):** Ampere's `cp.async` can overlap global → shared memory transfers with computation, further hiding the remaining Long Scoreboard (0.88) stall.
- **Occupancy vs Register Trade-off:** v3 uses 122 regs/thread → 33.3% occupancy. Reducing TM or using `__launch_bounds__` to cap register allocation could allow 2 blocks/SM (66.7% occupancy), trading per-thread ILP for more warp-level parallelism.
- **`--use_fast_math`:** Enables fused address computations and FTZ mode; may yield ~5–10% on compute-bound phases at the cost of IEEE compliance.
