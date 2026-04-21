# CUDA Optimization Final Report — `<kernel_name>` (`<date>`)

## Environment

| Item | Value |
|---|---|
| GPU | `<name>` (CC `<x.y>`) |
| CUDA / nvcc | `<version>` |
| ncu | `<version>` |
| nsight-python | `<version>` |
| Triton | `<version>` |
| PyTorch | `<version>` |
| Kernel file | `<path>` |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | ... | best |
|---|---|---|---|---|---|---|
| Execution Time (ms) | | | | | | |
| Speedup (×) | 1.00 | | | | | |
| Memory Throughput (%) | | | | | | |
| Compute Throughput (%) | | | | | | |
| SM Active Cycles (%) | | | | | | |
| Bottleneck | | | | | | |
| Achieved Occupancy (%) | | | | | | |
| Active Warps / SM | | | | | | |
| Registers / Thread | | | | | | |
| Warp Stall — Long SB (%) | | | | | | |
| Warp Stall — Short SB (%) | | | | | | |
| Branch Divergence (%) | | | | | | |
| ... | | | | | | |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | ... |
|---|---|---|---|---|
| Coalesced global memory access (128B aligned) | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Shared memory tiling | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| `__ldg` / L2 persistence | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| `cp.async` async prefetch | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Vectorized loads (`float4`) | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Tensor Core (`wmma` / `mma`) | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| ILP (loop unrolling) | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Mixed precision (FP16 / BF16 / FP8) | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Larger block size / `__launch_bounds__` | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Persistent kernel / Grid-stride loop | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Thread Block Cluster (Hopper) | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| ... | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |

**Decision rationale per version:**
- **v1:** _<strategy selection rationale and expected gain>_
- **v2:** _<strategy selection rationale and expected gain>_
- **v3:** _<strategy selection rationale and expected gain>_
- ...
---

## Best Version Conclusion

**Best version:** `v<N>` — execution time reduced from `<v0>` ms to `<vN>` ms, speedup `<×>`.
Key gains: `<primary optimization strategies>`.
Stopping reason: `<max iterations reached / performance target met / bottleneck saturated>`.

**Remaining optimization opportunities:** _<potential improvements for the next round, or N/A>_
