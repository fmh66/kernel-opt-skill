---
name: cuda
description: CUDA optimization strategies by bottleneck type. Assumes bottleneck has been classified by profiling/PROFILING.md.
---

# cuda-skill

## Directory Structure

```
cuda/
├── CUDA.md
└── reference/
    ├── compute-opt.md
    ├── latency-opt.md
    └── memory-opt.md
```

## Memory-Bound

**Optimization priority:**
1. Kernel Fusion — eliminate Global Memory round-trips; keep intermediates in registers
2. Coalesced access + SoA layout + vectorization (`float4/int4`)
3. Shared Memory Tiling + Bank Conflict elimination (padding / swizzle)
4. `cp.async` + double-buffering / multi-stage pipeline
5. `__ldg()` / `const __restrict__` / L2 Persistence (CC 8.0+)
6. Pinned Memory + CUDA Stream pipeline

> Detailed entries → `reference/memory-opt.md`

---

## Compute-Bound

**Optimization priority:**
1. Tensor Core / WMMA / MMA PTX — first choice for matrix kernels
2. FMA (`__fmaf_rn()`) + strength reduction (`rsqrtf` / shifts) + `--use_fast_math`
3. Eliminate branch divergence: predication / select instructions / rearrange data by warp / `__all_sync()` early exit
4. `#pragma unroll` + loop transformations (split / merge / interchange) + software pipelining

> Detailed entries → `reference/compute-opt.md`

---

## Latency-Bound

**Optimization priority:**
1. Tune block size (128 / 256 / 512 empirical testing) + `__launch_bounds__`
2. Warp Shuffle instead of Shared Memory three-step sync (write → sync → read)
3. `__syncwarp()` instead of `__syncthreads()` / Cooperative Groups minimum sync group
4. `cp.async` prefetch + increase per-thread independent work (ILP)
5. `--ptxas-options=-v` to check register spilling → reduce active variables / split kernel
6. CUDA Graphs — for dense small-kernel scenarios to reduce CPU launch overhead

> Detailed entries → `reference/latency-opt.md`

---

## General Principles

- **Occupancy is not always better when higher**: for Compute-Bound kernels, lower occupancy for more registers often yields faster kernels; use measured latency as the final criterion
- **Correctness before optimization**: pass correctness check each iteration before measuring performance
- **`--use_fast_math` requires caution**: may introduce precision issues; must re-validate numerics after enabling
- **Metrics-driven, not intuition-driven**: each optimization must include NCU evidence (summary + details)
- **Prioritize end-to-end gains**: single-kernel local improvements need validation with full benchmark
