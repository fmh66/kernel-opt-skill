# CUDA Kernel Latency Optimization (Latency-Bound)

---

## Occupancy & Launch Configuration

### Launch Configuration Tuning

Block size directly affects occupancy and hardware utilization. Common choices are 128/256/512, but the optimal value depends on the kernel's resource consumption. The CUDA Occupancy Calculator and `cudaOccupancyMaxPotentialBlockSize` API can assist in making the decision.

### Control Register Usage

The more registers each thread uses, the fewer warps can reside simultaneously (lower occupancy), and the weaker the scheduler's ability to hide latency. Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` to hint the compiler to control register allocation.

### Register Spilling

When registers are insufficient, the compiler spills variables to local memory (actually global memory, cached in L1/L2). Heavy spilling causes severe performance cliffs. Mitigate by reducing active variable count, shrinking loop unroll factor, or splitting the kernel. Use `--ptxas-options=-v` to inspect register and spill statistics.

### Occupancy Is Not Always Better When Higher

Higher occupancy means more warps available to hide latency, but also means fewer registers and shared memory available per thread. For compute-intensive kernels, a moderate reduction in occupancy in exchange for more registers (fewer spills, higher ILP) often yields better performance. Find the optimal balance through empirical testing.

---

## ILP (Instruction-Level Parallelism) Improvement

### Increase Per-Thread Independent Work

Have each thread process multiple data elements (thread coarsening), completing more computation in registers before writing back. This gives the scheduler more independent instructions to issue while a single warp is waiting, improving ILP.

### Loop Unrolling

Use `#pragma unroll` or `#pragma unroll N` to unroll loops. Unrolling reduces loop control instructions (comparisons, jumps) while exposing more independent instructions to the scheduler, improving ILP.

### Software Pipelining

Overlap the computation of the current iteration with the data prefetch for the next iteration within a loop body, maximizing functional unit utilization.

---

## Synchronization Optimization

### Reduce `__syncthreads()` Count

The most direct approach. If the data access pattern within a warp naturally has no cross-warp dependency, the sync is redundant. Audit every `__syncthreads()` call to confirm its necessity.

### Warp-level Sync Instead of Block-level Sync

The 32 threads within a warp naturally execute in lockstep (after Volta with independent thread scheduling, warp-level primitives are still valid). Using `__syncwarp()` instead of `__syncthreads()` reduces the sync granularity from the entire block to a single warp, dramatically lowering overhead.

- **Note (semantic boundary)**: Under Volta+ independent thread scheduling, "natural lockstep" should not be assumed as an implicit sync guarantee; intra-warp cooperation should still use the correct mask and explicit sync points to ensure visibility and convergence.

### Warp Shuffle

Data exchange between threads within a warp requires no shared memory, has no bank conflict issues, and has extremely low latency. Suitable for reduction, prefix sum, and broadcast patterns. This directly eliminates the "write shared → sync → read shared" three-step overhead.

### Cooperative Groups

The cooperative group mechanism introduced in CUDA 9 allows defining thread groups of arbitrary granularity and syncing within that group — for example, syncing only a tile of 8 threads — avoiding unnecessary full-block waits.

### `cuda::barrier` / `cuda::pipeline` (Ampere+)

In async copy and multi-stage pipelines, use explicit stage synchronization instead of "empirical synchronization". The core idea is to clearly specify producer-consumer commit/wait points, avoiding intermittent errors and performance jitter.

---

## Async Prefetch

### `cp.async` Prefetch

(CUDA 11+) Global → Shared Memory transfer is handled by hardware DMA, consuming no registers or compute units, and can overlap completely with computation. Combined with multi-stage buffers, this enables software pipelining that greatly hides global memory latency.

### `cudaMemPrefetchAsync`

In Unified Memory scenarios, proactively trigger page migration to avoid the high latency of on-demand page faults.

---

## Reduce Scheduling Overhead

### CUDA Graphs

When the kernel chain structure is stable and executed repeatedly, Graphs can reduce CPU submission and launch overhead, especially in dense small-kernel scenarios. Evaluate capture/update costs for dynamic graph scenarios.

---

## NCU Validation Checklist

Latency-Bound optimizations should include at least the following validations:

- **Sync wait reduction**: watch whether `Stall Barrier` and related wait stalls decrease.
- **Scheduling issuability**: watch whether `Eligible Warps Per Cycle` improves.
- **Occupancy change**: watch `Achieved Occupancy` combined with kernel latency to judge improvement.
- **Register spilling**: use `--ptxas-options=-v` + NCU to check if spills decrease.
- **Overall benefit**: final judgment by kernel latency (avg/median), not just individual sub-metrics.

Common misdiagnoses:

- Reducing `__syncthreads()` count while introducing data visibility errors.
- Only seeing occupancy rise without watching kernel latency — occupancy can go up while performance gets worse.
- Only seeing a single stall metric decrease without checking overall kernel latency and correctness.

Optimization entry point: `cuda/CUDA.md` — optimization priorities organized by bottleneck type.
