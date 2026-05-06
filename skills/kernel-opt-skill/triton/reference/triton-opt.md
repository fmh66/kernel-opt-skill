# Triton Kernel Optimization Guide

This document systematically covers Triton kernel optimization methodology across eight categories, from tile size tuning to engineering diagnostics. Each category is presented from the perspectives of principles, techniques, and key points, serving as a reference for writing and tuning Triton operators.

---

## Block and Tile Size Tuning

### Principles

Triton's programming model is "each program processes one tile." Tile size (e.g., `BLOCK_M`, `BLOCK_N`, `BLOCK_K`) is the most critical performance parameter, directly determining four things:

- **Parallelism**: grid size equals problem size divided by block size. Larger blocks mean a smaller grid and potentially lower SM occupancy; smaller blocks mean more programs, increasing launch and scheduling overhead.
- **Register pressure**: intermediate results within a block reside in registers. Excessively large blocks cause register spills to local memory (actually in DRAM), causing severe performance degradation.
- **Memory access efficiency**: tiles too small fail to saturate memory bandwidth per load; tiles too large reduce L1/L2 cache hit rates.
- **Arithmetic intensity**: for matrix multiplication, larger tiles mean higher data reuse and arithmetic intensity (FLOPs/Bytes), making it easier to transition from memory-bound to compute-bound.

### Tuning Techniques

- **Use `triton.autotune`**: provide multiple candidate `triton.Config` entries; Triton automatically benchmarks on first run to select the optimal one, caching by input shape via the `key` parameter.
- **Tune by shape range**: large matrices prefer large tiles (e.g., 128×256); small matrices or batched small GEMM prefer small tiles (e.g., 32×32). One config cannot serve all shapes.
- **Co-tune `num_warps`**: commonly 4 or 8, determining the number of threads per program and thus the per-thread register count and parallelism granularity.
- **Co-tune `num_stages`**: controls software pipeline depth, typically 2–5, limited by shared memory capacity.

### Key Points

- Tile sizes are generally powers of 2 (16, 32, 64, 128, 256) to align with warps (32 threads) and hardware vectorization instructions.
- Monitor `ncu` output for register usage and local memory usage; reduce tiles or increase `num_warps` immediately upon detecting spills.
- More autotune candidates is not always better — too large a set makes the first run extremely slow. Filter to 5–10 commonly used configs based on experience.
- Optimal configs differ significantly across hardware (A100, H100, consumer GPUs); tune separately for the target hardware.

---

## Memory Access Optimization

DRAM bandwidth is typically the primary bottleneck of a kernel; memory-level optimizations often yield the largest gains.

### Coalesced Access

The GPU DRAM controller transfers one contiguous 128-byte sector at a time. If 32 threads in a warp access contiguous addresses, they can be coalesced into a small number of transactions; strided access creates redundant transfers and wasted bandwidth.

In Triton, the key to ensuring coalesced access is: **keep the innermost dimension contiguous in memory**. When loading a 2D tile, place the stride-1 dimension at the `[None, :]` position, so threads within the same warp expand along contiguous addresses.

### Alignment and Vectorization Hints

The Triton compiler generates wide instructions like `ld.global.v4` and `ld.global.v2` based on pointer attributes. Use the following to help the compiler identify alignment:

- **`tl.multiple_of(x, n)`**: tells the compiler that `x` is a multiple of `n`.
- **`tl.max_contiguous(x, n)`**: tells the compiler the maximum length of contiguous elements in `x`.
- **Function argument annotation**: constraining certain strides to 1 via `tl.constexpr` can trigger better code generation paths.

These hints are especially important for irregular shapes — they determine whether the compiler dares to use 128-bit wide loads.

### Mask for Boundary Handling

For non-divisible shapes, always use the `mask=` parameter to handle tails instead of writing `if` branches. This keeps the main loop path uniform, avoids warp divergence, and fills out-of-bounds positions with neutral values via `other=` (0 for addition, 1 for multiplication, -inf for max).

### Shared Memory Reuse

Triton does not require manual shared memory management, but through proper tile shapes and loop structures, the compiler automatically places reused data in shared memory. For example, in matrix multiplication looping along the K dimension, both A and B tiles are cached for multiple reuses, improving arithmetic intensity.

### Avoid Bank Conflicts

Shared memory is divided into 32 banks; multiple threads in a warp accessing different addresses in the same bank simultaneously causes serialization. Ways to avoid bank conflicts:

- Use validated tile shapes (e.g., 128×128×32 from the matmul tutorial).
- For transpose-type kernels, avoid the naive `BLOCK_M == BLOCK_N == 32` pattern; mitigate via padding to 33 or let the compiler auto-swizzle.

### Reduce Redundant Loads

- Move loop-invariant loads (e.g., bias, scale coefficients) outside the loop to avoid repeated transfers each iteration.
- Cache reusable intermediate results in register arrays instead of repeatedly reading from DRAM.
- For read-only tensors accessed multiple times, consider proactively prefetching into shared memory.

---

## Pipelining and Async

### Software Pipelining (num_stages)

The core idea of software pipelining is: **while computing iteration i, asynchronously issue the load for iteration i+1 (or further ahead)**, so memory access latency is hidden by computation.

`num_stages=N` means the compiler maintains N shared memory buffers in rotation. Selection principles:

- **Ampere (A100)**: typically 3–4 stages is optimal.
- **Hopper (H100)**: can reach 5–6 stages with TMA.
- **Too many stages**: insufficient shared memory leads to lower occupancy or spills.

Software pipelining only applies to kernels with a stable load-compute pattern in the loop body; single-pass kernels (e.g., elementwise) do not need it.

### Async Copy and TMA

On newer hardware, Triton automatically generates async copy instructions:

- **Ampere's `cp.async`**: implements async global → shared transfer without blocking compute units.
- **Hopper's TMA (Tensor Memory Accelerator)**: a dedicated hardware unit for tensor transfers with lower latency, higher efficiency, and support for multi-dimensional addressing.

What users need to do:

- Ensure tile size and alignment meet async copy requirements (typically 4/8/16-byte alignment).
- Use a recent version of Triton so it recognizes the hardware and follows the optimal lowering path.

### Load / Compute / Store Ordering

The order of handwritten code affects the dependency graph the compiler generates. General recommendation:

1. Issue all needed loads first.
2. Compute on data that has arrived.
3. Store results in one batch at the end.

This lets the compiler construct as deep a pipeline as possible, maximizing memory/compute overlap.

---

## Compute-level Optimization

### Use `tl.dot` for Tensor Core

The core matrix multiply operation must use `tl.dot`, which is lowered to MMA / WMMA / WGMMA instructions with throughput over 10x that of scalar FMA.

Key points:

- Shape must meet the minimum granularity of hardware MMA (typically 16×16×16).
- Accumulator should stay fp32 to avoid precision loss from error accumulation.
- Strictly follow the data type specified by the upstream caller; do not arbitrarily reduce precision. Whether fp16/bf16/fp8 can be used is a framework/model-level decision — kernel authors should not override it.

### Operator Fusion

Fusion is one of Triton's biggest advantages over calling libraries. Merging multiple elementwise or reduction operations into a single kernel dramatically reduces DRAM round-trips. Typical examples:

- **Fused LayerNorm / RMSNorm**: one kernel computes mean, variance, normalization, and affine.
- **Fused Softmax**: one kernel computes max, exp, sum, and division.
- **Fused Attention (FlashAttention)**: fuses QK^T, softmax, and ×V together; uses online softmax to avoid materializing the N×N attention matrix.
- **Matmul + Epilogue**: handles bias, activation, dropout, and residual add immediately after `tl.dot`.

The cost of fusion is increased kernel complexity and register pressure, requiring tradeoffs at the scheduling level.

### Instruction-level Tricks

- **`tl.exp2` instead of `tl.exp`**: hardware has a dedicated fast instruction for `exp2`; precompute `x/ln 2` and fold it into upstream computation for softmax.
- **`rsqrt` instead of `1 / sqrt`**: commonly used in LayerNorm and RMSNorm; eliminates one division.
- **Explicit FMA usage**: the compiler usually fuses multiply-add automatically, but explicit use is more reliable in complex expressions.
- **Avoid integer division and modulo**: use bitwise operations or precomputed strides instead, especially in inner loops.

### Numerically Stable and Fast

- **Softmax subtract max**: maintains numerical stability while allowing fast paths like `exp2`.
- **Welford single-pass statistics**: use Welford's algorithm in LayerNorm to compute mean and variance in a single pass, saving one scan.
- **Online Softmax**: the key idea in FlashAttention — use a recurrence to complete softmax in a single scan, avoiding storing intermediate matrices.

---

## Parallelism and Grid Strategy

### Grid Partitioning

The grid determines the total number of programs and their mapping to data blocks. Common choices:

- **1D grid**: suitable for reduction and elementwise.
- **2D grid**: suitable for matmul, with the two dimensions corresponding to M and N.
- **1D + internal decode**: encode `(pid_m, pid_n)` as a single `pid` to facilitate swizzle.

The rule is: grid count must be large enough for all SMs to stay busy — generally grid count ≥ SM count × 2–4 to provide scheduling slack.

### Swizzle for L2 Hit Rate

In large matrix matmul, linear `(pid_m, pid_n)` ordering causes multiple programs to simultaneously access different columns of B, reducing L2 cache hit rate. Group-major swizzle concentrates adjacent programs in an L-shaped region, sharing access to A and B, significantly improving L2 hit rate. This is a classic technique from the Triton matmul tutorial, commonly yielding 10–30% speedup for large GEMM.

### Persistent Kernel

When the grid count far exceeds the SM count, each program's launch and context switch adds overhead. A persistent kernel sets the program count equal to the SM count (or a small multiple) and loops internally over multiple tiles. Advantages:

- Reduces launch overhead and scheduling jitter.
- Reuses constants and preloaded data in registers across tiles.
- On H100, combined with warp specialization, can widen the performance gap further.

### Split-K

When K is large but M and N are small, the normal grid has only a few dozen programs, unable to fill all SMs. Split-K divides the K dimension into S parts, with S programs computing partial sums in parallel and merging via atomic adds or a secondary reduction kernel.

The cost is multiple atomic adds or extra kernel overhead, but for "small M/N large K" shapes (e.g., certain attention projections, embedding back-propagation), it is usually well worth it.

### Load Balancing

- For variable-length inputs (e.g., NLP batches), use `cu_seqlens` to record per-sample offsets, with each program handling a token block, avoiding unnecessary computation from padding.
- For sparse or irregular problems, use bucketing or bin-packing strategies so each program has approximately equal workload, avoiding "long-tail programs" dragging down the entire kernel.

---

## Register and Occupancy Management

### Register Pressure

The total number of registers per SM is fixed (e.g., A100 has 64K 32-bit registers). More registers per thread means fewer resident warps, lowering occupancy.

**Diagnosis:**

- Use `ncu --set full` to check `registers per thread` and `achieved occupancy`.
- Monitor local memory usage; non-zero usually indicates spills.
- Set `TRITON_DEBUG=1` or examine PTX output to confirm register allocation.

**Mitigation:**

- Reduce tile dimensions like `BLOCK_M / BLOCK_N`.
- Reduce `num_stages` (pipeline buffers also consume registers).
- Increase `num_warps` to distribute work across more threads.
- Split the kernel to separate unrelated computations.
- Avoid holding unnecessary intermediate tensors.

### Occupancy vs. Latency Hiding

Occupancy is not always better when higher. For compute-intensive operators (like matmul), lower occupancy + larger tiles is often faster since the pipeline and Tensor Core already absorb the latency; for memory-intensive operators, higher occupancy is needed to switch warps and hide memory latency.

Rules of thumb:

- **Matmul / Attention**: target occupancy 25–50%, maximize tile size.
- **Elementwise / Reduction**: target occupancy 50–100%, maximize memory parallelism.

### Loop Invariant Hoisting and Strength Reduction

Move loop-invariant computations (e.g., normalization coefficients, constant reciprocals) outside the loop to reduce instruction count in the inner loop. Compilers usually optimize this automatically, but explicit extraction is safer for complex expressions.

---

## Special Data Paths

This section discusses acceleration techniques using hardware special paths **without changing the original data type**. Whether to reduce precision (fp32 → fp16/bf16/fp8) is a model and framework-level decision; kernel authors must strictly use the dtype passed by the upstream caller and must not change it.

### Structured Sparsity and Block Sparsity

- **2:4 sparsity**: supported from Ampere onwards; 2 of every 4 weights are zero, achieving approximately 2x throughput via `cusparseLt` or custom Triton kernels.
- **Block-sparse**: sparse storage by blocks (e.g., blocksparse attention), skipping empty blocks via an index array. Writing block-sparse kernels in Triton is relatively straightforward — just navigate by index in the outer loop.

### Masked and Variable-length Paths

For scenarios requiring masks (attention, padding), avoid `if` branches that cause warp divergence; use `tl.where` and masked load/store instead. For causal masks, directly skip blocks that are completely masked to reduce unnecessary computation. For variable-length batches, use `cu_seqlens` to index the start and end positions of each sample, avoiding meaningless computation on padding tokens.

---

## Engineering and Diagnostics

Half of optimization is writing; the other half is measuring. Tuning blind without a profiler is guesswork.

### Autotune Diagnostics

Set the environment variable `TRITON_PRINT_AUTOTUNING=1` to print the timing for each config and the selected optimal config. This allows you to:

- Narrow the search space by eliminating clearly inferior configurations.
- Lock in optimal configs for common shapes to avoid first-run jitter.
- Detect "no good config for certain shapes" signals, indicating the kernel structure needs redesigning.

### Inspecting Intermediate Code

Set `TRITON_CACHE_DIR` to a directory; Triton will preserve intermediate compilation artifacts including:

- **TTIR (Triton IR)**: high-level IR for understanding overall structure.
- **TTGIR (Triton GPU IR)**: lets you see if layout, pipeline, and num_stages took effect.
- **LLIR (LLVM IR)**: for deep debugging.
- **PTX**: check for key instructions like `ld.global.v4`, `mma.sync`, `cp.async`.
- **cubin / metadata**: view register count and shared memory usage.

TTGIR and PTX are the two most important artifacts for performance tuning.

### Nsight Compute

When using `ncu` for fine-grained profiling, focus on the following metrics:

- **SM utilization** (`sm__throughput.avg.pct_of_peak_sustained_elapsed`): whether compute units are saturated.
- **DRAM bandwidth utilization** (`dram__throughput...`): whether the kernel is memory-bound.
- **Global load transaction count**: whether there is access redundancy or uncoalesced access.
- **Tensor Core instruction count**: whether the MMA path is actually being used.
- **Registers per thread and shared memory per block**: to identify occupancy bottlenecks.

### Roofline Analysis

The Roofline model is a fundamental tool for determining optimization direction. Compute the hardware's theoretical arithmetic intensity:

- If the kernel's arithmetic intensity is below the hardware ridge point: **memory-bound** — optimize memory coalescing, tile enlargement, fusion, and reduction of redundant transfers.
- If the kernel's arithmetic intensity is above the hardware ridge point: **compute-bound** — optimize with Tensor Core, lower precision, and instruction selection.

Do a Roofline analysis before blindly optimizing to avoid going in the wrong direction.

### Reference Benchmarks

- Compare against official implementations like cuBLAS/cuDNN/CUTLASS/FlashAttention at the same shapes to see the gap.
- Compare against the Triton kernel generated by `torch.compile` — it is often an already-autotuned baseline.
- Test coverage across multiple shapes (large square matrices, tall/skinny matrices, batched small GEMM, extreme K lengths) to avoid tuning for only a single case.

### Correctness and Stability Validation

- Use fp64 reference values for allclose checks; set `atol/rtol` appropriately for the precision type.
- Always test boundary cases: M/N/K not divisible by block size, K=0, full mask, extreme values, NaN propagation.
- Cross-architecture validation (T4, A100, H100, consumer GPUs) to avoid running correctly on only one hardware platform.

### Performance Regression and Version Management

- Autotune results should be cached and frozen, packaged with releases to avoid slow first-run for users.
- Include key kernel benchmarks in CI; catch performance regressions immediately when Triton is upgraded.
- Record real-workload shape distributions and do profile-guided optimization rather than relying on intuition.

---

## Recommended Optimization Order

These eight categories are intertwined; the recommended order for practical tuning is:

1. **Correctness first, then optimize**: write a functionally correct naive version as baseline and reference.
2. **Tune tile size and `num_warps` / `num_stages`**: use autotune to quickly reach usable performance.
3. **Refine memory access**: coalescing, vectorization, masking, swizzle.
4. **Use Tensor Core path**: for compute-intensive operators, use `tl.dot` for the core computation to map to MMA instructions; strictly follow the dtype given by the upstream caller.
5. **Operator fusion**: fold epilogue, pre/post-processing into the kernel to reduce DRAM round-trips.
6. **Grid strategy**: swizzle, persistent, split-K — choose based on shape and hardware.
7. **Profile with `ncu`**: locate the last 10–20% bottleneck.
8. **Freeze configs and regression testing**: cache autotune configs and establish benchmarks and CI monitoring.

Different operators have different emphases:

- **Matmul**: tiles and Tensor Core, swizzle and split-K.
- **Attention**: fusion and online softmax, IO complexity.
- **Elementwise / Normalization**: almost entirely memory optimization, fusion and vectorization.
- **Variable-length / Sparse**: grid strategy and load balancing.

By addressing all these dimensions and refining incrementally, Triton kernels can typically reach or match hand-written CUDA performance.

---

## Official Documentation

https://triton-lang.org/main/index.html
