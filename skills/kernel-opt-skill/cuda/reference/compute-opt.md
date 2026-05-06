# CUDA Kernel Compute Optimization

---

## Tensor Core / Specialized Hardware

### Tensor Core Acceleration

From the Volta architecture onwards, Tensor Core can complete a small matrix multiply-accumulate operation (e.g., 16×16×16 FP16 MMA) in one clock cycle, accessed via the WMMA API, MMA PTX instructions, or libraries like cuBLAS/CUTLASS. Compute throughput is an order of magnitude higher than ordinary CUDA Cores. Not using Tensor Core for matrix operations essentially wastes more than half the chip's compute capacity.

- **Note (architecture-specific)**: "one clock cycle" is more of a conceptual description; actual throughput depends on architecture, data type, tile shape, issue cadence, and register/pipeline occupancy.
- **Note (data types)**: TF32/BF16 paths are common on Ampere and newer; FP8 paths are common on Hopper and newer. Select based on error budget, accumulation precision, and measured conversion overhead.
- **Note (Hopper+)**: Consider the WGMMA (warp-group MMA) path, which typically requires a stricter pipeline/sync setup.

### SFU (Special Function Unit)

GPUs have dedicated special function units for transcendental functions like sin, cos, rsqrt, log2, and exp2. Calling the corresponding intrinsics (e.g., `__sinf`) goes through the SFU pipeline, which runs in parallel with the regular ALU. However, SFU throughput is lower than ALU, so heavy use of transcendental functions can become a bottleneck.

### Integer Operations and Bit Operations

Integer multiplication and division are expensive on GPU. For constant divisors, the compiler automatically converts division/modulo to multiply+shift; for variable divisors, manually use bit operations (e.g., use shift and mask for powers of 2). Bit operation intrinsics like `__popc()`, `__clz()`, and `__ffs()` map directly to hardware instructions and complete in a single cycle.

---

## Warp-level Compute Primitives

### Warp Shuffle for Computation

`__shfl_sync`, `__shfl_xor_sync`, `__shfl_down_sync`, etc. are not just data exchange tools — they are compute primitives. Warp-level reduction, scan/prefix sum, and broadcast can all be done at the register level with shuffle, much faster than going through shared memory, with zero bank conflicts.

### Warp Vote Functions

`__ballot_sync()`, `__all_sync()`, `__any_sync()` can collect the condition results of an entire warp in a single instruction. Commonly used to quickly determine whether all/none of the threads in a warp satisfy a condition, enabling warp-level early exit to skip unnecessary computation.

### Warp Match Functions (Volta+)

`__match_any_sync()` and `__match_all_sync()` identify which threads in a warp hold the same value, enabling grouped processing and deduplication without using shared memory for scatter-gather.

---

## Loop Optimizations

### Loop Unrolling

Use `#pragma unroll` or `#pragma unroll N` to tell the compiler to unroll loops. Unrolling reduces loop control instructions (comparisons, jumps) and, more importantly, exposes more independent instructions to the scheduler, improving ILP. Highly effective for small loops with a known, bounded iteration count.

### Loop Splitting

Split a loop containing multiple unrelated computations into multiple independent loops. Each loop has lower register pressure, is easier for the compiler to optimize, and more amenable to vectorization.

### Loop Merging

Conversely, merge multiple loops traversing the same data into one, allowing data to be reused in registers and reducing repeated global memory accesses.

### Loop Interchange

Adjust the order of nested loops so that the innermost loop accesses memory consecutively, improving coalescing characteristics.

### Software Pipelining (Compute)

Overlap the computation of the current iteration with the data prefetch for the next iteration within a loop body. A single iteration is divided into "load → compute → store" phases, with different phases of different iterations interleaved to maximize functional unit utilization.

---

## Algorithm-level Compute Optimization

### Reduction Optimization

Classic parallel reduction progresses from naive interleaved addressing to sequential addressing, warp unrolling, and warp shuffle reduction, with measurable gains at each step. Final form: warp-level shuffle reduction, cross-warp shared memory reduction, cross-block atomic or secondary kernel reduction.

### Scan / Prefix Sum Optimization

The Blelloch algorithm (work-efficient scan) and the Hillis-Steele algorithm (step-efficient scan) each have their use cases. Large-scale scans are typically done in three phases: block-level scan, auxiliary array scan across blocks, then fill-back.

### Replace Division with Multiplication

Floating-point division throughput is far lower than multiplication. `a / b` can become `a * __frcp_rn(b)` or `a * (1.0f / b)`, letting the compiler use reciprocal approximation + multiply. `__fdividef(a, b)` is a similar fast path.

### Use FMA Instead of Separate Multiply-Add

`a * b + c` should compile to a single FMA (Fused Multiply-Add) instruction, but compilers sometimes won't fuse automatically for precision reasons. Use `__fmaf_rn()` or `fmaf()` explicitly to ensure a single instruction is used; this also provides higher precision (no intermediate truncation).

### Lookup Table (LUT)

For complex functions with a limited input range, precompute results in shared memory or constant memory and look them up with a single read instead of heavy computation. Especially suitable for discrete mappings or piecewise functions.

### Strength Reduction

Replace computations with cheaper equivalents: use shifts instead of power-of-2 multiplications/divisions, accumulate additions instead of per-iteration multiplications (`i*stride` → accumulate `stride`), use `rsqrtf()` instead of `1.0f/sqrtf()` (rsqrt is a single native hardware instruction).

---

## Compiler Optimization Control

### Compiler Option Tuning

`--use_fast_math` enables all fast math (including unsafe optimizations like flush-denormals-to-zero). For partial optimization, use individual flags: `--ftz=true` (flush denormals to zero), `--prec-div=false` (low-precision division), `--prec-sqrt=false` (low-precision square root).

### Inlining Control

`__forceinline__` forces inlining of small functions, eliminating function call overhead. `__noinline__` prevents inlining to reduce instruction cache pressure from code bloat.

### PTX / SASS Analysis

Use `cuobjdump --dump-sass` to inspect the final machine code and confirm that critical paths use efficient instructions like FMA, LDG, and STS as expected, and check for unexpected type conversions or register spills. `ncu` (Nsight Compute) can do finer-grained instruction-level analysis.

- **Note (recent practice)**: When working with async transfers or new-architecture optimizations, check the PTX layer for the expected async/matrix-related instruction paths, to avoid "code written but compilation degraded" situations.

### `__restrict__` Keyword

Tells the compiler that pointers do not alias each other, allowing more aggressive instruction reordering and register caching. The effect can sometimes be significant.

---

## Branch and Control Flow Optimization

### Predication

For very short branches (e.g., one or two instructions), the compiler automatically converts the branch to predicate instructions — both paths execute, but only the one satisfying the condition writes back results. This avoids warp divergence but wastes computation. Only suitable when branch bodies are very short.

### Select Instructions Instead of Branches

`condition ? a : b` is typically compiled to a branchless select instruction. Manually rewriting if-else as ternary operators or arithmetic expressions (e.g., `result = mask * a + (1-mask) * b`) can help the compiler generate branchless code.

### Rearrange Data by Warp

When branches are unavoidable, pre-sort or group data so that similar data resides in the same warp. For example, in particle simulations, separate active and inactive particles so that warps containing only inactive particles can skip computation entirely.

### Early Exit

In reduction or search kernels, if all threads in a warp have already met the termination condition, use `__all_sync()` to detect this and return early, avoiding unnecessary subsequent computation.

---

## Async Compute and Overlap

### Compute/Memory Overlap

This is the scheduler's core capability. When a warp is waiting for memory, the scheduler switches to another ready warp to execute compute instructions. Methods to improve this overlap: increase occupancy, increase the number of independent instructions per thread (ILP), and use software pipelining.

### Warp Specialization

Divide the warps in a block into "transport warps" and "compute warps". Transport warps handle global→shared memory data loading; compute warps handle arithmetic; they coordinate through shared memory and simple flags. This is a manual producer-consumer model that can achieve better resource utilization than traditional approaches for matrix multiplication-type kernels.

- **Note (architecture evolution)**: On newer architectures like Hopper/Blackwell, this producer-consumer division is often paired with deeper async pipelines for more consistent gains, but correctness of sync semantics is more demanding.

---

## NCU Validation Checklist

To avoid "changing code without checking metrics", compute optimizations should include at least the following checks:

- **Tensor Core path**: confirm that expected MMA/WGMMA instruction paths appear and compute pipeline utilization improves.
- **Instruction efficiency**: watch whether `Issue Slot Utilization` and `Eligible Warps Per Cycle` improve.
- **Branch quality**: watch whether `Warp Execution Efficiency` and branch-related stalls improve.

Common misdiagnoses:

- Only seeing occupancy rise without watching kernel latency — occupancy can go up while performance gets worse.
- Only seeing one metric improve while sync or memory path degrades, causing overall slowdown.

Optimization entry point: `cuda/CUDA.md` — optimization priorities organized by bottleneck type.
