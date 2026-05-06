---
name: triton
description: Triton optimization strategies by bottleneck type. Assumes bottleneck has been classified by profiling/PROFILING.md.
---

# triton-skill

## Directory Structure

```
triton/
├── TRITON.md
└── reference/
    └── triton-opt.md
```

## Memory-Bound

**Optimization priority:**
1. Co-tune tile size and `num_warps` to prioritize improved coalesced access and cache reuse
2. Ensure contiguous access (coalesced) + alignment hints (`tl.multiple_of` / `tl.max_contiguous`)
3. Use `mask` for boundary handling to avoid warp divergence from branches
4. Use swizzle / grouped ordering to improve L2 hit rate
5. Operator fusion (e.g., matmul + epilogue, fused norm/softmax) to reduce DRAM round-trips

> Detailed entries → `reference/triton-opt.md`: Memory Access Optimization · Parallelism & Grid Strategy · Engineering & Diagnostics

---

## Compute-Bound

**Optimization priority:**
1. Core computation uses `tl.dot` to ensure Tensor Core/MMA path
2. Tune `BLOCK_M/N/K` to improve arithmetic intensity and reduce invalid instructions
3. Set accumulation precision and data path appropriately (follow upstream dtype, do not arbitrarily reduce precision)
4. Instruction-level optimization (`exp2`, `rsqrt`, FMA-friendly expressions) and control register pressure
5. Use Roofline to determine if further compute-side optimization is needed

> Detailed entries → `reference/triton-opt.md`: Block & Tile Size Tuning · Compute-level Optimization · Register & Occupancy Management · Engineering & Diagnostics

---

## Latency-Bound

**Optimization priority:**
1. Adjust `num_stages` for software pipelining to hide global memory latency
2. Optimize load/compute/store order to strengthen memory/compute overlap
3. Use persistent kernel / split-K / well-designed grid to improve parallel coverage
4. Balance registers and occupancy to avoid spill-induced long scoreboard stalls
5. Use NCU stall metrics and warp state to locate sync and scheduling bottlenecks

> Detailed entries → `reference/triton-opt.md`: Pipelining & Async · Parallelism & Grid Strategy · Register & Occupancy Management · Engineering & Diagnostics

---

## General Principles

- **Correctness before optimization**: pass correctness check every iteration before measuring performance
- **Autotune must have bounds**: recommend 5-10 candidate configurations to avoid slow first-run tuning
- **Tune for target hardware**: A100/H100/consumer GPU optimal configs usually differ
- **Metrics-driven, not intuition-driven**: every optimization must include NCU evidence (summary + details)
- **Prioritize end-to-end gains**: single-kernel local improvements need full benchmark validation
