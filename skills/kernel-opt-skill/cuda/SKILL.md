---
name: cuda
description: CUDA optimization strategies by bottleneck type. Assumes bottleneck has been classified by profiling/SKILL.md.
---

# cuda-skill

## 目录结构

```
cuda/
├── SKILL.md
└── reference/
    ├── compute-opt.md
    ├── latency-opt.md
    └── memory-opt.md
```

## Memory-Bound

**优化优先级**：
1. Kernel Fusion — 消除 Global Memory 往返，中间结果留在寄存器
2. 合并访问 + SoA 布局 + 向量化（`float4/int4`）
3. Shared Memory Tiling + Bank Conflict 消除（padding / swizzle）
4. `cp.async` + 双缓冲多级流水线
5. `__ldg()` / `const __restrict__` / L2 Persistence（CC 8.0+）
6. Pinned Memory + CUDA Stream 流水线

> 详细条目 → `reference/memory-opt.md`

---

## Compute-Bound

**优化优先级**：
1. Tensor Core / WMMA / MMA PTX — 矩阵类 kernel 首选
2. FMA（`__fmaf_rn()`）+ 强度削减（`rsqrtf` / 移位）+ `--use_fast_math`
3. 消除分支发散：谓词化 / select 指令 / 按 warp 重组数据 / `__all_sync()` early exit
4. `#pragma unroll` + 循环变换（分裂 / 合并 / 交换）+ 软件流水线

> 详细条目 → `reference/compute-opt.md`

---

## Latency-Bound

**优化优先级**：
1. 调整 block size（128 / 256 / 512 实测）+ `__launch_bounds__`
2. Warp Shuffle 替代 Shared Memory 三步同步（写 → sync → 读）
3. `__syncwarp()` 替代 `__syncthreads()` / Cooperative Groups 最小同步组
4. `cp.async` 预取 + 增加每线程独立工作量（ILP）
5. `--ptxas-options=-v` 检查寄存器溢出 → 缩减活跃变量 / 拆分 kernel
6. CUDA Graphs — 小 kernel 密集场景降低 CPU launch 开销

> 详细条目 → `reference/latency-opt.md`

---

## 通用原则

- **Occupancy 不是越高越好**：Compute-Bound 时降 occupancy 换更多寄存器往往更快，以实测 latency 为准
- **先正确再优化**：每轮迭代先过 correctness，再看性能收益
- **`--use_fast_math` 须谨慎**：可能引入精度问题，开启后必须重新验证数值正确性
- **指标驱动而非直觉驱动**：每次优化都要附 NCU 证据（summary + details），targeted 只作初步定位
- **优先端到端收益**：单 kernel 局部提升需用完整 benchmark 验证
