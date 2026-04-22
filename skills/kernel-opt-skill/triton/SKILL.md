---
name: triton
description: Triton optimization strategies by bottleneck type. Assumes bottleneck has been classified by profiling/SKILL.md.
---

# triton-skill

## 目录结构

```
triton/
├── SKILL.md
└── reference/
    └── triton-opt.md
```

## Memory-Bound

**优化优先级**：
1. Tile 尺寸与 `num_warps` 联调，优先提升合并访存与缓存复用
2. 保证连续访问（coalesced）+ 对齐提示（`tl.multiple_of` / `tl.max_contiguous`）
3. 用 `mask` 处理边界，避免分支导致的 warp divergence
4. 使用 swizzle / grouped ordering 提升 L2 命中率
5. 做算子融合（如 matmul + epilogue、fused norm/softmax），减少显存往返

> 详细条目 → `reference/triton-opt.md`：内存访问优化 · 并行与 Grid 策略 · 工程化与诊断手段

---

## Compute-Bound

**优化优先级**：
1. 核心计算使用 `tl.dot`，确保走 Tensor Core/MMA 路径
2. 调优 `BLOCK_M/N/K`，提高算术强度并减少无效指令
3. 合理设置累加精度与数据路径（遵循上层 dtype，不擅自降精度）
4. 指令级优化（`exp2`、`rsqrt`、FMA 友好表达式）并控制寄存器压力
5. 结合 Roofline 判断是否仍需继续向计算侧优化

> 详细条目 → `reference/triton-opt.md`：Block 与 Tile 尺寸调优 · 计算层面的优化 · 寄存器与占用率管理 · 工程化与诊断手段

---

## Latency-Bound

**优化优先级**：
1. 调整 `num_stages` 做软件流水线，隐藏 global memory latency
2. 优化 load/compute/store 顺序，增强访存与计算重叠
3. 用 persistent kernel / split-K / 合理 grid 提升并行覆盖
4. 控制寄存器与 occupancy 平衡，避免 spill 导致长记分牌 stall
5. 通过 NCU 的 stall 指标与 warp 状态定位同步和调度瓶颈

> 详细条目 → `reference/triton-opt.md`：流水线与异步化 · 并行与 Grid 策略 · 寄存器与占用率管理 · 工程化与诊断手段

---

## 通用原则

- **先正确再优化**：每轮迭代都先过 correctness，再看性能收益
- **Autotune 要有边界**：候选配置建议 5-10 组，避免首轮调优过慢
- **按目标硬件调优**：A100/H100/消费级卡最优配置通常不同
- **指标驱动而非直觉驱动**：每次优化都要附 NCU 证据（summary + details）
- **优先端到端收益**：单 kernel 局部提升需用完整 benchmark 验证
