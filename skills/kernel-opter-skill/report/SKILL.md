---
name: kernel-opter-report
description: Generate a structured CUDA kernel optimization report. Invoke after each optimization round to surface per-step decisions, NCU metrics, and conclusions for Step 0–6 as a Markdown document.
---

# kernel-opter-report

## 职责

在每轮优化结束后（或用户请求时），按以下模板生成报告。**每个「决策」字段必须填写**，不得留空占位符——这是本 skill 的核心约束。

---

## 生成步骤

1. 从上下文（对话历史 / profiling sub-skill 输出 / opt-loop sub-skill 记录）中收集各步骤数据。
2. 对每个「决策」字段，基于收集到的数据写出**具体结论**，禁止输出 `_<填写>_` 占位符。
3. 若某步骤数据缺失，在对应单元格注明 `N/A — <缺失原因>`，并在决策字段说明影响。
4. 输出完整 Markdown，可直接粘贴至文档或 PR 描述。

---

## 报告模板

```markdown
# CUDA 优化报告  —  <kernel_name>  (<date>)

## 环境

| 项目 | 值 |
|---|---|
| GPU | <name> (CC <x.y>) |
| CUDA | <version> |
| PyTorch | <version> |
| 内核文件 | <path> |

---

## Step 0 · 基线采集

| 指标 | 值 |
|---|---|
| 执行时间 (ms) | |
| 吞吐量 (TFLOPS / GB/s) | |
| NCU 采集命令 | `ncu --set full -o baseline ...` |

**决策：** 基线数据是否可重复？若 CV > 5% 需先固定频率再继续。
→ _<必填：结论>_

---

## Step 1 · 全局定位（瓶颈类型）

| NCU 指标 | 值 | 阈值 | 判断 |
|---|---|---|---|
| Memory Throughput % | | >70% → Memory-Bound | |
| Compute Throughput % | | >70% → Compute-Bound | |
| SM Active Cycles % | | <40% → Latency-Bound | |
| Roofline 位置 | | | |

**决策：** 主瓶颈 = `Memory-Bound / Compute-Bound / Latency-Bound`（三选一）
→ _<必填：结论及依据>_

---

## Step 2 · 针对性优化

> 根据 Step 1 结论，仅填写对应分支，其余分支可删除。

### 2a · Memory-Bound（内存访问优化）

| 策略 | 是否采用 | 理由 |
|---|---|---|
| 合并全局内存访问（128B 对齐） | ✓ / ✗ | |
| Shared Memory tiling | ✓ / ✗ | |
| `__ldg` / L2 持久化 | ✓ / ✗ | |
| `cp.async` 异步预取 | ✓ / ✗ | |
| 向量化加载（`float4`） | ✓ / ✗ | |

**决策：** 本轮选择的策略组合及预期收益。
→ _<必填>_

### 2b · Compute-Bound（计算效率优化）

| 策略 | 是否采用 | 理由 |
|---|---|---|
| Tensor Core（`wmma` / `mma`） | ✓ / ✗ | |
| 指令级并行（ILP，展开循环） | ✓ / ✗ | |
| 混合精度（FP16 / BF16 / FP8） | ✓ / ✗ | |
| 减少 `__syncthreads` 频率 | ✓ / ✗ | |

**决策：** 本轮选择的策略组合及预期收益。
→ _<必填>_

### 2c · Latency-Bound（并行度优化）

| 策略 | 是否采用 | 理由 |
|---|---|---|
| 增大 block size | ✓ / ✗ | |
| Grid-stride loop | ✓ / ✗ | |
| Persistent kernel | ✓ / ✗ | |
| Thread Block Cluster (Hopper) | ✓ / ✗ | |

**决策：** 本轮选择的策略组合及预期收益。
→ _<必填>_

---

## Step 3 · 占用率分析

| 指标 | 值 | 目标 |
|---|---|---|
| 理论占用率 (%) | | ≥ 50% |
| 实测活跃 Warp / SM | | |
| 寄存器/线程 | | ≤ 128 |
| Shared Memory/block (KB) | | ≤ 上限 |
| 限制因子 | `registers / smem / block_size` | |

**决策：** 是否需要调整 `__launch_bounds__` 或 block 大小？
→ _<必填>_

---

## Step 4 · Warp 调度分析

| NCU 指标 | 值 | 解读 |
|---|---|---|
| Warp Stall — Long Scoreboard (%) | | 全局内存延迟 |
| Warp Stall — Short Scoreboard (%) | | Shared Memory bank conflict |
| Warp Stall — Sync (%) | | 同步屏障过多 |
| Warp Stall — Wait (%) | | 计算依赖链长 |

**决策：** 主要 stall 原因及对应解法。
→ _<必填>_

---

## Step 5 · 分支发散分析

| NCU 指标 | 值 | 阈值 |
|---|---|---|
| Branch Divergence (%) | | < 5% 可接受 |
| 热点分支位置（行号） | | |

**决策：** 是否需要重写条件判断或使用 `__ballot_sync`？
→ _<必填>_

---

## Step 6 · 结果对比

| 指标 | 基线 | 本轮 | 提升 |
|---|---|---|---|
| 执行时间 (ms) | | | |
| 吞吐量 | | | |
| Memory Throughput % | | | |
| Compute Throughput % | | | |
| 占用率 (%) | | | |

**决策：** 性能是否达标？
- [ ] 达标 → 优化完成
- [ ] 未达标 → 返回 Step 1，下一轮聚焦：_<必填：新方向>_

---

## 历轮迭代摘要

| 轮次 | 主策略 | 执行时间 (ms) | 备注 |
|---|---|---|---|
| 0（基线） | — | | |
| 1 | | | |
| 2 | | | |
| best | | | |
```
