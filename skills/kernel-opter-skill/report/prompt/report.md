# CUDA 优化最终报告 — `<kernel_name>` (`<date>`)

## 环境

| 项目 | 值 |
|---|---|
| GPU | `<name>` (CC `<x.y>`) |
| CUDA / nvcc | `<version>` |
| ncu | `<version>` |
| nsight-python | `<version>` |
| PyTorch | `<version>` |
| 内核文件 | `<path>` |

---

## 版本迭代对比

| 指标 | v0（基线） | v1 | v2 | v3 | ... | best |
|---|---|---|---|---|---|---|
| 执行时间 (ms) | | | | | | |
| 加速比 (×) | 1.00 | | | | | |
| Memory Throughput (%) | | | | | | |
| Compute Throughput (%) | | | | | | |
| SM Active Cycles (%) | | | | | | |
| 瓶颈判定 | | | | | | |
| 理论占用率 (%) | | | | | | |
| 活跃 Warp / SM | | | | | | |
| 寄存器 / 线程 | | | | | | |
| Shared Mem / block (KB) | | | | | | |
| Warp Stall — Long SB (%) | | | | | | |
| Warp Stall — Short SB (%) | | | | | | |
| Branch Divergence (%) | | | | | | |
| ... | | | | | | |

---

## 各版本优化策略

| 策略 | v1 | v2 | v3 | ... |
|---|---|---|---|---|
| 合并全局内存访问（128B 对齐） | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Shared Memory tiling | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| `__ldg` / L2 持久化 | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| `cp.async` 异步预取 | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| 向量化加载（`float4`） | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Tensor Core（`wmma` / `mma`） | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| ILP（循环展开） | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| 混合精度（FP16 / BF16 / FP8） | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| 增大 block size / `__launch_bounds__` | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Persistent kernel / Grid-stride loop | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| Thread Block Cluster (Hopper) | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |
| ... | ✓ / ✗ | ✓ / ✗ | ✓ / ✗ | |

**各轮决策说明：**
- **v1：** _<策略选择依据及预期收益>_
- **v2：** _<策略选择依据及预期收益>_
- **v3：** _<策略选择依据及预期收益>_
- ...
---

## 选优结论

**最优版本：** `v<N>` — 执行时间从 `<v0>` ms 降至 `<vN>` ms，加速比 `<×>`。
核心收益：`<主要优化策略>`。
停止原因：`<达到最大迭代次数 / 性能已达目标 / 瓶颈已饱和>`。

**未尽优化方向：** _<留待下一轮的潜力点，若无则填 N/A>_