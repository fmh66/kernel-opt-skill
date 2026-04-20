# NCU 指标解读参考

## 一、瓶颈分类（SpeedOfLight）

| 指标 | 含义 |
|---|---|
| `SM SOL %` | 计算单元利用率占峰值百分比 |
| `Memory SOL %` | 显存带宽利用率占峰值百分比 |

**判定规则**：

| 条件 | 结论 | 下一步 |
|---|---|---|
| Memory SOL > 60% 且远高于 SM SOL | **Memory-Bound** | 查 MemoryWorkloadAnalysis |
| SM SOL > 60% 且远高于 Memory SOL | **Compute-Bound** | 查 ComputeWorkloadAnalysis |
| 两者均 < 40% | **Latency-Bound** | 查 Occupancy + WarpStateStatistics |
| Achieved Occ << Theoretical 且限制因子明确 | **Occupancy-Bound** | 查 LaunchStatistics |
| 无单一主症状 | **Mixed** | 先处理最明确的一类 |

---

## 二、Memory-Bound 细分（MemoryWorkloadAnalysis）

| NCU 指标 | 问题信号 | 含义 |
|---|---|---|
| `Global Load/Store Efficiency` | < 100% | 未合并访问，存在无效带宽消耗 |
| `Sectors/Request` | > 1（理想值 1） | 每次请求读取了多个 sector，说明未对齐或未合并 |
| `L1 Hit Rate` | 过低 | 数据局部性差，L1 无法复用 |
| `L2 Hit Rate` | 过低 | 工作集超出 L2，频繁访问 DRAM |
| `Shared Memory Efficiency` | < 100% | 存在 bank conflict，访问被串行化 |
| `DRAM Throughput` | 接近峰值但 kernel 仍慢 | 已达带宽极限，需算法减少访存量 |

---

## 三、Compute-Bound 细分（ComputeWorkloadAnalysis）

| NCU 指标 | 问题信号 | 含义 |
|---|---|---|
| `FP32/FP16/Tensor Pipe Utilization` | 不均衡或偏低 | 未使用合适的计算管线（如该用 Tensor Core 没用） |
| `Issue Slot Utilization` | < 50% | 指令发射槽空闲，调度不饱和 |
| `Warp Execution Efficiency` | < 100% | 存在 warp divergence，部分 lane 空转 |
| `Eligible Warps Per Cycle` | 过低 | 可调度 warp 不足，ILP 不足或 occupancy 过低 |
| `Register Spill (Local Memory)` | > 0 | 寄存器溢出，退化为全局内存访问 |

---

## 四、Occupancy（Occupancy + LaunchStatistics）

| NCU 指标 | 含义 |
|---|---|
| `Achieved Occupancy` | 实际驻留 warp 数 / SM 理论最大值 |
| `Theoretical Occupancy` | 按资源限制计算的理论上限 |
| `Registers Per Thread` | 每线程寄存器用量（主要限制因子之一） |
| `Shared Memory Per Block` | 每 block 共享内存用量（主要限制因子之一） |

**解读**：Achieved << Theoretical 说明资源受限；具体限制因子见 `LaunchStatistics` section。注意 occupancy 不是越高越好，以实测 latency 为最终判据。

---

## 五、Warp 调度与 Stall（SchedulerStatistics + WarpStateStatistics）

| Stall 类型 | 含义 | 常见原因 |
|---|---|---|
| `Stall Barrier` | 等待 `__syncthreads()` 完成 | 同步点过多 |
| `Stall Long Scoreboard` | 等待全局内存数据就绪 | 全局内存延迟未隐藏 |
| `Stall Short Scoreboard` | 等待 Shared Memory / L1 数据就绪 | Bank conflict 或访问未合并 |
| `Stall MIO Throttle` | 内存指令队列已满 | 访存指令密度过高 |
| `Stall No Instructions` | 指令 cache miss | kernel 代码体积过大 |
| `Eligible Warps Per Cycle` 低 | 每周期可调度 warp 不足 | ILP 不足 / occupancy 过低 |

---

## 六、分支发散（SourceCounters + InstructionStatistics）

| NCU 指标 | 含义 |
|---|---|
| `Branch Efficiency` | 未发散分支占全部分支的比例；< 100% 说明有 warp divergence |
| `Divergent Branches` | 发散分支数量 |

**解读**：Branch Efficiency 偏低时，同一 warp 内线程走了不同分支路径，GPU 需串行执行两条路径，有效吞吐减半。

---

优化策略入口：`cuda/SKILL.md` — 按瓶颈类型给出实施优先级。
