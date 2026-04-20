# CUDA 优化最终报告 — `naive_softmax` (`2026-04-20`)

## 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / 12.6.85 |
| ncu | 2024.3.2.0 (build 34861637) |
| nsight-python | 0.9.6 |
| PyTorch | 2.11.0+cu126 |
| 内核文件 | `/home/kernel-opter-skill/test/test.cu` |

---

## 版本迭代对比

| 指标 | v0（基线） | v1 | v2 | v3 | best |
|---|---|---|---|---|---|
| 执行时间 (ns) | 869,184 | 127,680 | 123,648 | 135,424 | **123,648 (v2)** |
| 加速比 (×) | 1.00 | 6.81 | 7.03 | 6.42 | **7.03** |
| Memory Throughput (%) | 33.07 | 90.24 | 91.82 | 87.62 | 91.82 |
| Compute Throughput (%) | 3.00 | 30.32 | 26.97 | 22.42 | — |
| DRAM Total BW (GB/s) | 241 | 658 | 669 | 639 | — |
| 瓶颈判定 | Latency-Bound | Memory-Bound | Memory-Bound | Memory-Bound | — |
| 理论占用率 (%) | 100 | 100 | 100 | 100 | — |
| 达成占用率 (%) | 16.60 | 94.72 | 97.53 | 88.56 | — |
| Waves / SM | 0.079 | 20.32 | 20.32 | 20.32 | — |
| 寄存器 / 线程 | 38 | 21 | 19 | 24 | — |
| Shared Mem / block (bytes) | 0 | 128 | 4096+128 | 256 | — |
| Global Load Efficiency (%) | 12.5 | 100 | 100 | 100 | — |
| Global Store Efficiency (%) | 12.5 | 100 | 100 | 100 | — |
| L1 Hit Rate (%) | 91.84 | 54.63 | 0.00 | 21.51 | — |
| L2 Hit Rate (%) | 84.24 | 68.59 | 50.18 | 53.90 | — |
| Warp Stall — Long SB (%) | 46.24 | 29.71 | 32.32 | 28.58 | — |
| Warp Stall — Barrier (%) | 0.00 | 5.61 | 11.54 | 7.42 | — |
| Branch Divergence (分支目标数) | 515,520 | 1,348,608 | 1,273,856 | 348,160 | — |
| 分歧分支目标 | 0 | 0 | 0 | 0 | — |

---

## 各版本优化策略

| 策略 | v1 | v2 | v3 |
|---|---|---|---|
| 合并全局内存访问（128B 对齐） | ✓ | ✓ | ✓ |
| Shared Memory tiling（整行缓存） | ✗ | ✓ | ✗ |
| `__ldg` / L2 持久化 | ✗ | ✗ | ✗ |
| `cp.async` 异步预取 | ✗ | ✗ | ✗ |
| 向量化加载（`float4`） | ✗ | ✓ | ✓ |
| Tensor Core（`wmma` / `mma`） | ✗ | ✗ | ✗ |
| ILP（循环展开 `#pragma unroll`） | ✗ | ✗ | ✓ |
| 混合精度（FP16 / BF16 / FP8） | ✗ | ✗ | ✗ |
| 增大 block size / `__launch_bounds__` | ✗ | ✗ | ✓ |
| Warp Shuffle Reduction | ✓ | ✓ | ✓ |
| 1 block / 行（Grid = N） | ✓ | ✓ | ✓ |
| Online Softmax（2-pass Milakov 2018） | ✗ | ✗ | ✓ |
| `__expf` 快速超越函数 | ✗ | ✗ | ✓ |

**各轮决策说明：**

- **v1：** v0 的根本问题是"1 线程/行"模式 —— 仅 40 blocks（Waves/SM=0.079），95% 以上的 SM 空闲；同时 warp 内相邻线程跨行步长 D=1024，Global Load/Store Efficiency 仅 12.5%。将 block 分配改为"1 block/行"（Grid=N=10240, blockDim=256）后，Grid 扩大 256×，访存变为 warp 内连续行内元素（100% 合并），占用率从 16.6% 跃升至 94.7%，Stall Long Scoreboard 从 46.2% 降至 29.7%。增加 Warp Shuffle 消除 shared memory 三步同步。结果：**6.81× 加速**。

- **v2：** v1 在 3 遍扫描（max/exp+sum/normalize）时，pass 2 和 pass 3 对同一数据的重复访问由 L1 缓存承担（L1 命中率 54.6%），但 Stall Long Scoreboard 仍达 29.7%。v2 将整行数据（D=1024 floats = 4KB）显式缓存到 shared memory，使 pass 2 和 pass 3 的输入完全从 shared memory 读取，并加入 float4 向量化加载/写出。shared memory 消除了 L1 的不确定性，L1 Bank Conflicts 从 4.3×10⁷ 降至 3.6×10⁴（1000×），DRAM 写带宽有所提升。结果：额外 **+3.2% 提速**（7.03×），duration 123,648 ns，成为最优版本。

- **v3：** 尝试 Online Softmax（Milakov & Gimelshein 2018）将 3 遍改为 2 遍，pass 1 同时计算 running max 和 sum，省去中间 exp 值写回。然而 online update 在每个元素上引入 2 次额外 `__expf` 调用（`d * exp(m - m_new) + exp(x - m_new)`），使 pass 1 变为 compute-heavy；寄存器/线程从 19 升至 24，占用率从 97.5% 降至 88.6%，FMA 利用率上升至 12.3%。计算开销超过了省去 1 次输出写回的收益，kernel 反而**退步至 135,424 ns（−9.5%）**。

---

## 选优结论

**最优版本：** `v2` — 执行时间从 869,184 ns 降至 123,648 ns，加速比 **7.03×**。  
核心收益：`1 block/行` 解决根本性低占用率问题（16.6%→97.5%）+ 合并访存（12.5%→100%）；shared memory 整行缓存消除中间 exp 值的 L1/L2 压力，DRAM 带宽达到 669 GB/s（87% of peak）。  
停止原因：达到最大迭代次数 3 次。

**未尽优化方向：**
- **cp.async + 双缓冲**：用 `cp.async` 异步加载 shared memory，与 exp 计算重叠，进一步掩盖 Stall Long Scoreboard（当前 32.3%）。
- **在线 Softmax + 分离 kernel**：若 D 更大（D > L1 容量），shared memory 整行缓存方案会因占用率下降失效；此时将 max/sum 与 normalize 分为两个独立 kernel 并保留 v3 的 online 思想更优。
- **L2 Persistence**（sm_86 支持）：对 batch 重复调用 softmax 可用 `cudaStreamAttrValue` 锁定权重到 L2，减少 DRAM 延迟。
- **Warp-level unroll + ILP**：将 D/blockDim=4 次循环展开为 4 路 ILP（各自发射 float4 load → expf → multiply），进一步掩盖 28.6% 的 Long Scoreboard stall。
