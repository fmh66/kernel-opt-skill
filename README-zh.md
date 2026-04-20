# kernel-opt-skill

面向 CUDA 的 kernel 优化 Skill，通过系统化的性能分析、瓶颈定位和迭代优化，帮助开发者快速提升 CUDA kernel 性能。

[English](ReadMe.md)

## 环境要求

| 依赖项 | 版本要求 |
| --- | --- |
| NVIDIA GPU | Compute Capability 7.0+（Volta 及以上） |
| CUDA Toolkit | 11.6+（推荐 12.6+） |
| Nsight Compute | 2024.3.2+ |
| Python | 3.10+ |
| PyTorch | 2.0+ |
| nsight-python | 0.9.6+ |

## 项目结构

```text
kernel-opt-skill/
├── skills/kernel-opt-skill/
│   ├── SKILL.md                  # 主入口，定义优化流程
│   ├── env/                      # 环境检查与 GPU 配置
│   ├── profiling/                # NCU 性能分析与正确性验证
│   ├── benchmark/                # solution 与 reference 框架横向对比
│   ├── cuda/                     # 内存/计算/延迟优化策略参考
│   └── report/                   # 报告生成模板
└── demo/                         # 优化实战案例（softmax、gemm……）
```

## 快速开始

调用 Skill，指定待优化的 kernel 文件、迭代次数和输出目录：

```text
/kernel-opt-skill 请帮我优化这个 kernel <kernel.cu>，迭代三次，输出到 <output_dir> 目录
```

触发后，将按以下步骤自动执行优化循环：

```mermaid
flowchart TD
    A[Step 0: 正确性检查] --> B[Step 1: NCU 性能采集]
    B --> C["Step 2: 瓶颈全局定位（Speed of Light 分析）"]
    C --> D["Step 3: 针对性优化（Memory / Compute / Latency）"]
    D --> E["Step 4–6: 深入分析（占用率 / Warp 调度 / 分支发散）"]
    E --> F[Step 7: 生成下一版本]
    F -->|循环 N 次| B
    F -->|达到迭代上限| G["生成 final_report.md（汇总各版本，选出最优实现） & benchmark"]
```

### 输出目录结构

```text
<output_dir>/
├── ref.py                  # 参考实现
├── env_check.md            # 环境信息
├── v0/
│   ├── v0.cu               # 源码
│   ├── correctness.md      # 正确性验证结果
│   ├── ncu_summary.md      # NCU 指标摘要（LLM 友好格式）
│   └── ncu_details.md      # NCU 完整指标表格
├── v1/ v2/ v3/ ...         # 各迭代版本（结构同上）
├── final_report.md         # 最终优化对比报告
└── benchmark.md            # 最优版本与 reference 的性能横向对比
```

## Benchmark 对比

优化循环结束后，benchmark-skill 自动将 **best version** 与 **reference implementation（PyTorch / CUTLASS）** 进行横向性能对比：

| 维度 | 方式 | 说明 |
| --- | --- | --- |
| Execution Time | CUDA Events（100 次迭代） | 真实 wall-clock latency，不受 nsight replay 干扰 |
| SM Throughput / Memory Throughput | nsight-python | 硬件利用率 vs peak |
| DRAM Bandwidth | nsight-python | 实际 Memory Bandwidth 绝对值 |
| Achieved Occupancy | nsight-python | Active warp 占比，反映并行度 |

结果写入 `<output_dir>/benchmark.md`。

## 实战案例

### Softmax 优化

参见 [demo/softmax/](demo/softmax/) 目录，完整记录了从基线到最优版本的 4 轮迭代过程。

| 版本 | Execution Time | Speedup | Bottleneck | 关键优化 |
| --- | --- | --- | --- | --- |
| v0（基线） | 891,936 ns | 1.00× | Latency-Bound | 朴素实现（1 thread/行） |
| v1 | **124,896 ns** | **7.14×** | Memory-Bound | 1 block/行 + Warp Shuffle |
| v2 | 131,424 ns | 6.79× | Memory-Bound | Online Softmax + float4（L1 reuse 失效，性能下降） |
| v3 | 127,808 ns | 6.98× | Memory-Bound | 3-pass + float4 + `__expf__`（ILP 下降，性能下降） |

**v1 关键改进（最优版本）：**

- Block 分配策略从 1 thread/行改为 1 block/行，Block 数 40 → 10,240，Achieved Occupancy 16.6% → 94.8%
- Global Load Efficiency 从 12.5% 提升至 100%（coalesced access），DRAM Bandwidth 235 → 673 GB/s
- 使用 Warp Shuffle reduce，仅需 32 字节 Shared Memory 即可完成 max/sum broadcast，无需全局 atomic 操作
- `__ldg` / `__restrict__` read-only cache hint，减少 L2 访问压力

**Benchmark：v1 vs PyTorch reference（N=10240, D=1024）**

| Metric | v1（最优） | PyTorch reference |
| --- | --- | --- |
| Execution Time | **0.1465 ms** | 0.2657 ms |
| SM Throughput | 33.3% | 32.8% |
| Memory Throughput | 91.6% | 91.8% |
| DRAM Bandwidth | 668 GB/s | 669 GB/s |
| Achieved Occupancy | 93.0% | 93.2% |

v1 Execution Time 比 PyTorch 快 **1.81×**，硬件利用率几乎持平——说明两者已同等充分利用 Memory Bandwidth，性能差距来自 PyTorch 的 dispatch overhead 而非 kernel 本身的效率差异。

### GEMM 优化

参见 [demo/gemm/](demo/gemm/) 目录，完整记录了从基线到最优版本的 4 轮迭代过程。

| 版本 | Execution Time | Speedup | Bottleneck | 关键优化 |
| --- | --- | --- | --- | --- |
| v0（基线） | 62.00 ms | 1.00× | Compute-Bound | 朴素实现（non-coalesced access） |
| v1 | 44.80 ms | 1.38× | Compute-Bound | Shared Memory tiling（16×16） |
| v2 | 8.75 ms | 7.09× | Balanced | Register blocking（每线程 4×4，64×64 tile） |
| v3 | **6.28 ms** | **9.87×** | Memory-Bound | WMMA Tensor Core（FP16→FP32） |

**v3 关键改进（最优版本）：**

- 激活 WMMA Tensor Core pipeline（utilization 13.6%），FP16→FP32 fragment 理论峰值 310 TFLOPS
- v2 register blocking 将 FMA pipeline utilization 从 10.6% 提升至 47.8%（每 tile 迭代 256 次 FMA vs 16 次），为 v3 奠定基础
- v1–v3 Global Load Efficiency 保持 100%（coalesced access + tiling 策略）
- 代价：FP16 input 引入精度损失（max error 0.101）；每线程 118 个 register 导致 Achieved Occupancy 下降至 32.7%

**Benchmark：v3 vs cuBLAS reference（M=K=N=4096）**

| Metric | v3（最优） | cuBLAS reference |
| --- | --- | --- |
| Execution Time | 6.75 ms | **6.08 ms** |
| SM Throughput | 31.8% | 31.8% |
| Memory Throughput | 45.5% | 46.3% |
| DRAM Bandwidth | 331 GB/s | 338 GB/s |
| Achieved Occupancy | 32.7% | 32.7% |

v3 与 cuBLAS Execution Time 相差约 **11%**，硬件利用率几乎完全一致——差距不在 kernel 效率，而在 cuBLAS 更精细的 register / warp 调度策略上。
