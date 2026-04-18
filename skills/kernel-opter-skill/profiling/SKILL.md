---
name: profiling-skill
description: Compile, validate, and benchmark a CUDA kernel; collect NCU profiles; interpret NCU metrics to classify bottlenecks.
---

# Profiling & NCU 解读

## Benchmark（benchmark.py）

```bash
python skills/kernel-opter-skill/profiling/script/benchmark.py <kernel.cu> \
    [--ref=<ref.py>] [--M=1024 --N=1024 ...] \
    [--warmup=10] [--repeat=20] [--json-out=result.json]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `solution_file` | — | `.cu` 文件，需暴露 `extern "C" void solve(...)` |
| `--ref` | — | 参考实现 `.py`，启用 correctness 验证 + speedup |
| `--M/--N/...` | — | kernel 签名中的整型维度参数 |
| `--warmup` | 10 | 预热轮数 |
| `--repeat` | 20 | 计时轮数 |
| `--ptr-size` | 0 | 覆盖指针 buffer 元素数 |
| `--arch` | 自动探测 | 如 `sm_90` |
| `--gpu` | 0 | GPU 设备索引 |
| `--atol/--rtol` | 1e-4/1e-3 | correctness 容差 |
| `--seed` | 42 | 随机种子（验证时生效） |
| `--nvcc-bin` | nvcc | nvcc 可执行路径 |
| `--json-out` | — | 结果写入 JSON |
| `--skip-compile` | false | 跳过 nvcc 编译，直接加载已有 `.so`（NCU profiling 专用） |

**JSON 字段**：`solution_file` · `backend` · `dims` · `gpu_name` · `arch` · `correctness{checked,passed,atol,rtol}` · `kernel{average_ms,median_ms,min_ms,max_ms}` · `reference` · `speedup_vs_reference` · `error`

**错误处理**：correctness 失败 → 停止并报最大误差与首个错误位置；编译失败 → 原样返回 nvcc 错误；缺少维度 → `missing_dimension` 退出。

---

## NCU Profiling

> **重要**：必须先在 profiler 外编译（运行一次不带 `--skip-compile` 的 benchmark.py 生成 `.so`），然后 NCU 调用时带 `--skip-compile`，避免 ncu 拦截 nvcc 子进程导致编译失败。不使用 `--target-processes all`。

### Targeted（快速定位）

```bash
# Step 1: 先编译
python skills/kernel-opter-skill/profiling/script/benchmark.py <kernel.cu> \
    --M=1024 --N=1024 --warmup=0 --repeat=1

# Step 2: NCU profiling（--skip-compile 跳过编译）
ncu \
    --profile-from-start on \
    --launch-skip 20 --launch-count 1 \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables \
    --section ComputeWorkloadAnalysis \
    --section Occupancy --section LaunchStats \
    --section SchedulerStats --section WarpStateStats \
    -o <out> -f \
    python skills/kernel-opter-skill/profiling/script/benchmark.py <kernel.cu> \
    --M=1024 --N=1024 --repeat=22 --skip-compile
```

### Full（最终证据）

```bash
ncu \
    --profile-from-start on \
    --launch-skip 20 --launch-count 1 \
    --set full \
    -o <out> -f \
    python skills/kernel-opter-skill/profiling/script/benchmark.py <kernel.cu> \
    --M=1024 --N=1024 --repeat=22 --skip-compile
```

最终交付必须附 full 证据；targeted 只作初步定位。

> **数据量建议**：单一数据量的 NCU 结果可能受边界效应影响（如 warp 数量恰好整除、L2 完全命中等），结论不够可靠。建议对同一 kernel **至少用 3 个数据量级**（每次 ×10，如 `--ptr-size=1024`、`10240`、`102400`）分别采集 full profile，若三者瓶颈分类一致则结论可信；若出现分歧，以**最大规模**为准（小数据量更易受 launch overhead 和 cache 暖机干扰）。

---

## NCU 解读与瓶颈分类

### 一级分类（SpeedOfLight）

| 条件 | 结论 | 下一步 section |
|---|---|---|
| Memory SOL > 60% 且远高于 SM SOL | **Memory-Bound** | MemoryWorkloadAnalysis |
| SM SOL > 60% 且远高于 Memory SOL | **Compute-Bound** | ComputeWorkloadAnalysis |
| 两者均 < 40% | **Latency-Bound** | Occupancy + WarpStateStatistics |
| Achieved Occ << Theoretical | **Occupancy-Bound** | LaunchStatistics |

### 二级信号速查

| NCU 指标 | 问题信号 | 瓶颈类型 |
|---|---|---|
| `Global Load/Store Efficiency` | < 100% | Memory |
| `Sectors/Request` | > 1 | Memory |
| `L1 / L2 Hit Rate` | 过低 | Memory |
| `Shared Memory Efficiency` | < 100% | Memory（bank conflict） |
| `FP32/FP16/Tensor Pipe Utilization` | 不均衡 | Compute |
| `Issue Slot Utilization` | < 50% | Compute |
| `Warp Execution Efficiency` | < 100% | Compute（分支发散） |
| `Register Spill` | > 0 | Compute / Latency |
| `Stall Barrier` | 高 | Latency（同步） |
| `Stall Long Scoreboard` | 高 | Latency（全局内存延迟） |
| `Stall Short Scoreboard` | 高 | Latency（Shared/L1 延迟） |
| `Branch Efficiency` | < 100% | Compute（warp divergence） |

> 完整指标含义 → `reference/NCU.md`

---

> 优化策略（按瓶颈类型）→ `cuda/SKILL.md`
