---
name: profiling-skill
description: Validate CUDA kernel correctness and collect NCU profiles; interpret NCU metrics to classify bottlenecks. All scripts require a pre-compiled .so file.
---

# Profiling & NCU 解读

## 目录结构

```
profiling/
├── SKILL.md
├── reference/
│   └── NCU.md
└── script/
    ├── correctness_check.py
    └── ncu_profile.py
```

## Correctness Check

> **前置条件**：需先通过 nvcc 编译好 `.so` 文件，脚本只加载不编译。

```bash
# 先编译
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o kernel.so kernel.cu

# 再检查正确性
python script/correctness_check.py <kernel.cu> \
    --ref=<ref.py> \
    --M=<M> --N=<N> \
    --output-dir=<dir> \
    [--ptr-size=<n>] \
    [--arch=<sm_XX>] \
    [--gpu=<id>] \
    [--atol=<atol>] [--rtol=<rtol>] \
    [--seed=<seed>]
    
```

| 参数 | 必选 | 默认 | 说明 |
|---|:---:|---|---|
| `solution_file` | ✓ | — | `.cu` 文件，需暴露 `extern "C" void solve(...)` |
| `--ref` | ✓ | — | 参考实现 `.py`，定义 `reference(**kwargs)` |
| `--M/--N/...` | ✓ | — | kernel 签名中的整型维度参数 |
| `--output-dir` | ✓ | — | 写入 `correctness.md` 的目录 |
| `--ptr-size` | | 0 | 覆盖指针 buffer 元素数 |
| `--arch` | | 自动探测 | 如 `sm_90` |
| `--gpu` | | 0 | GPU 设备索引 |
| `--atol/--rtol` | | 1e-4/1e-3 | correctness 容差 |
| `--seed` | | 42 | 随机种子 |

---

## NCU Profiling（via nsight-python）

> **前置条件**：需先通过 nvcc 编译好 `.so` 文件，脚本只加载不编译。nsight-python 内部管理 ncu 子进程，无需手动构造 ncu 命令。

```bash
python script/ncu_profile.py <kernel.cu> \
    --output-dir=<dir> \
    --M=<M> --N=<N> \
    [--warmup=<n>] \
    [--ptr-size=<n>] \
    [--arch=<sm_XX>] \
    [--gpu=<id>] \
    [--seed=<seed>]
```

### 输出文件

| 文件 | 说明 |
|---|---|
| `ncu_summary.md` | 按类别组织的关键指标摘要（供 LLM 阅读） |
| `ncu_details.md` | 全部指标详细表格（含 avg/std/min/max 及稳定性标记） |

### CLI 参数

| 参数 | 必选 | 默认 | 说明 |
|---|:---:|---|---|
| `solution_file` | ✓ | — | `.cu` 文件 |
| `--output-dir` | ✓ | — | 输出目录 |
| `--M/--N/...` | ✓ | — | kernel 签名中的整型维度参数 |
| `--warmup` | | 20 | profiling 前的预热轮数 |
| `--ptr-size` | | 0 | 覆盖指针 buffer 元素数 |
| `--arch` | | 自动探测 | 如 `sm_90` |
| `--gpu` | | 0 | GPU 设备索引 |
| `--seed` | | 42 | 随机种子 |

> **整型维度建议**：优选选用`10240`、`102400`...数量级大的数据
> **注意**：所有迭代版本的整型维度保持一致

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

> 优化策略详见 cuda-skill（`kernel-opter-skill/cuda/SKILL.md`）
