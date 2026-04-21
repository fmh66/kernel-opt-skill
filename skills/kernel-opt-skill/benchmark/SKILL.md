---
name: benchmark
description: Benchmark a custom CUDA/Triton kernel against a reference implementation (PyTorch/CUTLASS). Measures execution time via CUDA Events and collects hardware metrics via nsight-python.
---

# benchmark-skill

## 目录结构

```
benchmark/
├── SKILL.md
└── script/
    └── benchmark.py
```

---

## 功能说明

对比 solution kernel 与参考实现的性能表现，输出：

- **执行时间**（CUDA Events，100 次迭代 mean ± std）
- **硬件指标**（nsight-python：SM 吞吐、内存吞吐、DRAM 带宽、Achieved Occupancy）
- **正确性验证**（`torch.allclose`，运行在正式计时之前）

> 测量策略：执行时间由 CUDA Events 采集（不受 nsight replay 干扰），nsight 只用于采集硬件利用率指标。

---

## 用法

> **前置条件**
> - CUDA：需先通过 nvcc 编译好 `.so` 文件，脚本只加载不编译
> - Triton：无需编译 `.so`

```bash
# 先编译
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o kernel.so kernel.cu

# benchmark（CUDA 或 Triton）
python script/benchmark.py <solution.{cu,py}> \
    --ref=<ref.py> \
    --output-dir=<dir> \
    --M=<M> --N=<N> \
    [--backend=<auto/cuda/triton>] \
    [--warmup=<n>] \
    [--iters=<n>] \
    [--ptr-size=<n>] \
    [--arch=<sm_XX>] \
    [--gpu=<id>] \
    [--atol=<atol>] [--rtol=<rtol>] \
    [--seed=<seed>] \
    [--skip-nsight]
```

---

## CLI 参数

| 参数 | 必选 | 默认 | 说明 |
|---|:---:|---|---|
| `solution_file` | ✓ | — | `.cu` 或 `.py`（Triton） |
| `--ref` | ✓ | — | 参考实现 `.py`，定义 `reference(**kwargs)` |
| `--output-dir` | ✓ | — | 输出目录 |
| `--M/--N/...` | ✓ | — | kernel 签名中的整型维度参数 |
| `--backend` | | `auto` | `auto/cuda/triton` |
| `--warmup` | | 20 | 正式计时前的预热轮数 |
| `--iters` | | 100 | CUDA Events 计时迭代次数 |
| `--ptr-size` | | 0 | 覆盖 CUDA 指针 buffer 元素数（Triton 可忽略） |
| `--arch` | | 自动探测 | 如 `sm_90` |
| `--gpu` | | 0 | GPU 设备索引 |
| `--atol/--rtol` | | 1e-4/1e-3 | 正确性容差 |
| `--seed` | | 42 | 随机种子 |
| `--skip-nsight` | | false | 跳过 nsight 硬件指标采集，仅输出执行时间 |

---

## 输出文件

| 文件 | 说明 |
|---|---|
| `benchmark.md` | solution 与 reference 对比报告，含正确性、执行时间、硬件指标 |
