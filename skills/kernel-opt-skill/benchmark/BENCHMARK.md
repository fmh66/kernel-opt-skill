---
name: benchmark
description: Benchmark a custom CUDA/Triton kernel against a reference implementation (PyTorch/CUTLASS). Measures execution time via CUDA Events and collects hardware metrics via nsight-python.
---

# benchmark-skill

## Directory Structure

```
benchmark/
├── BENCHMARK.md
└── script/
    └── benchmark.py
```

---

## Overview

Compares solution kernel performance against a reference implementation, outputting:

- **Execution time** (CUDA Events, 100-iteration mean ± std)
- **Hardware metrics** (nsight-python: SM throughput, memory throughput, DRAM bandwidth, Achieved Occupancy)
- **Correctness validation** (`torch.allclose`, run before timing)

> Measurement strategy: execution time is collected via CUDA Events (unaffected by nsight replay); nsight is only used for hardware utilization metrics.

---

## Usage

> **Prerequisites**
> - CUDA: compile `.so` first via nvcc; the script only loads, does not compile
> - Triton: no `.so` compilation required

```bash
# Compile first (CUDA only)
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o kernel.so kernel.cu

# Benchmark (CUDA or Triton)
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

## CLI Parameters

| Parameter | Required | Default | Description |
|---|:---:|---|---|
| `solution_file` | ✓ | — | `.cu` or `.py` (Triton) |
| `--ref` | ✓ | — | Reference implementation `.py`, defines `reference(**kwargs)` |
| `--output-dir` | ✓ | — | Output directory |
| `--M/--N/...` | ✓ | — | Integer dimension parameters from kernel signature |
| `--backend` | | `auto` | `auto/cuda/triton` |
| `--warmup` | | 20 | Warmup iterations before timing |
| `--iters` | | 100 | CUDA Events timing iterations |
| `--ptr-size` | | 0 | Override CUDA pointer buffer element count (ignored for Triton) |
| `--arch` | | auto-detected | e.g. `sm_90` |
| `--gpu` | | 0 | GPU device index |
| `--atol/--rtol` | | 1e-4/1e-3 | Correctness tolerance |
| `--seed` | | 42 | Random seed |
| `--skip-nsight` | | false | Skip nsight hardware metric collection; output execution time only |

---

## Output Files

| File | Description |
|---|---|
| `benchmark.md` | Solution vs. reference comparison report with correctness, execution time, and hardware metrics |
