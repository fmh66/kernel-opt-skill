---
name: profiling
description: Validate CUDA/Triton kernel correctness and collect NCU profiles; interpret NCU metrics to classify bottlenecks.
---

# profiling-skill

## Directory Structure

```
profiling/
├── PROFILING.md
├── reference/
│   └── NCU.md
└── script/
    ├── correctness_check.py
    └── ncu_profile.py
```

## Correctness Check

> **Prerequisites**
> - CUDA: compile the shared library via nvcc first; the script only loads, does not compile
> - Triton: no `.so` compilation required

```bash
# CUDA: compile first
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o kernel.so kernel.cu

# Correctness check (CUDA or Triton)
python script/correctness_check.py <solution.{cu,py}> \
    --ref=<ref.py> \
    --M=<M> --N=<N> \
    --output-dir=<dir> \
    [--backend=<auto/cuda/triton>] \
    [--ptr-size=<n>] \
    [--arch=<sm_XX>] \
    [--gpu=<id>] \
    [--atol=<atol>] [--rtol=<rtol>] \
    [--seed=<seed>]
    
```

| Parameter | Required | Default | Description |
|---|:---:|---|---|
| `solution_file` | ✓ | — | `.cu` or `.py` (Triton) |
| `--ref` | ✓ | — | Reference implementation `.py`, defines `reference(**kwargs)` |
| `--M/--N/...` | ✓ | — | Integer dimension parameters from kernel signature |
| `--output-dir` | ✓ | — | Directory to write `correctness.md` |
| `--backend` | | `auto` | `auto/cuda/triton` |
| `--ptr-size` | | 0 | Override CUDA pointer buffer element count (ignored for Triton) |
| `--arch` | | auto-detected | e.g. `sm_90` |
| `--gpu` | | 0 | GPU device index |
| `--atol/--rtol` | | 1e-4/1e-3 | Correctness tolerance |
| `--seed` | | 42 | Random seed |

---

## NCU Profiling (via nsight-python)

> **Prerequisites**
> - CUDA: compile the shared library via nvcc first; the script only loads, does not compile
> - Triton: no `.so` compilation required; execute directly as a Python module
> - `nsight-python` manages the ncu subprocess internally; no need to manually construct ncu commands

```bash
python script/ncu_profile.py <solution.{cu,py}> \
    --output-dir=<dir> \
    --M=<M> --N=<N> \
    [--backend=<auto/cuda/triton>] \
    [--warmup=<n>] \
    [--ptr-size=<n>] \
    [--arch=<sm_XX>] \
    [--gpu=<id>] \
    [--seed=<seed>]
```

### Output Files

| File | Description |
|---|---|
| `ncu_summary.md` | Key metrics organized by category (for LLM reading) |
| `ncu_details.md` | Full metric table with avg/std/min/max and stability flags |

### CLI Parameters

| Parameter | Required | Default | Description |
|---|:---:|---|---|
| `solution_file` | ✓ | — | `.cu` or `.py` (Triton) |
| `--output-dir` | ✓ | — | Output directory |
| `--M/--N/...` | ✓ | — | Integer dimension parameters from kernel signature |
| `--backend` | | `auto` | `auto/cuda/triton` |
| `--warmup` | | 20 | Warmup iterations before profiling |
| `--ptr-size` | | 0 | Override CUDA pointer buffer element count (ignored for Triton) |
| `--arch` | | auto-detected | e.g. `sm_90` |
| `--gpu` | | 0 | GPU device index |
| `--seed` | | 42 | Random seed |

> **Integer dimension recommendation**: prefer large values like `10240`, `102400`, etc.
> **Note**: use the same integer dimensions across all iteration versions.

---

## NCU Interpretation & Bottleneck Classification

### Primary Classification (SpeedOfLight)

| Condition | Conclusion | Next Section |
|---|---|---|
| Memory SOL > 60% and much higher than SM SOL | **Memory-Bound** | MemoryWorkloadAnalysis |
| SM SOL > 60% and much higher than Memory SOL | **Compute-Bound** | ComputeWorkloadAnalysis |
| Both < 40% | **Latency-Bound** | Occupancy + WarpStateStatistics |
| Achieved Occ << Theoretical | **Occupancy-Bound** | LaunchStatistics |

### Secondary Signal Quick Reference

| NCU Metric | Problem Signal | Bottleneck Type |
|---|---|---|
| `Global Load/Store Efficiency` | < 100% | Memory |
| `Sectors/Request` | > 1 | Memory |
| `L1 / L2 Hit Rate` | too low | Memory |
| `Shared Memory Efficiency` | < 100% | Memory (bank conflict) |
| `FP32/FP16/Tensor Pipe Utilization` | imbalanced | Compute |
| `Issue Slot Utilization` | < 50% | Compute |
| `Warp Execution Efficiency` | < 100% | Compute (branch divergence) |
| `Register Spill` | > 0 | Compute / Latency |
| `Stall Barrier` | high | Latency (synchronization) |
| `Stall Long Scoreboard` | high | Latency (global memory latency) |
| `Stall Short Scoreboard` | high | Latency (Shared/L1 latency) |
| `Branch Efficiency` | < 100% | Compute (warp divergence) |

> Full metric reference → `reference/NCU.md`

---

> CUDA optimization strategies → cuda-skill (`cuda/CUDA.md`)
> Triton optimization strategies → triton-skill (`triton/TRITON.md`)
