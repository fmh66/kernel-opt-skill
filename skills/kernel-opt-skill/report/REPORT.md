---
name: report
description: Generate the final CUDA/Triton kernel optimization report (final_report.md) after all iterations complete. Aggregates env, NCU metrics, strategy decisions, and best-version selection across all versions into a single structured Markdown document.
---

# report-skill

## Directory Structure

```
report/
├── REPORT.md
└── prompt/
    └── report.md
```

## Responsibility

After **all optimization iterations complete** (reaching max iterations N), read the artifacts from all versions and **strictly** follow the `prompt/report.md` template to generate `<output_dir>/final_report.md`.

**Core constraint: every field must be filled with a concrete value; placeholders (e.g., `_<fill in>_`) are prohibited. If data is missing, fill with `N/A — <reason for missing>`.**

---

## Data Sources

| Report Field | Read From |
|---|---|
| Environment (GPU, CUDA/nvcc, ncu, nsight-python, Triton, PyTorch) | `<output_dir>/env_check.md` |
| Execution time, Memory/Compute/SM Throughput, Warp Stall, Branch Divergence | `<output_dir>/v{n}/ncu_summary.md` |
| Occupancy, registers/thread, shared mem/block | `<output_dir>/v{n}/ncu_summary.md` |
| Bottleneck classification | `<output_dir>/v{n}/ncu_summary.md` (written by profiling-skill in Step 2) |
| Optimization strategies and decision rationale per version | Conversation context (cuda-skill output from Steps 3/4/5/6) |
| Correctness | `<output_dir>/v{n}/correctness.md` |
