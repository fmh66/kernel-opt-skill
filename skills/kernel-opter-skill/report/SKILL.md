---
name: report-skill
description: Generate the final CUDA kernel optimization report (final_report.md) after all iterations complete. Aggregates env, NCU metrics, strategy decisions, and best-version selection across all versions into a single structured Markdown document.
---

# report-skill

## 目录结构

```
report/
├── SKILL.md
└── prompt/
    └── report.md
```

## 职责

在**所有优化迭代完成后**（达到最大迭代次数 N），读取各版本的产物，**严格**按照 `prompt/report.md` 模板生成 `<output_dir>/final_report.md`。

**核心约束：每个字段必须填写具体值，禁止输出占位符（如 `_<填写>_`）。若数据缺失，填 `N/A — <缺失原因>`。**

---

## 数据来源

| 报告字段 | 读取文件 |
|---|---|
| 环境（GPU、CUDA/nvcc、ncu、nsight-python、PyTorch） | `<output_dir>/env_check.md` |
| 执行时间、Memory/Compute/SM Throughput、Warp Stall、Branch Divergence | `<output_dir>/v{n}/ncu_summary.md` |
| 占用率、寄存器/线程、Shared Mem/block | `<output_dir>/v{n}/ncu_summary.md` |
| 瓶颈判定 | `<output_dir>/v{n}/ncu_summary.md`（由 profiling-skill 在 Step 2 写入） |
| 各版本优化策略与决策说明 | 对话上下文（cuda-skill 在 Step 3/4/5/6 的输出） |
| 正确性 | `<output_dir>/v{n}/correctness.md` |
