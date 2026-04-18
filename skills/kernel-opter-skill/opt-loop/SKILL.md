---
name: opt-loop
description: Multi-iteration CUDA kernel optimization loop with strategy memory and best-version selection.
---

# Operator Optimization Loop

## Inputs

| 参数 | 必须 | 说明 |
|---|---|---|
| `kernel.cu` | ✓ | 需暴露 `extern "C" void solve(...)` |
| `--max-iterations=N` | ✓ | 未提供时停止并要求用户给出 |
| `--ref=<ref.py>` | 强烈建议 | 缺少则无法宣称 correctness 已验证 |
| `--M/--N/...` | 视签名 | kernel 整型维度参数 |
| `--warmup` / `--repeat` | — | 默认 10 / 20 |
| `--run-dir` | — | 指定输出目录 |
| `--resume-from=best\|source\|explicit` | — | 默认 `best` |
| `--arch` / `--gpu` / `--seed` / `--ptr-size` | — | 同 benchmark.py |
| `--nvcc-bin` / `--ncu-bin` | — | 工具路径 |

---

## Workflow

```mermaid
flowchart TD
    A[开始] --> B[Preflight\nGPU / nvcc / ncu / 输入文件]
    B --> C{环境就绪?}
    C -->|否| Z[停止 输出错误报告]
    C -->|是| D["迭代 v0 → vN"]

    D --> E[benchmark + targeted/full NCU]
    E --> F[读取证据\nbenchmark_result.json\nNCU summary / details]
    F --> G[制定优化方向\n写 optimization_proposal.md]
    G --> H[生成下一版 kernel]
    H --> I{达到 max_iterations?}
    I -->|否| E
    I -->|是| J[选出 best version\n输出 final_summary.md]
```

---

## Strategy Memory

两级记忆：当前 run（`run_manifest.json`）+ 全局（`strategy-memory/global_strategy_memory.json`）。

`optimization_proposal.md` 必须包含 `## Strategy tags` 节。

```mermaid
flowchart TD
    A[判定本轮 outcome] --> B{benchmark 失败?}
    B -->|是| R[rejected]
    B -->|否| C{correctness 失败?}
    C -->|是| R
    C -->|否| D{full NCU 缺失?}
    D -->|是| R
    D -->|否| E{median < 上一轮?}
    E -->|是| P[positive]
    E -->|否| N[negative]
```

- `blocked`：跳过 rejected 策略指纹，避免重复踩坑
- `preferred`：优先融合 positive 策略指纹

---

## Outputs

**Run 级**：`run_manifest.json` · `final_summary.md` · `preflight_check.md/json` · `iter_v0/` …

**Iter 级**：`<kernel>.cu` · `benchmark_result.json` · `benchmark.stdout/stderr.txt` · `iteration_summary.md` · `optimization_proposal.md` · `targeted.ncu-rep` + `targeted_summary/details.txt` · `full.ncu-rep` + `full_summary/details.txt`

**Final response 必须包含**：最佳版本路径 · baseline vs best 对比 · best full NCU 路径 · 主瓶颈与关键优化思路 · 淘汰版本及原因 · 策略记忆结论（positive/negative/rejected） · blocked/preferred 执行情况

---

## Failure Handling

- correctness 失败 → 标记 rejected，不参与 best 排名
- profiling 不可用或 full 缺失 → 明确原因，停止
- benchmark / 环境失败 → 输出失败证据，不得静默跳过

**提前停止条件**：语义错误需先修正确性 · `ncu` 无法运行 · 连续多轮无可解释性能改善 · 达到用户目标或收益明显递减

---

## Examples

```bash
# 基础用法
python skills/kernel-opter-skill/opt-loop/scripts/opt-loop.py <kernel.cu> \
    --max-iterations=N --ref=<ref.py> --M=1024 --N=1024 --warmup=10 --repeat=20

# 继续已有 run-dir
python skills/kernel-opter-skill/opt-loop/scripts/opt-loop.py <next.cu> \
    --run-dir=<existing_dir> --resume-from=best --max-iterations=N \
    --ref=<ref.py> --M=1024 --N=1024

# 仅做 preflight 检查
python skills/kernel-opter-skill/opt-loop/scripts/opt-loop.py <kernel.cu> \
    --max-iterations=N --preflight-only
```
