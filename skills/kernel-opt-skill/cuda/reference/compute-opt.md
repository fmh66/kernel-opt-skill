# CUDA Kernel 计算优化

---

## Tensor Core / 专用硬件单元利用

### Tensor Core 加速

从 Volta 架构开始，Tensor Core 可以在一个时钟周期内完成一个小矩阵乘加运算（如 16×16×16 的 FP16 MMA）。通过 WMMA API、MMA PTX 指令或 cuBLAS/CUTLASS 等库来调用。计算吞吐相比普通 CUDA Core 高一个数量级。不用 Tensor Core 做矩阵类运算基本等于浪费了一半以上的芯片算力。

- **补充（架构限定）**："一个时钟周期"更适合作为概念化表述，实际吞吐取决于架构、数据类型、tile 形状、发射节奏和寄存器/流水线占用。
- **补充（数据类型）**：Ampere 及更新架构常见 TF32/BF16 路径，Hopper 及更新架构常见 FP8 路径；选型要结合误差预算、累加精度和转换开销实测。
- **补充（Hopper+）**：可关注 WGMMA（warp-group MMA）路径，通常需要和更严格的 pipeline/sync 配合使用。

### SFU（Special Function Unit）

GPU 上有专门的特殊函数单元处理 sin、cos、rsqrt、log2、exp2 等超越函数。调用对应的 intrinsic（如 `__sinf`）走 SFU 流水线，和普通 ALU 是并行的，但 SFU 吞吐量比 ALU 低，大量超越函数调用会成为瓶颈。

### 整数运算与位操作优化

GPU 的整数乘法和除法代价高。整数除法/取模如果除数是常量，编译器会自动转为乘法+移位；如果是变量，应手动用位运算替代（比如对 2 的幂次用移位和掩码）。`__popc()`、`__clz()`、`__ffs()` 等位操作 intrinsic 直接映射到硬件指令，单周期完成。

---

## Warp 级计算原语

### Warp Shuffle 用于计算

`__shfl_sync`、`__shfl_xor_sync`、`__shfl_down_sync` 等不仅是数据交换工具，更是计算原语。Warp 内的归约（reduction）、扫描（scan/prefix sum）、广播都可以用 shuffle 在寄存器层级完成，比走 shared memory 快得多，且完全没有 bank conflict。

### Warp Vote 函数

`__ballot_sync()`、`__all_sync()`、`__any_sync()` 可以在一条指令内收集整个 warp 的条件判断结果。常用于快速判断"整个 warp 是否都满足/都不满足某条件"，从而跳过不必要的计算分支，相当于 warp 级别的 early exit。

### Warp Match 函数（Volta+）

`__match_any_sync()` 和 `__match_all_sync()` 可以识别 warp 内哪些线程持有相同的值，用于分组处理和去重，避免用 shared memory 做 scatter-gather。

---

## 循环优化

### 循环展开

用 `#pragma unroll` 或 `#pragma unroll N` 告诉编译器展开循环。展开后减少了循环控制指令（比较、跳转），更重要的是暴露了更多独立指令给调度器，提升 ILP。对小循环（迭代次数已知且不大）效果显著。

### 循环分裂

把一个包含多种不相关计算的循环拆成多个独立循环。每个循环的寄存器压力更小，编译器更容易优化，也更容易做向量化。

### 循环合并

相反地，把多个遍历同一数据的循环合并为一个，让数据在寄存器中被多次使用，减少全局内存的重复访问。

### 循环交换

调整嵌套循环的顺序，让最内层循环的访存模式变为连续访问，改善合并访问特性。

### 软件流水（计算层面）

循环体内把当前迭代的计算和下一迭代的数据预取重叠起来。一个迭代分为"加载→计算→存储"三个阶段，不同迭代的不同阶段交错执行，最大化功能单元利用率。

---

## 算法层面的计算优化

### 归约优化

经典的并行归约从最朴素的交错寻址，到 sequential addressing、warp unrolling、warp shuffle 归约，每一步优化都有明显收益。最终形态是：warp 内用 shuffle 归约，warp 间用 shared memory 归约，block 间用 atomic 或二次 kernel 归约。

### Scan / Prefix Sum 优化

Blelloch 算法（work-efficient scan）和 Hillis-Steele 算法（step-efficient scan）各有适用场景。大规模 scan 通常分为 block 内 scan、block 间辅助数组 scan、再回填三个阶段。

### 用乘法替代除法

浮点除法的吞吐远低于乘法。`a / b` 可以变为 `a * __frcp_rn(b)` 或 `a * (1.0f / b)`，让编译器用倒数近似+乘法替代。`__fdividef(a, b)` 也是类似的快速路径。

### 用 FMA 替代分离的乘加

`a * b + c` 应该被编译为一条 FMA（Fused Multiply-Add）指令，但有时编译器出于精度考虑不会自动合并。用 `__fmaf_rn()` 或 `fmaf()` 显式调用，确保只用一条指令完成，同时精度还更高（中间结果不截断）。

### 查表法（LUT）

对于复杂函数，如果输入范围有限，可以预计算结果放在 shared memory 或 constant memory 中做查表，用一次读取替代大量计算。特别适合离散映射或分段函数。

### 强度削减

用计算代价更低的等价运算替换。比如：用移位替代 2 的幂乘除法、用加法累积替代循环内的乘法（`i*stride` → 累加 `stride`）、用 `rsqrtf()` 替代 `1.0f/sqrtf()`（rsqrt 是硬件原生支持的单条指令）。

---

## 编译器优化控制

### 编译选项调优

`--use_fast_math` 开启全部快速数学（包括不安全的优化如 flush-denormals-to-zero）。如果只需要部分优化，可以单独开 `--ftz=true`（非规格化数清零）、`--prec-div=false`（低精度除法）、`--prec-sqrt=false`（低精度平方根）。

### 内联控制

`__forceinline__` 强制内联小函数，消除函数调用开销。`__noinline__` 阻止内联，用于降低代码膨胀导致的指令缓存压力。

### PTX / SASS 分析

用 `cuobjdump --dump-sass` 查看最终机器码，确认关键路径是否如预期使用了 FMA、LDG、STS 等高效指令，是否出现了意外的类型转换或寄存器溢出。`ncu`（Nsight Compute）可以做更细粒度的指令级分析。

- **补充（近两年实践）**：异步搬运/新架构优化时，建议同时核查 PTX 层是否出现预期的异步/矩阵相关指令路径，避免"代码写了但编译退化"。

### `__restrict__` 关键字

告诉编译器指针不会互相 alias，编译器因此可以更激进地做指令重排和寄存器缓存，效果有时非常显著。

---

## 分支与控制流优化

### 谓词化（Predication）

对于很短的分支（比如只有一两条指令），编译器会自动把分支转为谓词指令——两条路径都执行，但只有满足条件的那条写回结果。这避免了 warp divergence，但浪费了计算。只适合分支体很短的情况。

### Select 指令替代分支

`condition ? a : b` 通常被编译为无分支的 select 指令。手动把 if-else 改写为三目运算符或算术表达式（如 `result = mask * a + (1-mask) * b`）可以帮助编译器生成无分支代码。

### 按 Warp 重组数据

如果分支不可避免，通过预排序或分组让同类数据聚集在同一个 warp 中。比如在粒子模拟中，把活跃粒子和非活跃粒子分开排列，这样非活跃粒子所在的 warp 可以整体跳过计算。

### Early Exit

在归约或搜索类 kernel 中，如果 warp 内所有线程都已经满足终止条件，可以通过 `__all_sync()` 检测后直接 return，避免无意义的后续计算。

---

## 异步计算与重叠

### 计算与访存重叠

这是调度器的核心能力。当一个 warp 在等待内存返回时，调度器切换到另一个就绪的 warp 执行计算指令。提升这种重叠能力的方法包括：增加 occupancy、增加每个线程的独立指令数（ILP）、以及软件流水线。

### Warp Specialization

把一个 block 内的 warp 分为"搬运 warp"和"计算 warp"。搬运 warp 专门负责 global→shared 的数据加载，计算 warp 专门做运算，两者通过 shared memory 和简单的标志位协同。这是一种手动实现的 producer-consumer 模型，在矩阵乘法等场景中可以获得比传统方法更好的资源利用率。

- **补充（架构演进）**：在 Hopper/Blackwell 等新架构上，这类 producer-consumer 分工常与更深的异步 pipeline 配合，收益更稳定，但对同步语义正确性要求更高。

---

## 验证清单（NCU）

为避免"只改代码不验指标"，计算优化建议至少配套以下检查：

- **Tensor Core 路径**：确认是否出现预期 MMA/WGMMA 指令路径，且相关计算管线利用率提升。
- **指令效率**：关注 `Issue Slot Utilization`、`Eligible Warps Per Cycle` 是否改善。
- **分支质量**：关注 `Warp Execution Efficiency` 与分支相关 stall 是否改善。

常见误判：

- 只看 occupancy 升高，不看 kernel latency，可能出现"occupancy 上去但性能变差"。
- 只看某个单项指标改善，忽略了同步或访存路径退化导致总体变慢。

优化策略入口：`cuda/SKILL.md` — 按瓶颈类型给出实施优先级。
