# CUDA Kernel 延迟优化（Latency-Bound）

---

## Occupancy 与 Launch 配置

### Launch Configuration 调优

block size 的选择直接影响 occupancy 和硬件利用率。通常选择 128/256/512，但最优值取决于 kernel 的资源消耗。CUDA Occupancy Calculator 和 `cudaOccupancyMaxPotentialBlockSize` API 可以辅助决策。

### 控制寄存器使用量

每个线程用的寄存器越多，同时驻留的 warp 就越少（occupancy 下降），调度器隐藏延迟的能力就越弱。用 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 提示编译器控制寄存器分配。

### 寄存器溢出（Register Spill）的处理

当寄存器不够时，编译器会把变量溢出到 local memory（实际是全局内存，经 L1/L2 缓存）。溢出严重时性能断崖式下降。通过减少活跃变量数量、缩小循环展开因子、或拆分 kernel 来缓解。用 `--ptxas-options=-v` 编译选项查看寄存器和溢出情况。

### Occupancy 不是越高越好

高 occupancy 意味着更多 warp 可以隐藏延迟，但也意味着每个线程可用的寄存器和 shared memory 更少。对于计算密集型 kernel，适度降低 occupancy 换取更多寄存器（从而减少 spill 和提升 ILP）往往性能更好。需要通过实测找到最优平衡点。

---

## ILP（指令级并行）提升

### 增加每线程独立工作量

让每个线程处理多个数据元素（thread coarsening），在寄存器层面完成更多计算后再写回。这样调度器在单个 warp 等待时有更多独立指令可以发射，提升 ILP。

### 循环展开

用 `#pragma unroll` 或 `#pragma unroll N` 展开循环。展开后减少循环控制指令（比较、跳转），同时暴露更多独立指令给调度器，提升 ILP。

### 软件流水

循环体内把当前迭代的计算与下一迭代的数据预取重叠起来，最大化功能单元利用率。

---

## 同步优化

### 减少 `__syncthreads()` 次数

最直接的方式。如果 warp 内的数据访问模式天然不存在跨 warp 依赖，同步就是多余的。审查每一个 `__syncthreads()` 调用，确认其必要性。

### Warp 级同步替代 Block 级同步

一个 warp 内的 32 个线程天然是 lockstep 执行的（Volta 之后有 independent thread scheduling，但 warp 级原语仍然有效）。用 `__syncwarp()` 替代 `__syncthreads()` 可以把同步粒度从整个 block 缩小到单个 warp，开销大幅降低。

- **补充（语义边界）**：在 Volta+ 独立线程调度下，不应把"天然 lockstep"当作隐式同步保证；warp 内协作仍应使用正确 mask 与显式同步点确保可见性与收敛。

### Warp Shuffle

warp 内线程之间交换数据完全不需要经过共享内存，没有 bank conflict 问题，延迟极低。适用于归约、前缀和、广播等模式。这直接消除了"写 shared → sync → 读 shared"的三步开销。

### Cooperative Groups

CUDA 9 引入的协作组机制，可以定义任意粒度的线程组并在该组内同步，比如只同步一个 tile（比如 8 个线程），避免不必要的全 block 等待。

### `cuda::barrier` / `cuda::pipeline`（Ampere+）

在异步拷贝和多级流水线中，用显式阶段同步替代"经验式同步"。核心思想是把 producer-consumer 的提交/等待点写清楚，避免偶现错误和性能抖动。

---

## 异步预取

### `cp.async` 预取

（CUDA 11+）Global → Shared Memory 的搬运由硬件 DMA 完成，不占用寄存器和计算单元，可与计算完全重叠。配合多级 buffer 实现软件流水线，极大隐藏全局内存延迟。

### `cudaMemPrefetchAsync`

Unified Memory 场景下提前触发页迁移，避免按需缺页的高延迟。

---

## 减少调度开销

### CUDA Graphs

当 kernel 链路结构稳定且反复执行时，Graph 可以降低 CPU 提交与 launch 开销，特别是小 kernel 密集场景。动态图场景要评估 capture/update 成本。

---

## 验证清单（NCU）

Latency-Bound 优化建议至少配套以下验证：

- **同步等待是否下降**：关注 `Stall Barrier` 与相关等待 stall 是否降低。
- **调度可发射性是否改善**：关注 `Eligible Warps Per Cycle` 是否提升。
- **Occupancy 变化**：关注 `Achieved Occupancy`，结合 kernel latency 判断是否改善。
- **寄存器溢出**：用 `--ptxas-options=-v` + NCU 检查 spill 是否下降。
- **整体收益**：最终以 kernel latency（avg/median）判断，不只看单个子指标。

常见误判：

- 只减少了 `__syncthreads()` 次数，但引入了数据可见性错误。
- 只看 occupancy 升高，不看 kernel latency，可能出现"occupancy 上去但性能变差"。
- 只看单个 stall 指标下降，没有检查整体 kernel latency 与 correctness。

优化策略入口：`cuda/SKILL.md` — 按瓶颈类型给出实施优先级。
