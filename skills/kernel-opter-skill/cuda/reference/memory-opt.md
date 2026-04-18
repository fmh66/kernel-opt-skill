CUDA kernel内存优化的完整方案：

---
## 一、全局内存（Global Memory）访问优化

**1. 合并访问（Coalesced Access）**
同一warp的32个线程访问连续、对齐的地址，硬件合并为最少的内存事务。理想情况下一次128B事务服务整个warp。反例是stride访问或随机访问，会导致事务数暴增。

**2. 对齐访问（Aligned Access）**
访问的起始地址对齐到128B（或32B的sector）边界。未对齐会浪费带宽，因为硬件以sector为单位读取，不对齐意味着额外sector被拉入但无用。

**3. 向量化访存（Vectorized Load/Store）**
使用 `float2`、`float4`、`int4`、`double2` 等宽类型，单条指令读写128-bit。好处是减少指令数、提升每事务有效字节数。需要保证地址对齐到向量宽度。

**4. 只读数据路径**

* `__ldg()` 内建函数：显式走只读数据缓存（L1 texture缓存），绕过L1常规路径，对不规则访问模式有更好的缓存行为。
  * **补充（架构限定）**：在新架构和新编译器下，`__ldg()`不再是"无脑必开"优化，收益依赖访问模式与缓存行为，建议以Nsight Compute数据为准。
* `const __restrict__` 指针修饰：告诉编译器该指针所指数据不会被写入且无别名，编译器自动选择 `__ldg()` 路径。

**5. L2缓存优化（Compute Capability 8.0+）**

* **L2 Persistence**：使用 `cudaAccessPolicyWindow` API将热点数据"钉"在L2缓存中，防止被冷数据驱逐。适合数据量小但反复访问的场景（如lookup table、embedding）。
* **L2 访问属性**：`cudaAccessProperty::cudaAccessPropertyPersisting` vs `Streaming`，精细控制不同数据的L2驻留策略。

**6. Sector化理解**
从Volta架构开始，L1缓存以32B sector为粒度工作（而非传统128B cache line整体fetch）。这意味着不连续访问的代价比Kepler/Maxwell时代降低了，但连续访问仍然最优。

---

## 二、共享内存（Shared Memory）优化

**7. Tiling（分块）**
将全局内存中的数据按tile搬到Shared Memory，在片上多次复用。矩阵乘法的经典优化——每个tile只从全局内存读一次，在Shared Memory中被复用 (O(N)) 次。

**8. Bank Conflict消除**
Shared Memory被划分为32个bank，每个bank宽4B（或在某些模式下8B）。同一warp中多个线程访问同一bank的不同地址会串行化。解决方式：
* **Padding**：给二维数组每行末尾加一个元素，如 `__shared__ float s[32][33]`，错开bank映射。
* **Swizzle/XOR索引**：用异或操作重映射索引，是更高级且不浪费空间的方案。

**9. 异步拷贝（Async Copy）**

* `cp.async`（CUDA 11+）：Global → Shared Memory的搬运由硬件DMA完成，不占用寄存器和计算单元。可以与计算完全重叠。
* `cuda::memcpy_async`（C++ API）：语义更清晰的封装。
* **多级流水线（Multi-stage Pipeline）**：分配多个Shared Memory buffer，一个在加载数据，一个在计算，实现软件流水线，极大隐藏全局内存延迟。

**10. Shared Memory容量配置**
使用 `cudaFuncSetAttribute` 将L1/Shared Memory的比例调向Shared Memory一侧（如从48KB提升到100KB+），适合Shared Memory需求大的kernel。Ampere架构Shared Memory最大可达164KB。

---

## 三、常量内存（Constant Memory）

**11. `__constant__` 内存**
总共64KB，硬件有专用缓存。当warp内所有线程读同一个地址时效率最高（广播机制，一次读取服务32个线程）。适合存储kernel参数、卷积核权重等小型只读数据。如果同一warp内线程读不同地址，则串行化，性能反而差。

---

## 四、纹理内存（Texture Memory）

**12. Texture / Surface 对象**

* 硬件针对2D空间局部性优化缓存（Morton/Z-order布局），适合图像处理、模板计算等二维访问模式。
* 自动处理边界条件（clamp、wrap模式），免去手写边界判断。
* 现代GPU上 `__ldg()` 在大多数一维场景可以替代texture，但二维空间局部性场景texture仍有优势。

---

## 五、数据布局优化

**13. SoA vs AoS**

* **AoS（Array of Structures）**：`struct { float x, y, z; } particles[N]`——同一粒子的字段连续存放。warp访问同一字段时地址不连续，无法合并。
* **SoA（Structure of Arrays）**：`float x[N], y[N], z[N]`——同一字段的所有粒子连续存放。warp访问时天然合并。
* **CUDA中几乎总是优选SoA**，差距可以达到数倍。

**14. Padding与对齐**
对二维数组的行宽做padding（`cudaMallocPitch` / `cudaMalloc3D`），保证每一行的起始地址对齐到合并访问边界。

---

## 六、访存与计算重叠

**15. 双缓冲 / 多级流水线**
在Shared Memory或寄存器中分配两组buffer：

* 阶段1：buffer A 计算，buffer B 加载下一批数据
* 阶段2：buffer B 计算，buffer A 加载下一批数据

这是所有高性能GEMM实现（如cuBLAS、CUTLASS）的核心技术。

**16. CUDA Streams重叠**

* 不同stream之间的kernel执行、H2D拷贝、D2H拷贝可以并行。
* 把大数据分chunk，在多个stream中做"拷贝-计算-回拷"流水线。

**17. Prefetch**

* 对Global Memory使用 `__builtin_prefetch` 或手动用额外线程提前加载下一次迭代的数据。
* Unified Memory场景下用 `cudaMemPrefetchAsync` 显式触发页迁移，避免按需缺页的高延迟。

---

## 七、减少不必要的内存访问

**18. Kernel Fusion**
把相邻的producer-consumer kernel合并为一个，中间结果保留在寄存器或Shared Memory中，消除一次完整的Global Memory往返。这往往是最高收益的单项优化。

**19. Warp Shuffle（`__shfl_sync` 系列）**
warp内线程直接交换寄存器值，无需经过Shared Memory。适用于warp级归约（reduction）、前缀和（scan）、广播。延迟约为1个时钟周期，比Shared Memory还快且不存在bank conflict。

**20. 协作组（Cooperative Groups）**
灵活定义比warp更小或跨block的同步组，精确控制同步粒度，避免不必要的 `__syncthreads()` 全block同步带来的等待开销。

**21. 寄存器缓存（Register Caching）**
当每线程处理的元素数可在编译期确定时，用模板参数 `N4` 将所有元素装入寄存器数组（`float4 regs[N4]`），在寄存器中完成全部中间计算（max、exp、sum），最终只做一次 DRAM 写入，彻底消除中间写回。需配合 `__launch_bounds__` 控制寄存器分配，N4 不宜超过 8。

---

## 八、Pinned Memory 与 Unified Memory

**21. Pinned Memory（Page-locked）**
`cudaHostAlloc` 或 `cudaMallocHost` 分配的锁页内存，H2D/D2H传输带宽比可分页内存高得多（可达2倍），且支持异步传输。

**22. Unified Memory优化**
用 `cudaMemPrefetchAsync` 提前迁移到目标设备；用 `cudaMemAdvise`（如 `cudaMemAdviseSetReadMostly`）给驱动提供访问模式提示，避免按需缺页的高延迟。

---

## 九、验证清单（NCU）

内存优化建议至少配套以下验证：

1. **带宽利用率**：关注 `Memory SOL %`、`DRAM Throughput`，确认是否接近预期上界。
2. **访问质量**：关注 `Global Load/Store Efficiency`、`Sectors/Request`，确认 coalescing/对齐是否改善。
3. **缓存行为**：关注 `L1 Hit Rate`、`L2 Hit Rate`，确认优化方向与局部性变化一致。
4. **Shared 路径健康度**：关注 `Shared Memory Efficiency`，确认 bank conflict 是否下降。
5. **整体收益**：最终以 kernel latency（avg/median）判断，不只看单个子指标。

优化策略入口：`cuda/SKILL.md` — 按瓶颈类型给出实施优先级。
