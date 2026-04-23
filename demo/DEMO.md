# CUDA / Triton Kernel Optimization Demo

**Environment**: NVIDIA RTX A6000 (CC 8.6) · CUDA 12.6 · Triton 3.6.0 · ncu 2024.3.2.0 · PyTorch 2.11.0+cu126

---

## CUDA

### Softmax

**Shape**: N=10240, D=1024

| Version | Time (ms) | Speedup | SM Throughput (%) | Mem Throughput (%) | Occupancy (%) | Load Eff (%) | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 0.8992 | 1.00× | 2.99 | 33.2 | 16.6 | 12.5 | Latency | Naive (1 thread/row) |
| v1 | 0.2219 | 4.05× | 16.8 | 81.9 | 87.5 | 100 | Memory | 1 block/row + warp shuffle reduce |
| v2 | 0.1515 | 5.93× | — | 90.0 | 88.0 | 100 | Memory | Online softmax + float4 loads |
| v3 | **0.1424** | **6.32×** | — | **91.4** | **97.0** | 100 | Memory | Shared memory row cache + `__launch_bounds__` |

**Benchmark**: v3 vs PyTorch (N=10240, D=1024)

| Metric | v3 | PyTorch |
| --- | ---: | ---: |
| Time (ms) | **0.1469** | 0.2721 |
| Mem Throughput (%) | 91.9 | 92.6 |
| DRAM Bandwidth (GB/s) | 670 | 675 |
| Occupancy (%) | 94.3 | 94.5 |

v3 is **1.85× faster** than PyTorch at near-identical DRAM bandwidth — the gap is PyTorch dispatch overhead.

---

### GEMM

**Shape**: M=K=N=4096

| Version | Time (ms) | Speedup | SM Throughput (%) | Mem Throughput (%) | Occupancy (%) | FMA Util (%) | IPC | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 64.23 | 1.00× | 98.5 | 39.6 | 99.8 | 19.3 | 0.286 | Latency (Long SB=12.75) | Naive (non-coalesced, no smem) |
| v1 | 47.88 | 1.34× | 79.3 | 25.7 | 66.7 | 7.9 | 0.215 | Latency (Barrier=3.01) | Shared memory tiling (32×32) |
| v2 | 11.64 | 5.52× | 55.0 | 53.6 | 65.3 | 38.6 | 0.525 | Balanced (Short SB=2.97) | 2D register blocking (TM=4×TN=4, BK=8) |
| v3 | **9.43** | **6.81×** | **72.6** | **73.9** | 32.7 | **55.8** | **0.717** | Balanced | BK=16 + TM=8 + smem padding + float4 store |

**Benchmark**: v3 vs PyTorch/cuBLAS (M=K=N=4096)

| Metric | v3 | PyTorch |
| --- | ---: | ---: |
| Time (ms) | 9.41 | **6.18** |
| SM Throughput (%) | 72.6 | 72.5 |
| Mem Throughput (%) | 74.0 | 74.3 |
| DRAM Bandwidth (GB/s) | 539 | 541 |
| Occupancy (%) | 32.7 | 32.7 |

v3 is **1.52× slower** than cuBLAS at near-identical hardware utilization — the gap is cuBLAS's assembly-level ILP and double buffering.

---

### MHA

**Shape**: N=1024, d_model=512, num_heads=8 (d_k=64)

| Version | Time (ms) | Speedup | SM Throughput (%) | Occupancy (%) | Load Eff (%) | Long SB | Barrier | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 15.45 | 1.00× | 87.0 | 62.7 | 17.6 | 4.85 | **14.51** | Latency (Barrier) | Naive (serial QK^T in t==0) |
| v1 | 4.15 | 3.72× | 17.8 | 63.9 | 22.2 | 6.73 | 2.06 | Latency (Short SB) | Parallel QK^T dot product |
| v2 | 1.86 | 8.31× | 33.4 | 64.8 | 66.7 | **21.39** | 0.73 | Latency (Long SB) | float4 + 4-way FMA ILP |
| v3 | 1.91 | 8.09× | 32.4 | 81.0 | 66.7 | 18.12 | 1.82 | Latency (Long SB) | BLOCKQ=2 (L1 reuse, slight regression) |
| v4 | 1.99 | 7.76× | 87.9 | 20.5 | **100** | 2.02 | 0.53 | Compute (low occ.) | Flash Attn 2 + smem K/V tiling |
| v5 | **1.51** | **10.23×** | **96.1** | **40.3** | **100** | **1.56** | 1.37 | Compute | v4 + BLOCKQ=2 (2× warps/SM) |

**Benchmark**: v5 vs PyTorch Flash Attention (N=1024, d_model=512, num_heads=8)

| Metric | v5 | PyTorch |
| --- | ---: | ---: |
| Time (ms) | 1.4753 | **0.5153** |
| SM Throughput (%) | 96.1 | 96.1 |
| Mem Throughput (%) | 0.94 | 0.94 |
| DRAM Bandwidth (GB/s) | 6.82 | 6.83 |
| Occupancy (%) | 40.3 | 40.3 |

v5 is **2.86× slower** than PyTorch at identical hardware utilization — PyTorch uses Tensor Cores (FP16/BF16) while v5 runs FP32.

---

## Triton

### GEMM

**Shape**: M=N=K=10240

| Version | Time (ms) | Speedup | SM Throughput (%) | Mem Throughput (%) | TC Util (%) | FMA Util (%) | Occupancy (%) | Regs/Thread | L2 Hit (%) | Math Throttle | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 129.74 | 1.00× | 73.95 | 82.80 | 0.00 | 63.61 | 25.31 | 96 | 47.7 | 0.04 | Memory | Naive (FMA, no TC, no pipelining) |
| v1 | 44.26 | 2.93× | **82.53** | 75.39 | **82.45** | 1.44 | 16.50 | 168 | 66.95 | 7.19 | Compute | `tf32` MMA + grouped L2 swizzle + `tl.range(num_stages=3)` |
| v2 | 43.92 | 2.95× | 80.61 | 76.99 | 79.07 | 1.37 | 8.10 | 241 | 65.68 | 3.01 | Compute | `@autotune` (broken: `range()` bug disabled pipelining) |
| v3 | **42.25** | **3.07×** | 75.32 | 73.23 | 76.91 | 2.45 | 16.85 | **128** | **82.00** | 5.20 | Compute | Fixed `tl.range()` + correct autotune + boundary mask fix |

**Benchmark**: v3 vs `torch.mm` (M=N=K=10240)

| Metric | v3 | torch.mm |
| --- | ---: | ---: |
| Time (ms) | 42.77 | 97.36 |
| SM Throughput (%) | 76.12 | 75.85 |
| Mem Throughput (%) | 77.71 | 80.09 |
| DRAM Bandwidth (GB/s) | 566 | 584 |
| Occupancy (%) | 16.66 | 16.66 |

v3 is **2.28× faster** than `torch.mm` (which includes `copy_()` overhead). cuBLAS achieves ~7 ms for this shape — a **~6× gap** remains, limited by register pressure (128 regs/thread → 1 block/SM) and Math Pipe Throttle stall (5.2).

---

### MHA

**Shape**: N=1024, d_model=1024, num_heads=16 (d_k=64)

| Version | Time (ms) | Speedup | SM Throughput (%) | Mem Throughput (%) | TC Util (%) | Occupancy (%) | Regs/Thread | Long SB | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 135.40 | 1.00× | 39.96 | 13.99 | 0 | 49.90 | 80 | 15.74 | Latency | Bug-fixed baseline; grid (H,N,d_k) — 64× redundant attention row computation |
| v1 | 3.68 | 36.8× | 35.86 | 87.91 | 0 | 65.66 | 63 | 25.48 | Memory | Grid (H,N): one program computes full output row, eliminates 64× redundancy |
| v2 | 0.73 | 185× | 20.21 | 12.47 | 0 | 8.33 | 255 | 0.09 | Latency (low occ.) | Flash Attention tiling + `tl.dot` for QK^T and PV (BLOCK_M=32) |
| v3 | **0.22** | **626×** | 32.12 | 30.06 | **41.35** | 8.35 | 144 | 0.38 | Balanced (TC) | `allow_tf32=True` + `@triton.autotune` (BLOCK_M=64/BLOCK_N=64/num_warps=8) |

**Benchmark**: v3 vs PyTorch reference (N=1024, d_model=1024, num_heads=16)

| Metric | v3 | PyTorch |
| --- | ---: | ---: |
| Time (ms) | **0.2183** | 0.8989 |
| SM Throughput (%) | 32.23 | 32.18 |
| Mem Throughput (%) | 29.95 | 29.77 |
| DRAM Bandwidth (GB/s) | 218 | 217 |
| Occupancy (%) | 8.33 | 8.33 |

v3 is **4.12× faster** than PyTorch reference at identical hardware utilization — the speedup comes from eliminating 64× algorithmic redundancy (grid fix), Flash Attention tiling, and TF32 Tensor Core acceleration.

---

### Softmax

**Shape**: N=10240, D=1024

| Version | Time (ms) | Speedup | SM Throughput (%) | Mem Throughput (%) | Occupancy (%) | Regs/Thread | Long SB (%) | Barrier (%) | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | **0.1477** | **1.00×** | 11.2 | 92.9 | 97.0 | 23 | 46.8 | 13.7 | Memory | Already optimal: single-pass fused softmax at 92.9% memory SOL |
| v1 | 0.1632 | 0.90× | 11.0 | 91.2 | 97.8 | 22 | 51.6 | 15.1 | Memory | `tl.exp2` attempt — EX2 and EXP share identical MUFU throughput; extra FMUL hurts |
| v2 | 0.1610 | 0.92× | 11.0 | 92.7 | 97.0 | 36 | 31.1 | 24.9 | Memory | 2 rows/program: hides DRAM latency (Long SB ↓) but doubles Barrier stalls |
| v3 | 0.1491 | 0.99× | 7.2 | 93.2 | 65.4 | 31 | 31.6 | 8.4 | Memory | `num_warps=2` (64 threads): fewer reduction stages, lower occupancy offsets gain |
| v4 | 0.1496 | 0.99× | 22.4 | 93.0 | 97.4 | 28 | 18.8 | 4.5 | Memory | Online softmax via `tl.reduce`: collapses all stalls but 10240 divergent branch targets + 3× extra `exp()` cancel savings |

**Benchmark**: v0 (best) vs PyTorch (N=10240, D=1024)

| Metric | v0 | PyTorch |
| --- | ---: | ---: |
| Time (ms) | **0.1479** | 0.2648 |
| Mem Throughput (%) | 92.6 | 93.2 |
| DRAM Bandwidth (GB/s) | 675 | 679 |
| Occupancy (%) | 94.9 | 94.9 |

v0 is **1.79× faster** than PyTorch at near-identical DRAM bandwidth — PyTorch uses multiple kernel passes (doubling DRAM traffic) while v0 fuses max-reduction, exp, sum-reduction, and normalization into a single pass. No further improvement was achievable within 4 iterations.
