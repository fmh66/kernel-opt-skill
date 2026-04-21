# CUDA Kernel Optimization Demo

**Environment**: NVIDIA RTX A6000 (CC 8.6) · CUDA 12.6 · ncu 2024.3.2.0 · PyTorch 2.11.0+cu126

---

## Softmax

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

## GEMM

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

## MHA

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
