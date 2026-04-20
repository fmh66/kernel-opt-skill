# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 4096, 'K': 4096, 'N': 4096} |
| **Execution Time** | 47.8810 ms ± 0.2603 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 79.2728 |
| Memory Throughput (% of peak) | 25.6569 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.87e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.86e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.47e+09 |
| L1 Global Load Bandwidth (bytes/s) | 3.65e+11 |
| L1 Global Store Bandwidth (bytes/s) | 1.43e+09 |
| L2 Total Bandwidth (bytes/s) | 3.66e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 49.1264 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.9277 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2154 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 66.6760 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 1024.0000 |
| Grid Size | 16384.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 8192.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 195.0476 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 21.5373 |
| Eligible Warps / Cycle | 0.9893 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.0076 |
| Stall: Long Scoreboard | 5.6563 |
| Stall: Short Scoreboard | 0.0864 |
| Stall: Math Pipe Throttle | 0.6202 |
| Stall: Wait | 1.8012 |
| Stall: No Instruction | 0.0981 |
| Stall: Not Selected | 3.5936 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6.76e+07 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 774.4432 |
| LSU Pipe Utilization (% of peak) | 19.9118 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3.23e+07 |
| Shared Memory Bandwidth (bytes/s) | 6.39e+12 |

**Kernel name:** `tiled_gemm`