# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 4096, 'K': 4096, 'N': 4096} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 98.5172 |
| Memory Throughput (% of peak) | 38.4853 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.81e+11 |
| DRAM Read Bandwidth (bytes/s) | 2.79e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.12e+09 |
| L1 Global Load Bandwidth (bytes/s) | 4.44e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.08e+09 |
| L2 Total Bandwidth (bytes/s) | 5.59e+11 |
| Global Load Efficiency (%) | 56.2500 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 87.3797 |
| L2 Hit Rate (%) | 49.9906 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 19.2800 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2858 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 99.7199 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 65536.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 130.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 28.5844 |
| Eligible Warps / Cycle | 1.3305 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 12.6711 |
| Stall: Short Scoreboard | 0.0031 |
| Stall: Math Pipe Throttle | 0.0411 |
| Stall: Wait | 2.0134 |
| Stall: No Instruction | 0.0406 |
| Stall: Not Selected | 3.6549 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.38e+08 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 661.7931 |
| LSU Pipe Utilization (% of peak) | 24.6720 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2.11e+09 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 6.20e+07 |

**Kernel name:** `naive_gemm`