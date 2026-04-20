# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 4096, 'K': 4096, 'N': 4096} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 96.4831 |
| Memory Throughput (% of peak) | 53.0979 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 3.87e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.86e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.55e+09 |
| L1 Global Load Bandwidth (bytes/s) | 7.55e+11 |
| L1 Global Store Bandwidth (bytes/s) | 1.50e+09 |
| L2 Total Bandwidth (bytes/s) | 7.68e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0777 |
| L2 Hit Rate (%) | 49.7920 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 10.6286 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.3088 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 99.8181 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 65536.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 2048.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 130.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 30.8812 |
| Eligible Warps / Cycle | 1.1912 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 6.7834 |
| Stall: Long Scoreboard | 8.7248 |
| Stall: Short Scoreboard | 0.4820 |
| Stall: Math Pipe Throttle | 0.1455 |
| Stall: Wait | 1.8967 |
| Stall: No Instruction | 0.0759 |
| Stall: Not Selected | 2.7045 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.35e+08 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 864.0671 |
| LSU Pipe Utilization (% of peak) | 24.2558 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2.81e+08 |
| Shared Memory Bandwidth (bytes/s) | 4.22e+12 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 4.48e+07 |

**Kernel name:** `tiled_gemm`