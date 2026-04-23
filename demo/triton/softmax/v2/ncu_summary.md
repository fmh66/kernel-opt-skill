# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.1610 ms ± 0.0016 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 10.9734 |
| Memory Throughput (% of peak) | 92.7012 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.76e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.43e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.33e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.43e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.43e+11 |
| L2 Total Bandwidth (bytes/s) | 6.87e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.1769 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 4.2468 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0907 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 97.0398 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 5120.0000 |
| Registers / Thread | 36.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 5.0794 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 9.1058 |
| Eligible Warps / Cycle | 0.1838 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 24.8805 |
| Stall: Long Scoreboard | 31.0980 |
| Stall: Short Scoreboard | 28.1227 |
| Stall: Math Pipe Throttle | 0.1356 |
| Stall: Wait | 1.6674 |
| Stall: No Instruction | 0.0503 |
| Stall: Not Selected | 0.9912 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 122.4387 |
| FMUL Throughput (per cycle) | 89.0463 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 2.9060 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 16375.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.70e+09 |

**Kernel name:** `softmax_kernel`