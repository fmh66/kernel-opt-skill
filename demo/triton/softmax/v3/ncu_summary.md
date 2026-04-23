# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.1491 ms ± 0.0018 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 7.2168 |
| Memory Throughput (% of peak) | 93.1552 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.79e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.43e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.36e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.43e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.43e+11 |
| L2 Total Bandwidth (bytes/s) | 6.88e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.0785 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.8215 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0764 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 65.3833 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 31.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 8.0000 |
| Waves / SM | 7.6190 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 7.6575 |
| Eligible Warps / Cycle | 0.1236 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 8.3910 |
| Stall: Long Scoreboard | 31.5355 |
| Stall: Short Scoreboard | 31.0577 |
| Stall: Math Pipe Throttle | 0.0577 |
| Stall: Wait | 1.6624 |
| Stall: No Instruction | 0.0377 |
| Stall: Not Selected | 0.6134 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 103.0376 |
| FMUL Throughput (per cycle) | 89.1136 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 1.5939 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 19704.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.69e+09 |

**Kernel name:** `softmax_kernel`