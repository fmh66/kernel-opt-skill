# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 135.3997 ms ± 0.9566 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 39.9607 |
| Memory Throughput (% of peak) | 13.9961 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.02e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.02e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.74e+08 |
| L1 Global Load Bandwidth (bytes/s) | 2.33e+12 |
| L1 Global Store Bandwidth (bytes/s) | 2.52e+08 |
| L2 Total Bandwidth (bytes/s) | 2.31e+12 |
| Global Load Efficiency (%) | 90.3114 |
| Global Store Efficiency (%) | 12.5000 |
| L1 Hit Rate (%) | 0.8459 |
| L2 Hit Rate (%) | 95.6001 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 11.6756 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2540 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 49.9000 |
| Theoretical Occupancy (%) | 50.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1.05e+06 |
| Registers / Thread | 80.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 512.0000 |
| Waves / SM | 2080.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 25.4091 |
| Eligible Warps / Cycle | 0.3286 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.2719 |
| Stall: Long Scoreboard | 15.7446 |
| Stall: Short Scoreboard | 1.4718 |
| Stall: Math Pipe Throttle | 0.0945 |
| Stall: Wait | 1.1789 |
| Stall: No Instruction | 0.0093 |
| Stall: Not Selected | 0.2928 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6.29e+07 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 506.7565 |
| FMUL Throughput (per cycle) | 244.3290 |
| FFMA Throughput (per cycle) | 246.1389 |
| LSU Pipe Utilization (% of peak) | 9.9551 |
| Warp Execution Efficiency | 31.9966 |
| L1 Bank Conflicts (total) | 7.49e+08 |
| Shared Memory Bandwidth (bytes/s) | 7.44e+10 |

**Kernel name:** `fused_mha_kernel`