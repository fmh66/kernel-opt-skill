# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.1515 ms ± 0.0027 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 28.3745 |
| Memory Throughput (% of peak) | 89.9832 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.56e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.44e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.12e+11 |
| L1 Global Load Bandwidth (bytes/s) | 6.38e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.19e+11 |
| L2 Total Bandwidth (bytes/s) | 7.54e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 21.4697 |
| L2 Hit Rate (%) | 54.6532 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 17.6758 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.3062 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 88.0199 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 28.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 64.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 30.6331 |
| Eligible Warps / Cycle | 0.5236 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.7423 |
| Stall: Long Scoreboard | 23.2498 |
| Stall: Short Scoreboard | 0.7048 |
| Stall: Math Pipe Throttle | 0.1051 |
| Stall: Wait | 1.0044 |
| Stall: No Instruction | 0.1113 |
| Stall: Not Selected | 0.6688 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 430080.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 432.6072 |
| FMUL Throughput (per cycle) | 280.1522 |
| FFMA Throughput (per cycle) | 965.5479 |
| LSU Pipe Utilization (% of peak) | 4.1344 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 39471.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.56e+10 |

**Kernel name:** `softmax_v2`