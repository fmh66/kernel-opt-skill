# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 32.9575 |
| Memory Throughput (% of peak) | 92.2880 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.73e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.41e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.32e+11 |
| L1 Global Load Bandwidth (bytes/s) | 1.01e+12 |
| L1 Global Store Bandwidth (bytes/s) | 6.72e+11 |
| L2 Total Bandwidth (bytes/s) | 1.11e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 53.9672 |
| L2 Hit Rate (%) | 68.7000 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 10.6358 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2872 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 94.7855 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 32.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 28.7518 |
| Eligible Warps / Cycle | 0.5908 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 6.5385 |
| Stall: Long Scoreboard | 18.0320 |
| Stall: Short Scoreboard | 3.7675 |
| Stall: Math Pipe Throttle | 0.3698 |
| Stall: Wait | 1.9576 |
| Stall: No Instruction | 0.3331 |
| Stall: Not Selected | 1.0745 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.27e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 193.9620 |
| FMUL Throughput (per cycle) | 99.0444 |
| FFMA Throughput (per cycle) | 176.0790 |
| LSU Pipe Utilization (% of peak) | 8.5953 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 118764.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.64e+10 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 124896.0000 |

**Kernel name:** `softmax_block_per_row`