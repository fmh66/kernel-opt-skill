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
| SM Throughput (% of peak) | 30.3247 |
| Memory Throughput (% of peak) | 90.2431 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.58e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.34e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.24e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.86e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.57e+11 |
| L2 Total Bandwidth (bytes/s) | 1.08e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 54.6329 |
| L2 Hit Rate (%) | 68.5943 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.9392 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2670 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 94.7216 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 21.0000 |
| Static Shared Memory (bytes) | 128.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 26.7248 |
| Eligible Warps / Cycle | 0.3991 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.6132 |
| Stall: Long Scoreboard | 29.7080 |
| Stall: Short Scoreboard | 1.8970 |
| Stall: Math Pipe Throttle | 0.2208 |
| Stall: Wait | 2.5150 |
| Stall: No Instruction | 0.1540 |
| Stall: Not Selected | 0.5031 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.35e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 190.2868 |
| FMUL Throughput (per cycle) | 97.1677 |
| FFMA Throughput (per cycle) | 172.7426 |
| LSU Pipe Utilization (% of peak) | 7.6646 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 130437.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.60e+10 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 127680.0000 |

**Kernel name:** `softmax_v1`