# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 26.9679 |
| Memory Throughput (% of peak) | 91.8183 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.69e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.39e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.30e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.39e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.39e+11 |
| L2 Total Bandwidth (bytes/s) | 6.80e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.1810 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.8230 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1986 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 97.5264 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 19.0000 |
| Static Shared Memory (bytes) | 128.0000 |
| Dynamic Shared Memory (bytes) | 4096.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 19.8761 |
| Eligible Warps / Cycle | 0.2638 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 11.5413 |
| Stall: Long Scoreboard | 32.3244 |
| Stall: Short Scoreboard | 6.6900 |
| Stall: Math Pipe Throttle | 0.0605 |
| Stall: Wait | 2.7124 |
| Stall: No Instruction | 0.1759 |
| Stall: Not Selected | 0.3296 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.27e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 195.0584 |
| FMUL Throughput (per cycle) | 99.6043 |
| FFMA Throughput (per cycle) | 177.0743 |
| LSU Pipe Utilization (% of peak) | 7.1072 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 35975.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.71e+12 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 123648.0000 |

**Kernel name:** `softmax_v2`