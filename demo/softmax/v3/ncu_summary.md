# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 22.4200 |
| Memory Throughput (% of peak) | 87.6190 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.39e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.36e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.03e+11 |
| L1 Global Load Bandwidth (bytes/s) | 6.19e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.10e+11 |
| L2 Total Bandwidth (bytes/s) | 7.32e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 21.5071 |
| L2 Hit Rate (%) | 53.8966 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 12.2994 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2395 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 88.5635 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 24.0000 |
| Static Shared Memory (bytes) | 256.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 23.9617 |
| Eligible Warps / Cycle | 0.3547 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 7.4186 |
| Stall: Long Scoreboard | 28.5838 |
| Stall: Short Scoreboard | 1.4893 |
| Stall: Math Pipe Throttle | 0.0963 |
| Stall: Wait | 1.4389 |
| Stall: No Instruction | 0.0724 |
| Stall: Not Selected | 0.4845 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 348160.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 234.9552 |
| FMUL Throughput (per cycle) | 364.3542 |
| FFMA Throughput (per cycle) | 97.2664 |
| LSU Pipe Utilization (% of peak) | 3.9678 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 37250.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.51e+10 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 135424.0000 |

**Kernel name:** `softmax_v3`