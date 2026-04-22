# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.1424 ms ± 0.0016 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 30.9153 |
| Memory Throughput (% of peak) | 91.3596 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.66e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.36e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.30e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.36e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.36e+11 |
| L2 Total Bandwidth (bytes/s) | 6.74e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.1168 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 18.2709 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.3306 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 96.9654 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 24.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 4160.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 33.0792 |
| Eligible Warps / Cycle | 0.7368 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 7.3099 |
| Stall: Long Scoreboard | 19.4427 |
| Stall: Short Scoreboard | 2.1193 |
| Stall: Math Pipe Throttle | 0.1222 |
| Stall: Wait | 1.0490 |
| Stall: No Instruction | 0.1107 |
| Stall: Not Selected | 1.2746 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 593920.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 456.1790 |
| FMUL Throughput (per cycle) | 295.4171 |
| FFMA Throughput (per cycle) | 1018.1585 |
| LSU Pipe Utilization (% of peak) | 4.9773 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 36732.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.02e+12 |

**Kernel name:** `softmax_v3`