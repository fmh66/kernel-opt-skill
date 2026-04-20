# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.2219 ms ± 0.0070 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 16.8314 |
| Memory Throughput (% of peak) | 81.8511 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.97e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.30e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.67e+11 |
| L1 Global Load Bandwidth (bytes/s) | 6.99e+11 |
| L1 Global Store Bandwidth (bytes/s) | 4.66e+11 |
| L2 Total Bandwidth (bytes/s) | 8.84e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 43.7189 |
| L2 Hit Rate (%) | 62.2289 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.3820 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1771 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 87.5463 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 21.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 10.1587 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 17.7293 |
| Eligible Warps / Cycle | 0.2219 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 7.4464 |
| Stall: Long Scoreboard | 52.5288 |
| Stall: Short Scoreboard | 1.2069 |
| Stall: Math Pipe Throttle | 0.0889 |
| Stall: Wait | 2.4927 |
| Stall: No Instruction | 0.1496 |
| Stall: Not Selected | 0.2538 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.23e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 118.8101 |
| FMUL Throughput (per cycle) | 60.8308 |
| FFMA Throughput (per cycle) | 129.2654 |
| LSU Pipe Utilization (% of peak) | 4.7054 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 108712.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.92e+09 |

**Kernel name:** `softmax_v1`