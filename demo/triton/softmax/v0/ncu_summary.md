# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.1477 ms ± 0.0020 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 11.2018 |
| Memory Throughput (% of peak) | 92.8776 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.77e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.43e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.34e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.43e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.43e+11 |
| L2 Total Bandwidth (bytes/s) | 6.88e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.1810 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 4.2681 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0944 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 97.0020 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 23.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 10.1587 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 9.4512 |
| Eligible Warps / Cycle | 0.1672 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 13.7290 |
| Stall: Long Scoreboard | 46.7730 |
| Stall: Short Scoreboard | 31.1603 |
| Stall: Math Pipe Throttle | 0.1114 |
| Stall: Wait | 1.6502 |
| Stall: No Instruction | 0.0648 |
| Stall: Not Selected | 0.7830 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 122.6721 |
| FMUL Throughput (per cycle) | 89.2161 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 2.9624 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 18548.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.71e+09 |

**Kernel name:** `softmax_kernel`