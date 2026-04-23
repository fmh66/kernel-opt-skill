# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.1496 ms ± 0.0047 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 22.3529 |
| Memory Throughput (% of peak) | 92.9551 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.78e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.44e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.34e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.44e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.44e+11 |
| L2 Total Bandwidth (bytes/s) | 6.90e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.1353 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 11.8580 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2359 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 97.3872 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 28.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 32.0000 |
| Waves / SM | 10.1587 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 23.6175 |
| Eligible Warps / Cycle | 0.5595 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 4.4593 |
| Stall: Long Scoreboard | 18.7722 |
| Stall: Short Scoreboard | 8.4284 |
| Stall: Math Pipe Throttle | 0.3012 |
| Stall: Wait | 0.7433 |
| Stall: No Instruction | 0.0420 |
| Stall: Not Selected | 1.3909 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 40960.0000 |
| Divergent Branch Targets (total) | 10240.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 279.3427 |
| FMUL Throughput (per cycle) | 402.2535 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 2.7999 |
| Warp Execution Efficiency | 31.9488 |
| L1 Bank Conflicts (total) | 19567.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.73e+09 |

**Kernel name:** `softmax_kernel`