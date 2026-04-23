# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 3.6763 ms ± 0.1210 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 35.8597 |
| Memory Throughput (% of peak) | 87.9108 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.41e+11 |
| DRAM Read Bandwidth (bytes/s) | 6.39e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.97e+09 |
| L1 Global Load Bandwidth (bytes/s) | 2.74e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.34e+09 |
| L2 Total Bandwidth (bytes/s) | 2.60e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 5.1352 |
| L2 Hit Rate (%) | 75.6092 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.6921 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2222 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 65.6618 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 16384.0000 |
| Registers / Thread | 63.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 1024.0000 |
| Waves / SM | 24.3810 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 22.2243 |
| Eligible Warps / Cycle | 0.2711 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.2045 |
| Stall: Long Scoreboard | 25.4776 |
| Stall: Short Scoreboard | 2.8865 |
| Stall: Math Pipe Throttle | 0.1257 |
| Stall: Wait | 1.5010 |
| Stall: No Instruction | 0.0110 |
| Stall: Not Selected | 0.2168 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 2.10e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 345.5869 |
| FMUL Throughput (per cycle) | 172.0613 |
| FFMA Throughput (per cycle) | 333.8721 |
| LSU Pipe Utilization (% of peak) | 8.7502 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 6.12e+07 |
| Shared Memory Bandwidth (bytes/s) | 2.95e+11 |

**Kernel name:** `mha_kernel_v1`