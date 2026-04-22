# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 10240, 'N': 10240, 'K': 10240} |
| **Execution Time** | 42.2495 ms ± 0.5008 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 75.3213 |
| Memory Throughput (% of peak) | 73.2274 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.34e+11 |
| DRAM Read Bandwidth (bytes/s) | 5.23e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.04e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.67e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.04e+10 |
| L2 Total Bandwidth (bytes/s) | 1.68e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 81.9968 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.4522 |
| Tensor Core Utilization (% of peak) | 76.9149 |
| IPC (instructions per cycle) | 0.1523 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 16.8528 |
| Theoretical Occupancy (%) | 16.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 6400.0000 |
| Registers / Thread | 128.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 65536.0000 |
| Waves / SM | 76.1905 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 15.2318 |
| Eligible Warps / Cycle | 0.1952 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.6710 |
| Stall: Long Scoreboard | 0.4660 |
| Stall: Short Scoreboard | 2.1658 |
| Stall: Math Pipe Throttle | 5.1989 |
| Stall: Wait | 2.8719 |
| Stall: No Instruction | 0.0113 |
| Stall: Not Selected | 0.2817 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.64e+07 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 9.1950 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 1689.0000 |
| Shared Memory Bandwidth (bytes/s) | 6.70e+12 |

**Kernel name:** `_gemm_kernel`