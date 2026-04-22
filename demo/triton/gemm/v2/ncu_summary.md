# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 10240, 'N': 10240, 'K': 10240} |
| **Execution Time** | 43.9178 ms ± 2.8745 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 80.6057 |
| Memory Throughput (% of peak) | 76.9882 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.61e+11 |
| DRAM Read Bandwidth (bytes/s) | 5.50e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.11e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.77e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.10e+10 |
| L2 Total Bandwidth (bytes/s) | 1.78e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 65.6829 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 1.3707 |
| Tensor Core Utilization (% of peak) | 79.0747 |
| IPC (instructions per cycle) | 0.1235 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 8.0981 |
| Theoretical Occupancy (%) | 8.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 6400.0000 |
| Registers / Thread | 241.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 65536.0000 |
| Waves / SM | 76.1905 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 12.3521 |
| Eligible Warps / Cycle | 0.1235 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.7452 |
| Stall: Long Scoreboard | 1.7188 |
| Stall: Short Scoreboard | 0.0033 |
| Stall: Math Pipe Throttle | 3.0118 |
| Stall: Wait | 2.7938 |
| Stall: No Instruction | 0.0068 |
| Stall: Not Selected | 0.0000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.71e+07 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 7.9494 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 30785.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.12e+12 |

**Kernel name:** `_gemm_kernel`