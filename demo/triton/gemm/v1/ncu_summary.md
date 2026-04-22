# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 10240, 'N': 10240, 'K': 10240} |
| **Execution Time** | 44.2621 ms ± 3.8800 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 82.5257 |
| Memory Throughput (% of peak) | 75.3868 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.50e+11 |
| DRAM Read Bandwidth (bytes/s) | 5.38e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.16e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.84e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.15e+10 |
| L2 Total Bandwidth (bytes/s) | 1.85e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 66.9545 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 1.4384 |
| Tensor Core Utilization (% of peak) | 82.4515 |
| IPC (instructions per cycle) | 0.1430 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 16.5003 |
| Theoretical Occupancy (%) | 16.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 6400.0000 |
| Registers / Thread | 168.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 65536.0000 |
| Waves / SM | 76.1905 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 14.2979 |
| Eligible Warps / Cycle | 0.2014 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.9772 |
| Stall: Long Scoreboard | 0.2634 |
| Stall: Short Scoreboard | 0.0043 |
| Stall: Math Pipe Throttle | 7.1889 |
| Stall: Wait | 2.2867 |
| Stall: No Instruction | 0.0128 |
| Stall: Not Selected | 0.4089 |

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
| LSU Pipe Utilization (% of peak) | 9.6974 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 29195.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.84e+12 |

**Kernel name:** `_gemm_kernel`