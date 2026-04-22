# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 10240, 'N': 10240, 'K': 10240} |
| **Execution Time** | 129.7360 ms ± 2.3110 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 73.9500 |
| Memory Throughput (% of peak) | 82.7978 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.04e+11 |
| DRAM Read Bandwidth (bytes/s) | 6.00e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.28e+09 |
| L1 Global Load Bandwidth (bytes/s) | 1.04e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.26e+09 |
| L2 Total Bandwidth (bytes/s) | 1.05e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 47.7430 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 63.6114 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.7513 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 25.3086 |
| Theoretical Occupancy (%) | 25.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 25600.0000 |
| Registers / Thread | 96.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 32768.0000 |
| Waves / SM | 101.5873 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 75.1279 |
| Eligible Warps / Cycle | 1.7122 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.2296 |
| Stall: Long Scoreboard | 0.0221 |
| Stall: Short Scoreboard | 0.7322 |
| Stall: Math Pipe Throttle | 0.0372 |
| Stall: Wait | 0.0810 |
| Stall: No Instruction | 0.0195 |
| Stall: Not Selected | 1.2790 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6.55e+07 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 6679.0623 |
| LSU Pipe Utilization (% of peak) | 13.7006 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2.77e+06 |
| Shared Memory Bandwidth (bytes/s) | 2.79e+12 |

**Kernel name:** `_gemm_kernel`