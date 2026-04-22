# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 4096, 'K': 4096, 'N': 4096} |
| **Execution Time** | 9.4263 ms ± 0.6828 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 72.5639 |
| Memory Throughput (% of peak) | 73.9261 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.39e+11 |
| DRAM Read Bandwidth (bytes/s) | 5.30e+11 |
| DRAM Write Bandwidth (bytes/s) | 8.87e+09 |
| L1 Global Load Bandwidth (bytes/s) | 1.09e+12 |
| L1 Global Store Bandwidth (bytes/s) | 8.73e+09 |
| L2 Total Bandwidth (bytes/s) | 1.12e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.2341 |
| L2 Hit Rate (%) | 52.7390 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 55.8108 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.7174 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 32.6892 |
| Theoretical Occupancy (%) | 33.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 122.0000 |
| Static Shared Memory (bytes) | 8256.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 12.1905 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 71.7376 |
| Eligible Warps / Cycle | 1.9155 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.4276 |
| Stall: Long Scoreboard | 0.8804 |
| Stall: Short Scoreboard | 0.1497 |
| Stall: Math Pipe Throttle | 0.1739 |
| Stall: Wait | 0.3043 |
| Stall: No Instruction | 0.0151 |
| Stall: Not Selected | 1.6693 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 8.63e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 5424.7639 |
| LSU Pipe Utilization (% of peak) | 18.3430 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 6.73e+07 |
| Shared Memory Bandwidth (bytes/s) | 3.95e+12 |

**Kernel name:** `tiled_gemm_v3`