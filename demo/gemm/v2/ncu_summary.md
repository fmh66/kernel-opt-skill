# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 4096, 'K': 4096, 'N': 4096} |
| **Execution Time** | 11.6437 ms ± 0.2155 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 55.0002 |
| Memory Throughput (% of peak) | 53.5985 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 3.91e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.84e+11 |
| DRAM Write Bandwidth (bytes/s) | 6.87e+09 |
| L1 Global Load Bandwidth (bytes/s) | 8.35e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.71e+10 |
| L2 Total Bandwidth (bytes/s) | 8.72e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 25.0000 |
| L1 Hit Rate (%) | 2.6776 |
| L2 Hit Rate (%) | 56.7290 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 38.5756 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.5246 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 65.2521 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 56.0000 |
| Static Shared Memory (bytes) | 4096.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 12.1905 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 52.4576 |
| Eligible Warps / Cycle | 1.4301 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.9948 |
| Stall: Long Scoreboard | 3.0129 |
| Stall: Short Scoreboard | 2.9720 |
| Stall: Math Pipe Throttle | 0.1204 |
| Stall: Wait | 0.4113 |
| Stall: No Instruction | 0.0234 |
| Stall: Not Selected | 1.7259 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.69e+07 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 3936.6480 |
| LSU Pipe Utilization (% of peak) | 13.8343 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3.37e+08 |
| Shared Memory Bandwidth (bytes/s) | 4.76e+12 |

**Kernel name:** `tiled_gemm_v2`