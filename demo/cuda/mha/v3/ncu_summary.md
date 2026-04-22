# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 512, 'num_heads': 8} |
| **Execution Time** | 1.9106 ms ± 0.0088 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 32.3864 |
| Memory Throughput (% of peak) | 0.7418 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.41e+09 |
| DRAM Read Bandwidth (bytes/s) | 3.67e+09 |
| DRAM Write Bandwidth (bytes/s) | 1.74e+09 |
| L1 Global Load Bandwidth (bytes/s) | 3.71e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.21e+09 |
| L2 Total Bandwidth (bytes/s) | 2.45e+12 |
| Global Load Efficiency (%) | 66.6775 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 34.3143 |
| L2 Hit Rate (%) | 99.7669 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.6740 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1492 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 81.0475 |
| Theoretical Occupancy (%) | 83.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 8832.0000 |
| Waves / SM | 4.8762 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 14.9262 |
| Eligible Warps / Cycle | 0.3463 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.8192 |
| Stall: Long Scoreboard | 18.1181 |
| Stall: Short Scoreboard | 4.4309 |
| Stall: Math Pipe Throttle | 0.0419 |
| Stall: Wait | 1.9363 |
| Stall: No Instruction | 0.1083 |
| Stall: Not Selected | 1.3194 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6.80e+06 |
| Divergent Branch Targets (total) | 16384.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 16.8937 |
| FMUL Throughput (per cycle) | 7.9584 |
| FFMA Throughput (per cycle) | 343.0211 |
| LSU Pipe Utilization (% of peak) | 8.2151 |
| Warp Execution Efficiency | 31.9267 |
| L1 Bank Conflicts (total) | 6.21e+07 |
| Shared Memory Bandwidth (bytes/s) | 1.95e+11 |

**Kernel name:** `mha_v3_kernel`