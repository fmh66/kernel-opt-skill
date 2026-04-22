# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 512, 'num_heads': 8} |
| **Execution Time** | 1.5139 ms ± 0.0152 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 96.0993 |
| Memory Throughput (% of peak) | 0.9515 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.94e+09 |
| DRAM Read Bandwidth (bytes/s) | 4.58e+09 |
| DRAM Write Bandwidth (bytes/s) | 2.35e+09 |
| L1 Global Load Bandwidth (bytes/s) | 1.56e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.52e+09 |
| L2 Total Bandwidth (bytes/s) | 1.56e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.1604 |
| L2 Hit Rate (%) | 99.7063 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 10.3859 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.3066 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 40.3311 |
| Theoretical Occupancy (%) | 41.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 17792.0000 |
| Waves / SM | 9.7524 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 30.6641 |
| Eligible Warps / Cycle | 0.6543 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.3740 |
| Stall: Long Scoreboard | 1.5575 |
| Stall: Short Scoreboard | 1.8414 |
| Stall: Math Pipe Throttle | 0.0561 |
| Stall: Wait | 1.7229 |
| Stall: No Instruction | 0.1743 |
| Stall: Not Selected | 1.1334 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 9.98e+06 |
| Divergent Branch Targets (total) | 262144.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 51.2967 |
| FMUL Throughput (per cycle) | 12.8117 |
| FFMA Throughput (per cycle) | 439.8000 |
| LSU Pipe Utilization (% of peak) | 24.1997 |
| Warp Execution Efficiency | 31.4529 |
| L1 Bank Conflicts (total) | 3.67e+06 |
| Shared Memory Bandwidth (bytes/s) | 4.81e+12 |

**Kernel name:** `mha_v5_kernel`