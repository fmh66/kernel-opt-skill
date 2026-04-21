# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 512, 'num_heads': 8} |
| **Execution Time** | 15.4461 ms ± 0.1770 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 86.9901 |
| Memory Throughput (% of peak) | 0.0852 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.21e+08 |
| DRAM Read Bandwidth (bytes/s) | 4.07e+08 |
| DRAM Write Bandwidth (bytes/s) | 2.14e+08 |
| L1 Global Load Bandwidth (bytes/s) | 2.29e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.32e+08 |
| L2 Total Bandwidth (bytes/s) | 2.54e+11 |
| Global Load Efficiency (%) | 17.6471 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 88.9549 |
| L2 Hit Rate (%) | 99.7348 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.2507 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2576 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 62.6796 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 48.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 4096.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 25.7557 |
| Eligible Warps / Cycle | 0.8814 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 14.5097 |
| Stall: Long Scoreboard | 4.8549 |
| Stall: Short Scoreboard | 0.2012 |
| Stall: Math Pipe Throttle | 0.0344 |
| Stall: Wait | 1.8228 |
| Stall: No Instruction | 0.1006 |
| Stall: Not Selected | 2.4209 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.15e+08 |
| Divergent Branch Targets (total) | 8192.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.8252 |
| FMUL Throughput (per cycle) | 0.2756 |
| FFMA Throughput (per cycle) | 39.0615 |
| LSU Pipe Utilization (% of peak) | 21.9328 |
| Warp Execution Efficiency | 1.7172 |
| L1 Bank Conflicts (total) | 76208.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.70e+10 |

**Kernel name:** `multi_head_attention_kernel`