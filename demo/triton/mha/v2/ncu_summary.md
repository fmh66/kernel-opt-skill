# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 0.7325 ms ± 0.0091 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 20.2068 |
| Memory Throughput (% of peak) | 12.4656 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 9.09e+10 |
| DRAM Read Bandwidth (bytes/s) | 8.26e+10 |
| DRAM Write Bandwidth (bytes/s) | 8.28e+09 |
| L1 Global Load Bandwidth (bytes/s) | 3.92e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.03e+09 |
| L2 Total Bandwidth (bytes/s) | 3.99e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 79.3136 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 18.4765 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2315 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 8.3340 |
| Theoretical Occupancy (%) | 8.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 512.0000 |
| Registers / Thread | 255.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 81920.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 23.1501 |
| Eligible Warps / Cycle | 0.2315 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.1033 |
| Stall: Long Scoreboard | 0.0870 |
| Stall: Short Scoreboard | 2.6376 |
| Stall: Math Pipe Throttle | 0.0014 |
| Stall: Wait | 0.0858 |
| Stall: No Instruction | 0.0061 |
| Stall: Not Selected | 0.0000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 65536.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 37.6023 |
| FMUL Throughput (per cycle) | 41.9109 |
| FFMA Throughput (per cycle) | 1607.4974 |
| LSU Pipe Utilization (% of peak) | 5.3422 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 5.87e+07 |
| Shared Memory Bandwidth (bytes/s) | 2.19e+12 |

**Kernel name:** `mha_kernel_v2`