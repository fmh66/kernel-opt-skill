# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 4096, 'K': 4096, 'N': 4096} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 31.8430 |
| Memory Throughput (% of peak) | 44.7148 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 3.26e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.15e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.08e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.34e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.07e+10 |
| L2 Total Bandwidth (bytes/s) | 1.25e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 9.1930 |
| L2 Hit Rate (%) | 76.3587 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 8.5427 |
| Tensor Core Utilization (% of peak) | 13.6059 |
| IPC (instructions per cycle) | 0.1997 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 32.6554 |
| Theoretical Occupancy (%) | 33.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 118.0000 |
| Static Shared Memory (bytes) | 4096.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 12.1905 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 19.9710 |
| Eligible Warps / Cycle | 0.2697 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.1287 |
| Stall: Long Scoreboard | 8.2523 |
| Stall: Short Scoreboard | 2.5139 |
| Stall: Math Pipe Throttle | 0.3984 |
| Stall: Wait | 1.6110 |
| Stall: No Instruction | 0.0132 |
| Stall: Not Selected | 0.3445 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 4.26e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 8.0897 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 5.38e+08 |
| Shared Memory Bandwidth (bytes/s) | 2.73e+12 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 6.28e+06 |

**Kernel name:** `wmma_gemm`