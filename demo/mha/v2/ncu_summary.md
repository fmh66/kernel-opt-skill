# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 512, 'num_heads': 8} |
| **Execution Time** | 1.8626 ms ± 0.0392 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 33.3886 |
| Memory Throughput (% of peak) | 0.7616 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.55e+09 |
| DRAM Read Bandwidth (bytes/s) | 3.75e+09 |
| DRAM Write Bandwidth (bytes/s) | 1.81e+09 |
| L1 Global Load Bandwidth (bytes/s) | 3.82e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.25e+09 |
| L2 Total Bandwidth (bytes/s) | 2.59e+12 |
| Global Load Efficiency (%) | 66.6775 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 32.2463 |
| L2 Hit Rate (%) | 99.9753 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.7602 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1560 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 64.8497 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 52.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 4480.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 15.6056 |
| Eligible Warps / Cycle | 0.2828 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.7275 |
| Stall: Long Scoreboard | 21.3904 |
| Stall: Short Scoreboard | 1.9960 |
| Stall: Math Pipe Throttle | 0.0422 |
| Stall: Wait | 2.0039 |
| Stall: No Instruction | 0.1229 |
| Stall: Not Selected | 0.8084 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 7.29e+06 |
| Divergent Branch Targets (total) | 24576.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 17.4295 |
| FMUL Throughput (per cycle) | 8.2107 |
| FFMA Throughput (per cycle) | 353.8998 |
| LSU Pipe Utilization (% of peak) | 8.3922 |
| Warp Execution Efficiency | 31.9068 |
| L1 Bank Conflicts (total) | 7.63e+07 |
| Shared Memory Bandwidth (bytes/s) | 2.01e+11 |

**Kernel name:** `mha_v2_kernel`