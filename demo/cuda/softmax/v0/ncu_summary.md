# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.8992 ms ± 0.0040 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 2.9888 |
| Memory Throughput (% of peak) | 33.1605 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.42e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.45e+11 |
| DRAM Write Bandwidth (bytes/s) | 9.66e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.16e+12 |
| L1 Global Store Bandwidth (bytes/s) | 7.73e+11 |
| L2 Total Bandwidth (bytes/s) | 9.19e+11 |
| Global Load Efficiency (%) | 12.5000 |
| Global Store Efficiency (%) | 12.5000 |
| L1 Hit Rate (%) | 91.8341 |
| L2 Hit Rate (%) | 84.2493 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.0335 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0392 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 16.5756 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 40.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 0.0794 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 3.9164 |
| Eligible Warps / Cycle | 0.0394 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 46.1785 |
| Stall: Short Scoreboard | 1.3225 |
| Stall: Math Pipe Throttle | 0.0005 |
| Stall: Wait | 1.8615 |
| Stall: No Instruction | 0.1034 |
| Stall: Not Selected | 0.0055 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 515520.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 18.9070 |
| FMUL Throughput (per cycle) | 6.3023 |
| FFMA Throughput (per cycle) | 56.7209 |
| LSU Pipe Utilization (% of peak) | 1.2455 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4.29e+07 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `naive_softmax`