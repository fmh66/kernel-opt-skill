# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 512, 'num_heads': 8} |
| **Execution Time** | 4.1528 ms ± 0.0143 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 17.7772 |
| Memory Throughput (% of peak) | 0.3220 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.35e+09 |
| DRAM Read Bandwidth (bytes/s) | 1.58e+09 |
| DRAM Write Bandwidth (bytes/s) | 7.66e+08 |
| L1 Global Load Bandwidth (bytes/s) | 4.84e+12 |
| L1 Global Store Bandwidth (bytes/s) | 5.26e+08 |
| L2 Total Bandwidth (bytes/s) | 2.07e+12 |
| Global Load Efficiency (%) | 22.2307 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 57.3837 |
| L2 Hit Rate (%) | 99.9496 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.1611 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0642 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 63.8939 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 4480.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 6.4234 |
| Eligible Warps / Cycle | 0.1859 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.0637 |
| Stall: Long Scoreboard | 6.7289 |
| Stall: Short Scoreboard | 8.4803 |
| Stall: Math Pipe Throttle | 0.0150 |
| Stall: Wait | 2.0541 |
| Stall: No Instruction | 0.0740 |
| Stall: Not Selected | 1.8949 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 5.01e+06 |
| Divergent Branch Targets (total) | 16384.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 3.6769 |
| FMUL Throughput (per cycle) | 3.3984 |
| FFMA Throughput (per cycle) | 146.4764 |
| LSU Pipe Utilization (% of peak) | 4.4856 |
| Warp Execution Efficiency | 31.9276 |
| L1 Bank Conflicts (total) | 3.68e+08 |
| Shared Memory Bandwidth (bytes/s) | 8.48e+10 |

**Kernel name:** `mha_v1_kernel`