# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 21.2147 |
| Memory Throughput (% of peak) | 90.4451 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.59e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.34e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.25e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.85e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.56e+11 |
| L2 Total Bandwidth (bytes/s) | 1.08e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 53.2579 |
| L2 Hit Rate (%) | 69.2950 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.0621 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2247 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 96.9361 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 32.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 22.4963 |
| Eligible Warps / Cycle | 0.3739 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 11.2550 |
| Stall: Long Scoreboard | 24.5005 |
| Stall: Short Scoreboard | 5.5610 |
| Stall: Math Pipe Throttle | 0.3095 |
| Stall: Wait | 2.0835 |
| Stall: No Instruction | 0.1301 |
| Stall: Not Selected | 0.6462 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.02e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 146.4204 |
| FMUL Throughput (per cycle) | 96.7181 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 5.5643 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 97584.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.60e+10 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 127808.0000 |

**Kernel name:** `softmax_vec4_fast`