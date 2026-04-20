# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 31.0850 |
| Memory Throughput (% of peak) | 90.0792 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.57e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.45e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.11e+11 |
| L1 Global Load Bandwidth (bytes/s) | 6.38e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.19e+11 |
| L2 Total Bandwidth (bytes/s) | 7.55e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 21.0841 |
| L2 Hit Rate (%) | 54.1189 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 19.8122 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.3295 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 85.4341 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 32.0000 |
| Static Shared Memory (bytes) | 64.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 32.9695 |
| Eligible Warps / Cycle | 0.6156 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.8698 |
| Stall: Long Scoreboard | 20.2763 |
| Stall: Short Scoreboard | 0.9872 |
| Stall: Math Pipe Throttle | 0.1449 |
| Stall: Wait | 0.8898 |
| Stall: No Instruction | 0.0767 |
| Stall: Not Selected | 0.9253 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 348160.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 486.5383 |
| FMUL Throughput (per cycle) | 354.4405 |
| FFMA Throughput (per cycle) | 1073.7848 |
| LSU Pipe Utilization (% of peak) | 4.0769 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 39598.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.56e+10 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 131424.0000 |

**Kernel name:** `softmax_online_vec4`