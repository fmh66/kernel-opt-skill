# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 2.9202 |
| Memory Throughput (% of peak) | 32.2666 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.35e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.41e+11 |
| DRAM Write Bandwidth (bytes/s) | 9.41e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.13e+12 |
| L1 Global Store Bandwidth (bytes/s) | 7.52e+11 |
| L2 Total Bandwidth (bytes/s) | 9.00e+11 |
| Global Load Efficiency (%) | 12.5000 |
| Global Store Efficiency (%) | 12.5000 |
| L1 Hit Rate (%) | 91.3183 |
| L2 Hit Rate (%) | 83.9143 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.0306 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0391 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 16.6060 |
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
| Issue Slot Utilization (% of peak) | 3.9107 |
| Eligible Warps / Cycle | 0.0393 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 46.3153 |
| Stall: Short Scoreboard | 1.3226 |
| Stall: Math Pipe Throttle | 0.0005 |
| Stall: Wait | 1.8615 |
| Stall: No Instruction | 0.1035 |
| Stall: Not Selected | 0.0055 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 515520.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 18.4798 |
| FMUL Throughput (per cycle) | 6.1599 |
| FFMA Throughput (per cycle) | 55.4393 |
| LSU Pipe Utilization (% of peak) | 1.2437 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4.29e+07 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 891936.0000 |

**Kernel name:** `naive_softmax`