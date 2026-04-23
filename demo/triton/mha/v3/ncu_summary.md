# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 0.2159 ms ± 0.0017 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 32.1181 |
| Memory Throughput (% of peak) | 30.0628 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.19e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.88e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.10e+10 |
| L1 Global Load Bandwidth (bytes/s) | 8.50e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.58e+10 |
| L2 Total Bandwidth (bytes/s) | 8.78e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 78.6400 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.7601 |
| Tensor Core Utilization (% of peak) | 41.3501 |
| IPC (instructions per cycle) | 0.2023 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 8.3536 |
| Theoretical Occupancy (%) | 8.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 144.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 81920.0000 |
| Waves / SM | 3.0476 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 20.2414 |
| Eligible Warps / Cycle | 0.2024 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.1317 |
| Stall: Long Scoreboard | 0.3845 |
| Stall: Short Scoreboard | 1.0812 |
| Stall: Math Pipe Throttle | 0.4108 |
| Stall: Wait | 1.6318 |
| Stall: No Instruction | 0.0117 |
| Stall: Not Selected | 0.0000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 32768.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 114.6617 |
| FMUL Throughput (per cycle) | 169.0417 |
| FFMA Throughput (per cycle) | 3.3724 |
| LSU Pipe Utilization (% of peak) | 9.3877 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 6014.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.71e+12 |

**Kernel name:** `mha_kernel_v3`