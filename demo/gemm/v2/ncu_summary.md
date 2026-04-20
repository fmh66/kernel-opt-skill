# NCU Profile Summary

| | |
|---|---|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 4096, 'K': 4096, 'N': 4096} |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 65.0687 |
| Memory Throughput (% of peak) | 65.7344 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 4.79e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.71e+11 |
| DRAM Write Bandwidth (bytes/s) | 7.78e+09 |
| L1 Global Load Bandwidth (bytes/s) | 9.56e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.07e+10 |
| L2 Total Bandwidth (bytes/s) | 9.92e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 25.0000 |
| L1 Hit Rate (%) | 2.3039 |
| L2 Hit Rate (%) | 52.4611 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 47.8428 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.6280 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 49.2338 |
| Theoretical Occupancy (%) | 50.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 72.0000 |
| Static Shared Memory (bytes) | 8192.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 16.2540 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 62.7963 |
| Eligible Warps / Cycle | 2.0585 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.0721 |
| Stall: Long Scoreboard | 1.4271 |
| Stall: Short Scoreboard | 0.9921 |
| Stall: Math Pipe Throttle | 0.2187 |
| Stall: Wait | 0.4403 |
| Stall: No Instruction | 0.0178 |
| Stall: Not Selected | 2.2527 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.68e+07 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 4656.5426 |
| LSU Pipe Utilization (% of peak) | 16.4278 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3.51e+07 |
| Shared Memory Bandwidth (bytes/s) | 5.40e+12 |

## Kernel Runtime

| Metric | Value |
|--------|------:|
| Kernel Duration (ns) | 8.75e+06 |

**Kernel name:** `regblock_gemm`