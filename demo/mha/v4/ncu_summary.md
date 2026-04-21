# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 512, 'num_heads': 8} |
| **Execution Time** | 1.9863 ms ± 0.0160 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 87.8863 |
| Memory Throughput (% of peak) | 0.7184 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.24e+09 |
| DRAM Read Bandwidth (bytes/s) | 3.50e+09 |
| DRAM Write Bandwidth (bytes/s) | 1.74e+09 |
| L1 Global Load Bandwidth (bytes/s) | 2.36e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.15e+09 |
| L2 Total Bandwidth (bytes/s) | 2.30e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 2.4555 |
| L2 Hit Rate (%) | 99.8050 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 13.0727 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2915 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 20.5224 |
| Theoretical Occupancy (%) | 20.8333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 64.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 17280.0000 |
| Waves / SM | 19.5048 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 29.1519 |
| Eligible Warps / Cycle | 0.3775 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.5298 |
| Stall: Long Scoreboard | 2.0202 |
| Stall: Short Scoreboard | 0.8510 |
| Stall: Math Pipe Throttle | 0.0482 |
| Stall: Wait | 1.7673 |
| Stall: No Instruction | 0.1382 |
| Stall: Not Selected | 0.2949 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.16e+07 |
| Divergent Branch Targets (total) | 262144.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 39.2276 |
| FMUL Throughput (per cycle) | 9.7973 |
| FFMA Throughput (per cycle) | 336.3243 |
| LSU Pipe Utilization (% of peak) | 22.2012 |
| Warp Execution Efficiency | 31.5831 |
| L1 Bank Conflicts (total) | 7.63e+06 |
| Shared Memory Bandwidth (bytes/s) | 4.81e+12 |

**Kernel name:** `mha_v4_kernel`