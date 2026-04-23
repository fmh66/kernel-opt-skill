# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Execution Time** | 0.1632 ms ± 0.0014 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 11.0080 |
| Memory Throughput (% of peak) | 91.1739 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.64e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.37e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.27e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.37e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.37e+11 |
| L2 Total Bandwidth (bytes/s) | 6.76e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.1815 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.2898 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0839 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 97.7607 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 22.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 10.1587 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 8.4219 |
| Eligible Warps / Cycle | 0.1507 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 15.1351 |
| Stall: Long Scoreboard | 51.5883 |
| Stall: Short Scoreboard | 35.0740 |
| Stall: Math Pipe Throttle | 0.1103 |
| Stall: Wait | 1.5119 |
| Stall: No Instruction | 0.0652 |
| Stall: Not Selected | 0.7776 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 120.5502 |
| FMUL Throughput (per cycle) | 87.6729 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 2.9619 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 18872.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.56e+09 |

**Kernel name:** `softmax_kernel`