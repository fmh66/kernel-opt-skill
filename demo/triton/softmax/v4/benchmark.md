# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v4.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| Execution Time (ms) | 0.1496 | 0.2649 |
| Std dev (ms)        | 0.0025 | 0.0014 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 22.7021 | 22.8564 |
| Memory Throughput (% peak) | 92.7646 | 92.2517 |
| DRAM Bandwidth (bytes/s) | 6.76e+11 | 6.72e+11 |
| Achieved Occupancy (%) | 94.8334 | 94.8479 |
