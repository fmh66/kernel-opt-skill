# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v1.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| Execution Time (ms) | 0.1465 | 0.2657 |
| Std dev (ms)        | 0.0041 | 0.0034 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 33.2930 | 32.8008 |
| Memory Throughput (% peak) | 91.6127 | 91.8213 |
| DRAM Bandwidth (bytes/s) | 6.68e+11 | 6.69e+11 |
| Achieved Occupancy (%) | 93.0198 | 93.1522 |
