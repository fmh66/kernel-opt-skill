# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v3.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 4096, 'K': 4096, 'N': 4096} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| Execution Time (ms) | 9.4131 | 6.1841 |
| Std dev (ms)        | 0.5836 | 0.0319 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 72.5530 | 72.4947 |
| Memory Throughput (% peak) | 73.9836 | 74.2628 |
| DRAM Bandwidth (bytes/s) | 5.39e+11 | 5.41e+11 |
| Achieved Occupancy (%) | 32.7011 | 32.6999 |
