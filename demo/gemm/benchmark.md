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
| Execution Time (ms) | 6.7477 | 6.0756 |
| Std dev (ms)        | 0.0378 | 0.0635 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 31.7507 | 31.8485 |
| Memory Throughput (% peak) | 45.4636 | 46.3317 |
| DRAM Bandwidth (bytes/s) | 3.31e+11 | 3.38e+11 |
| Achieved Occupancy (%) | 32.6534 | 32.6834 |
