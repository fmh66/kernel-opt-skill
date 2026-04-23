# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v0.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| Execution Time (ms) | 0.1479 | 0.2648 |
| Std dev (ms)        | 0.0018 | 0.0012 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 11.4614 | 11.4860 |
| Memory Throughput (% peak) | 92.6481 | 93.1703 |
| DRAM Bandwidth (bytes/s) | 6.75e+11 | 6.79e+11 |
| Achieved Occupancy (%) | 94.8596 | 94.8945 |
