# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v3.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| Execution Time (ms) | 0.2183 | 0.8989 |
| Std dev (ms)        | 0.0081 | 0.0220 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 32.2295 | 32.1780 |
| Memory Throughput (% peak) | 29.9524 | 29.7742 |
| DRAM Bandwidth (bytes/s) | 2.18e+11 | 2.17e+11 |
| Achieved Occupancy (%) | 8.3299 | 8.3299 |
