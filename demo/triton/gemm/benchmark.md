# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v3.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 10240, 'N': 10240, 'K': 10240} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| Execution Time (ms) | 42.7708 | 97.3622 |
| Std dev (ms)        | 0.7546 | 0.5796 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 76.1248 | 75.8458 |
| Memory Throughput (% peak) | 77.7079 | 80.0901 |
| DRAM Bandwidth (bytes/s) | 5.66e+11 | 5.84e+11 |
| Achieved Occupancy (%) | 16.6634 | 16.6635 |
