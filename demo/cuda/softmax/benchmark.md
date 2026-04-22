# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v3.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 1024} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| Execution Time (ms) | 0.1469 | 0.2721 |
| Std dev (ms)        | 0.0047 | 0.0277 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 32.1647 | 32.0478 |
| Memory Throughput (% peak) | 91.8881 | 92.5685 |
| DRAM Bandwidth (bytes/s) | 6.70e+11 | 6.75e+11 |
| Achieved Occupancy (%) | 94.2768 | 94.4619 |
