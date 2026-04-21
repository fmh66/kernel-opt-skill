# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v5.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 512, 'num_heads': 8} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| Execution Time (ms) | 1.4753 | 0.5153 |
| Std dev (ms)        | 0.0158 | 0.0081 |

## Hardware Metrics (nsight-python)

| Metric | Solution | Reference |
|--------|----------:|----------:|
| SM Throughput (% peak) | 96.0926 | 96.1141 |
| Memory Throughput (% peak) | 0.9352 | 0.9369 |
| DRAM Bandwidth (bytes/s) | 6.82e+09 | 6.83e+09 |
| Achieved Occupancy (%) | 40.3317 | 40.3347 |
