#Environment Check

## Status
- ready: yes
- checked at: 2026-04-22T13:05:09
- python: /home/kernel-opt-skill/.venv/bin/python
- python version: 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]
- selected gpu index: 0

## Requirements

| Requirement | Status | Detail |
| --- | --- | --- |
| PyTorch import | ok | 2.11.0+cu126 |
| CUDA runtime | ok | torch CUDA 12.6 |
| GPU index 0 | ok | NVIDIA RTX A6000 (sm_86) |
| nvcc executable | ok | /usr/local/cuda-12.6/bin/nvcc |
| ncu executable | ok | /usr/local/cuda-12.6/bin/ncu |
| nsight-python package | ok | nsight 0.9.6 |
| triton package | ok | triton 3.6.0 |

## GPU
- model: NVIDIA RTX A6000
- compute capability: 8.6
- sm: sm_86
- driver version: 575.57.08
- torch: 2.11.0+cu126
- torch cuda: 12.6
- device count: 2
- nvidia-smi: /usr/bin/nvidia-smi

## Tools
- nvcc: /usr/local/cuda-12.6/bin/nvcc
- nvcc version: nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
- ncu: /usr/local/cuda-12.6/bin/ncu
- ncu version: NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2024 NVIDIA Corporation
Version 2024.3.2.0 (build 34861637) (public-release)
- nsight-python: 0.9.6
- triton: 3.6.0

## Environment variables
- CUDA_PATH: (unset)
- CUDA_HOME: (unset)
- CUDA_ROOT: (unset)

## Errors
- none

## Warnings
- none