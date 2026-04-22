# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v1.py |
| **Backend** | triton |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'N': 1024, 'K': 1024} |
| **Buf/ptr** | 3145728 elems |
| **Tolerance** | atol=0.5  rtol=0.05 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| C | tensor[float32] | ✓ | 1.2822e-01 | 1.9730e-02 | 6.1906e-03 | — |

## Value Previews

### C

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | 49.0160 | 24.6777 | 10.2323 | 40.2058 | -9.3309 | -28.8865 | 26.2627 | 23.4996 |
| ref    | 49.0491 | 24.6927 | 10.2291 | 40.2421 | -9.3307 | -28.9023 | 26.2671 | 23.5246 |
