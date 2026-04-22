# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v0.py |
| **Backend** | triton |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'N': 1024, 'K': 1024} |
| **Buf/ptr** | 3145728 elems |
| **Tolerance** | atol=0.0001  rtol=0.001 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| C | tensor[float32] | ✓ | 1.9836e-04 | 1.3129e-05 | 3.5808e-06 | — |

## Value Previews

### C

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | 49.0491 | 24.6927 | 10.2291 | 40.2421 | -9.3308 | -28.9023 | 26.2671 | 23.5246 |
| ref    | 49.0491 | 24.6927 | 10.2291 | 40.2421 | -9.3307 | -28.9023 | 26.2671 | 23.5246 |
