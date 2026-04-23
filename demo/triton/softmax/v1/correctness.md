# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v1.py |
| **Backend** | triton |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Buf/ptr** | 2097152 elems |
| **Tolerance** | atol=0.0001  rtol=0.001 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| output | tensor[float32] | ✓ | 7.4506e-09 | 7.5583e-11 | 9.1768e-08 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | 0.0007 | 0.0048 | 0.0005 | 0.0013 | 0.0001 | 0.0011 | 0.0003 | 0.0002 |
| ref    | 0.0007 | 0.0048 | 0.0005 | 0.0013 | 0.0001 | 0.0011 | 0.0003 | 0.0002 |
