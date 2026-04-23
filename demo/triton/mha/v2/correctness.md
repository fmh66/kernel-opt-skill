# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v2.py |
| **Backend** | triton |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Buf/ptr** | 4194304 elems |
| **Tolerance** | atol=0.001  rtol=0.001 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| output | tensor[float32] | ✓ | 6.5565e-07 | 2.9211e-08 | 4.5414e-06 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | -0.0284 | -0.0278 | 0.0612 | -0.0143 | -0.0186 | 0.1029 | -0.0668 | -0.0249 |
| ref    | -0.0284 | -0.0278 | 0.0612 | -0.0143 | -0.0186 | 0.1029 | -0.0668 | -0.0249 |
