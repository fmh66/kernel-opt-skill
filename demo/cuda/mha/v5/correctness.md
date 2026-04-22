# Correctness Check

| Field | Value |
|---|---|
| **Kernel** | v5.cu |
| **Reference** | ref.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 512, 'num_heads': 8} |
| **Buf/ptr** | 524288 elems |
| **Tolerance** | atol=0.001  rtol=0.01 |
| **Result** | **ALL PASS** |

## Output Tensors

| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |
|--------|------|:----:|---------:|----------:|---------:|------------|
| output | float* | ✓ | 6.8545e-07 | 2.5108e-08 | 4.8629e-06 | — |

## Value Previews

### output

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel | -0.0466 | 0.0011 | 0.0286 | 0.0680 | -0.1487 | -0.0315 | 0.0133 | 0.1010 |
| ref    | -0.0466 | 0.0011 | 0.0286 | 0.0680 | -0.1487 | -0.0315 | 0.0133 | 0.1010 |
