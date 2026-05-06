---
name: env
description: Environment readiness check and configuration for CUDA/Triton kernel optimization.
---

# env-skill

## Directory Structure

```
env/
├── ENV.md
└── scripts/
    ├── enc_config.py
    └── env_check.py
```

Environment check and configuration is essential preparation before kernel optimization. **If any required item fails, immediately stop all subsequent optimization.**

## Scripts

| Script | Responsibility |
|---|---|
| `scripts/env_check.py` | Detect and validate CUDA/PyTorch/Triton environment, output Markdown report |
| `scripts/enc_config.py` | GPU clock locking and similar configuration operations |

---

## scripts/env_check.py

### Usage

```bash
python scripts/env_check.py -o <output_dir>/env_check.md [--gpu 0]
```

Reads `<output_dir>/env_check.md` as the environment baseline for kernel optimization. All subsequent environment queries use this file.

The check validates the following as required items: `PyTorch`, `CUDA runtime`, `nvcc`, `ncu`, `nsight-python`, `triton`.

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `-o / --out` | Yes | — | Markdown report output path |
| `--gpu` | No | `0` | GPU device index |

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Environment ready — all required items passed |
| `1` | Environment not ready — one or more required items failed |
| `2` | Parameter error |

---

## scripts/enc_config.py

- Call before kernel optimization to lock the target GPU's SM clocks to maximum frequency, eliminating frequency jitter from performance data.
- If the setting fails, further optimization is also not allowed.

### Usage

```bash
python scripts/enc_config.py --gpu [0,1,2...]
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Failure |
