---
name: env-skill
description: Environment readiness check and configuration for CUDA kernel optimization.
---

# env

## 目录结构

```
env/
├── SKILL.md
└── scripts/
    ├── enc_config.py
    └── env_check.py
```

环境检查和配置是 kernel 优化前的重要准备工作。**若任一 required 项失败，立即停止后续优化。**

## scripts

| 脚本 | 职责 |
|---|---|
| `scripts/env_check.py` | 探测并验证 CUDA 环境，输出 Markdown 报告 |
| `scripts/enc_config.py` | GPU 时钟锁定等配置操作 |

---

## scripts/env_check.py

### 用法

```bash
python scripts/env_check.py -o <output_dir>/env_check.md [--gpu 0]
```

读取`<output_dir>/env_check.md` 文件获取 kernel 优化的环境基础，后续的所有环境信息在此查询。

### 参数

| 参数 | 必选 | 默认 | 说明 |
|---|---|---|---|
| `-o / --out` | 是 | — | Markdown 报告写出路径 |
| `--gpu` | 否 | `0` | GPU 设备编号 |

### 退出码

| 码 | 含义 |
|---|---|
| `0` | 环境就绪，所有 required 项通过 |
| `1` | 环境未就绪，存在 required 项失败 |
| `2` | 参数错误 |

---

## scripts/enc_config.py

- 在 kernel 优化前调用，将目标 GPU 的 SM 时钟锁定到最大频率，消除频率抖动对性能数据的干扰。
- 若设置失败，同样不允许继续后续优化

### 用法

```bash
python scripts/enc_config.py --gpu [0,1,2...]
```

### 退出码

| 码 | 含义 |
|---|---|
| `0` | 设置成功 |
| `1` | 设置失败 |
