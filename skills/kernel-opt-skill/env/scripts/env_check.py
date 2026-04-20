#!/usr/bin/env python3
"""Environment checks for the CUDA kernel optimization loop.

Usage:
    python env_check.py [-o report.md] [--gpu 0]
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def trim_output(text: str, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["..."])


def run_probe(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"command": shell_join(cmd), "returncode": 127, "stdout": "", "stderr": str(exc)}
    return {"command": shell_join(cmd), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def find_cuda_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    return roots


def find_ncu_roots() -> list[Path]:
    roots: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        nvidia_dir = Path(program_files) / "NVIDIA Corporation"
        if nvidia_dir.exists():
            roots.extend(sorted(nvidia_dir.glob("Nsight Compute*")))
    return roots


def resolve_executable(candidate: str, tool_name: str) -> str:
    candidate = candidate.strip().strip('"')
    direct = Path(candidate).expanduser()
    if direct.exists():
        return str(direct.resolve())
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    if any(sep in candidate for sep in ("\\", "/")):
        return ""
    extra_names = [candidate]
    if os.name == "nt" and not Path(candidate).suffix:
        extra_names.extend([f"{candidate}.exe", f"{candidate}.bat", f"{candidate}.cmd"])
    search_roots: list[Path] = []
    if tool_name == "nvcc":
        search_roots.extend(root / "bin" for root in find_cuda_roots())
    elif tool_name == "ncu":
        search_roots.extend(find_ncu_roots())
    for root in search_roots:
        for name in extra_names:
            probe = root / name
            if probe.exists():
                return str(probe.resolve())
    return ""


def probe_executable(candidate: str, tool_name: str, version_args: list[str]) -> dict[str, Any]:
    resolved = resolve_executable(candidate, tool_name)
    info: dict[str, Any] = {
        "requested": candidate,
        "resolved": resolved,
        "exists": bool(resolved),
        "version_command": "",
        "version_returncode": None,
        "version_output": "",
    }
    if not resolved:
        return info
    probe = run_probe([resolved, *version_args])
    output = (probe["stdout"] or probe["stderr"]).strip()
    info["version_command"] = probe["command"]
    info["version_returncode"] = probe["returncode"]
    info["version_output"] = trim_output(output)
    return info


def probe_nvidia_smi() -> dict[str, Any]:
    resolved = shutil.which("nvidia-smi")
    info: dict[str, Any] = {"exists": bool(resolved), "resolved": resolved or "", "query_command": "", "returncode": None, "query_output": "", "gpus": []}
    if not resolved:
        return info
    primary = run_probe([resolved, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    probe = primary
    if primary["returncode"] != 0 or not primary["stdout"].strip():
        fallback = run_probe([resolved, "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if fallback["returncode"] == 0 and fallback["stdout"].strip():
            probe = fallback
    info["query_command"] = probe["command"]
    info["returncode"] = probe["returncode"]
    info["query_output"] = trim_output((probe["stdout"] or probe["stderr"]).strip())
    if probe["returncode"] == 0:
        for line in probe["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                info["gpus"].append({"name": parts[0], "compute_capability": parts[1], "driver_version": parts[2]})
            elif len(parts) >= 2:
                info["gpus"].append({"name": parts[0], "compute_capability": "", "driver_version": parts[1]})
    return info


def probe_torch_cuda(gpu_index: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "importable": False, "version": "", "cuda_version": "", "cuda_available": False,
        "device_count": 0, "selected_gpu_index": gpu_index, "selected_gpu_name": "",
        "selected_gpu_compute_capability": "", "selected_sm": "", "error": "",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["error"] = str(exc)
        return info
    info["importable"] = True
    info["version"] = getattr(torch, "__version__", "")
    info["cuda_version"] = getattr(torch.version, "cuda", "") or ""
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            if 0 <= gpu_index < info["device_count"]:
                info["selected_gpu_name"] = torch.cuda.get_device_name(gpu_index)
                major, minor = torch.cuda.get_device_capability(gpu_index)
                info["selected_gpu_compute_capability"] = f"{major}.{minor}"
                info["selected_sm"] = f"sm_{major}{minor}"
    except Exception as exc:
        info["error"] = str(exc)
    return info


def probe_nsight_python() -> dict[str, Any]:
    info: dict[str, Any] = {"importable": False, "version": "", "error": ""}
    try:
        import nsight  # type: ignore
    except Exception as exc:
        info["error"] = str(exc)
        return info
    info["importable"] = True
    info["version"] = getattr(nsight, "__version__", "unknown")
    return info


def collect_env_check(gpu_index: int) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    requirements: list[dict[str, Any]] = []

    def add_req(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        requirements.append({"name": name, "ok": ok, "detail": detail, "required": required})
        if required and not ok:
            errors.append(f"{name}: {detail}")

    result: dict[str, Any] = {
        "checked_at": now_iso(),
        "ready": False,
        "python_executable": sys.executable,
        "python_version": sys.version.splitlines()[0],
        "selected_gpu_index": gpu_index,
        "env_vars": {
            "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
            "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            "CUDA_ROOT": os.environ.get("CUDA_ROOT", ""),
        },
        "requirements": requirements,
        "warnings": warnings,
        "errors": errors,
    }

    torch_info = probe_torch_cuda(gpu_index)
    result["torch"] = torch_info
    add_req("PyTorch import", torch_info["importable"],
            torch_info["version"] if torch_info["importable"] else (torch_info["error"] or "torch import failed"))
    add_req("CUDA runtime", torch_info["cuda_available"],
            f"torch CUDA {torch_info['cuda_version']}" if torch_info["cuda_available"] else (torch_info["error"] or "torch.cuda.is_available() returned false"))
    selected_gpu_ok = torch_info["cuda_available"] and 0 <= gpu_index < int(torch_info["device_count"])
    add_req(f"GPU index {gpu_index}", selected_gpu_ok,
            f"{torch_info['selected_gpu_name']} ({torch_info['selected_sm']})" if selected_gpu_ok else f"available device count: {torch_info['device_count']}")

    nvidia_smi_info = probe_nvidia_smi()
    result["nvidia_smi"] = nvidia_smi_info
    if not nvidia_smi_info["exists"]:
        warnings.append("nvidia-smi not found; GPU model falls back to PyTorch detection.")
    elif nvidia_smi_info.get("returncode") not in (None, 0):
        warnings.append("nvidia-smi is present but GPU query failed.")

    gpu_info: dict[str, Any] = {
        "name": torch_info.get("selected_gpu_name", ""),
        "compute_capability": torch_info.get("selected_gpu_compute_capability", ""),
        "sm": torch_info.get("selected_sm", ""),
        "driver_version": "",
        "source": "torch" if torch_info.get("selected_gpu_name") else "",
    }
    if nvidia_smi_info.get("gpus") and gpu_index < len(nvidia_smi_info["gpus"]):
        smi_gpu = nvidia_smi_info["gpus"][gpu_index]
        if smi_gpu.get("name"):
            gpu_info["name"] = smi_gpu["name"]
            gpu_info["source"] = "nvidia-smi"
        if smi_gpu.get("compute_capability"):
            gpu_info["compute_capability"] = smi_gpu["compute_capability"]
            if not gpu_info["sm"] and "." in smi_gpu["compute_capability"]:
                major, minor = smi_gpu["compute_capability"].split(".", 1)
                gpu_info["sm"] = f"sm_{major}{minor}"
        if smi_gpu.get("driver_version"):
            gpu_info["driver_version"] = smi_gpu["driver_version"]
    result["gpu"] = gpu_info

    nvcc_info = probe_executable("nvcc", "nvcc", ["--version"])
    result["nvcc"] = nvcc_info
    add_req("nvcc executable", nvcc_info["exists"], nvcc_info["resolved"] or "cannot resolve nvcc")
    if nvcc_info["exists"] and nvcc_info.get("version_returncode") not in (None, 0):
        warnings.append("nvcc exists but `--version` did not exit cleanly.")

    ncu_info = probe_executable("ncu", "ncu", ["--version"])
    result["ncu"] = ncu_info
    add_req("ncu executable", ncu_info["exists"], ncu_info["resolved"] or "cannot resolve ncu")
    if ncu_info["exists"] and ncu_info.get("version_returncode") not in (None, 0):
        warnings.append("ncu exists but `--version` did not exit cleanly.")

    nsight_info = probe_nsight_python()
    result["nsight_python"] = nsight_info
    add_req("nsight-python package", nsight_info["importable"],
            f"nsight {nsight_info['version']}" if nsight_info["importable"] else (nsight_info["error"] or "import nsight failed"))

    result["ready"] = not errors
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "#Environment Check",
        "",
        "## Status",
        f"- ready: {'yes' if result.get('ready') else 'no'}",
        f"- checked at: {result.get('checked_at', '')}",
        f"- python: {result.get('python_executable', '')}",
        f"- python version: {result.get('python_version', '')}",
        f"- selected gpu index: {result.get('selected_gpu_index')}",
        "",
        "## Requirements",
        "",
        "| Requirement | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for item in result.get("requirements", []):
        status = "ok" if item.get("ok") else "missing"
        detail = str(item.get("detail", "")).replace("\n", "<br>")
        lines.append(f"| {item.get('name')} | {status} | {detail} |")

    gpu = result.get("gpu") or {}
    torch_info = result.get("torch") or {}
    nvidia_smi = result.get("nvidia_smi") or {}
    nvcc = result.get("nvcc") or {}
    ncu = result.get("ncu") or {}
    nsight_py = result.get("nsight_python") or {}

    lines.extend([
        "",
        "## GPU",
        f"- model: {gpu.get('name') or 'unknown'}",
        f"- compute capability: {gpu.get('compute_capability') or 'unknown'}",
        f"- sm: {gpu.get('sm') or 'unknown'}",
        f"- driver version: {gpu.get('driver_version') or 'unknown'}",
        f"- torch: {torch_info.get('version') or 'not importable'}",
        f"- torch cuda: {torch_info.get('cuda_version') or 'unknown'}",
        f"- device count: {torch_info.get('device_count')}",
        f"- nvidia-smi: {nvidia_smi.get('resolved') or 'not found'}",
        "",
        "## Tools",
        f"- nvcc: {nvcc.get('resolved') or 'not found'}",
        f"- nvcc version: {nvcc.get('version_output') or 'n/a'}",
        f"- ncu: {ncu.get('resolved') or 'not found'}",
        f"- ncu version: {ncu.get('version_output') or 'n/a'}",
        f"- nsight-python: {nsight_py.get('version') or 'not importable'}",
        "",
        "## Environment variables",
        f"- CUDA_PATH: {result.get('env_vars', {}).get('CUDA_PATH') or '(unset)'}",
        f"- CUDA_HOME: {result.get('env_vars', {}).get('CUDA_HOME') or '(unset)'}",
        f"- CUDA_ROOT: {result.get('env_vars', {}).get('CUDA_ROOT') or '(unset)'}",
        "",
        "## Errors",
    ])
    if result.get("errors"):
        lines.extend(f"- {item}" for item in result["errors"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings"])
    if result.get("warnings"):
        lines.extend(f"- {item}" for item in result["warnings"])
    else:
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check environment readiness for CUDA kernel optimization")
    parser.add_argument("-o", "--out", default="", help="Write markdown report to this path (default: print to stdout)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")
    args = parser.parse_args()

    result = collect_env_check(args.gpu)
    md = render_markdown(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"Env check report written to: {out_path}")
    else:
        print(md)

    return 0 if result.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
