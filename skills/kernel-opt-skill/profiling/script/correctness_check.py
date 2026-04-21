#!/usr/bin/env python3
"""CUDA kernel correctness checker.

Compiles a .cu kernel via nvcc, runs it against a Python reference
implementation, and reports pass/fail with detailed error diagnostics.

The kernel must expose `extern "C" void solve(...)`.
The reference .py must define `reference(**kwargs)` and may set
module-level atol/rtol overrides.

Usage:
    python correctness_check.py kernel.cu --ref=ref.py [--M=1024 --N=1024 ...]
"""

import re
import os
import sys
import copy
import subprocess
import ctypes
import argparse
import importlib.util
import torch

SUPPORTED_TYPES = {
    "float*": ("float*", ctypes.c_void_p),
    "double*": ("double*", ctypes.c_void_p),
    "unsigned char*": ("unsigned char*", ctypes.c_void_p),
    "unsigned short*": ("unsigned short*", ctypes.c_void_p),
    "unsigned int*": ("unsigned int*", ctypes.c_void_p),
    "char*": ("char*", ctypes.c_void_p),
    "short*": ("short*", ctypes.c_void_p),
    "long*": ("long*", ctypes.c_void_p),
    "int*": ("int*", ctypes.c_void_p),
    "int": ("int", ctypes.c_int),
    "long": ("long", ctypes.c_long),
    "size_t": ("size_t", ctypes.c_size_t),
    "unsigned int": ("unsigned int", ctypes.c_uint),
    "unsigned short": ("unsigned short", ctypes.c_ushort),
    "unsigned char": ("unsigned char", ctypes.c_ubyte),
    "char": ("char", ctypes.c_char),
    "short": ("short", ctypes.c_short),
}

DTYPE_MAP = {
    "float*": torch.float32,
    "double*": torch.float64,
    "int*": torch.int32,
    "long*": torch.int64,
    "short*": torch.int16,
    "char*": torch.int8,
    "unsigned char*": torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*": getattr(torch, "uint32", torch.int32),
}

INT_TYPES = {"int", "long", "size_t", "unsigned int"}


def parse_solve_signature(cu_file: str):
    """Extract parameter list from `extern "C" void solve(...)` in a .cu file."""
    with open(cu_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f'Cannot find \'extern "C" void solve(...)\' in {cu_file}')

    raw = match.group(1)
    raw = re.sub(r"/\*.*?\*/", "", raw)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())

    params = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        is_const = "const" in token
        token_clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
        matched = False
        for key in sorted(SUPPORTED_TYPES.keys(), key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            m = re.match(rf"({base})\s+(\w+)", token_clean)
            if m:
                params.append((key, m.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token.strip()}'")

    return params


def detect_arch(device_index: int | None = None) -> str:
    if torch.cuda.is_available():
        if device_index is None:
            device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        return f"sm_{major}{minor}"
    return "sm_80"


_STRIP_INCLUDES = re.compile(r'^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$', re.MULTILINE)


def _preprocess_cu(cu_file: str) -> str:
    """Strip clang-specific includes that break nvcc."""
    with open(cu_file, "r", encoding="utf-8") as f:
        src = f.read()
    cleaned = _STRIP_INCLUDES.sub("", src)
    if cleaned == src:
        return cu_file
    tmp = cu_file + ".nvcc_clean.cu"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return tmp


def compile_cu(cu_file: str, output_so: str, arch: str, nvcc_bin: str):
    clean_file = _preprocess_cu(cu_file)
    cmd = [nvcc_bin]
    if os.name != "nt":
        cmd.extend(["-Xcompiler", "-fPIC"])
    else:
        cmd.extend([
            "-allow-unsupported-compiler",
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        ])

    cmd.extend(["-shared", "-std=c++17", f"-arch={arch}", "-O3", "-o", output_so, clean_file])

    # strip NCU injection env vars that intercept subprocess execution
    _NCU_PREFIXES = ("NV_NSIGHT_", "NV_CUDA_", "NV_TPS_", "NV_COMPUTE_PROFILER_")
    _NCU_KEYS = ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH",
                 "CUPTI_INJECTION64_PATH", "DYLD_INSERT_LIBRARIES")
    clean_env = {
        k: v for k, v in os.environ.items()
        if k not in _NCU_KEYS and not any(k.startswith(p) for p in _NCU_PREFIXES)
    }

    print(f"[compile] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=clean_env,
        )
    except OSError as exc:
        if clean_file != cu_file and os.path.exists(clean_file):
            os.remove(clean_file)
        print(f"Compilation failed:\n{exc}", file=sys.stderr)
        sys.exit(1)
    if clean_file != cu_file and os.path.exists(clean_file):
        os.remove(clean_file)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[compile] -> {output_so}")


def load_reference(ref_file: str):
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference file not found: {ref_file}")
    spec = importlib.util.spec_from_file_location("_ref_module", ref_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {ref_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reference"):
        raise AttributeError(f"'{ref_file}' must define a `reference(**kwargs)` function.")
    return mod


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return copy.deepcopy(value)


def _determine_ptr_elems(int_values: list, ptr_size_override: int) -> int:
    if ptr_size_override > 0:
        ptr_elems = ptr_size_override
    elif len(int_values) == 0:
        ptr_elems = 1024 * 1024
    elif len(int_values) == 1:
        ptr_elems = int_values[0]
    else:
        sv = sorted(int_values, reverse=True)
        ptr_elems = sv[0] * sv[1]
    return min(ptr_elems, 256 * 1024 * 1024)  # cap at 256M elements (~1 GB for float32)


def _fmt_vals(vals, width=10):
    return "[" + ", ".join(f"{v:>{width}.4f}" for v in vals) + "]"


def _color(text: str, ok: bool) -> str:
    """ANSI color: green for pass, red for fail (only when stdout is a tty)."""
    if not sys.stdout.isatty():
        return text
    code = "\033[92m" if ok else "\033[91m"
    return f"{code}{text}\033[0m"


def _write_md_out(output_dir: str, content: str):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "correctness.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[ncu_profile] correctness -> {path}")



def _setup_cuda(solution_file, dim_values, ptr_size_override, arch, seed=None):
    params = parse_solve_signature(solution_file)
    sig_str = ", ".join(f"{'const ' if c else ''}{t} {n}" for t, n, c in params)
    print(f"[signature] solve({sig_str})\n")

    lib_ext = ".dll" if os.name == "nt" else ".so"
    so_file = os.path.abspath(os.path.splitext(solution_file)[0] + lib_ext)
    if not os.path.exists(so_file):
        print(f"[error] .so not found: {so_file}", file=sys.stderr)
        print(f"[error] Compile first: nvcc -shared -std=c++17 -arch=<arch> -O3 -Xcompiler -fPIC -o {so_file} {solution_file}", file=sys.stderr)
        sys.exit(1)
    print(f"[load] {so_file}")
    lib = ctypes.CDLL(so_file)

    for ptype, pname, _ in params:
        if ptype in INT_TYPES and pname not in dim_values:
            raise ValueError(f"Missing dimension: --{pname}=<value>  (required by kernel signature)")

    int_vals = [dim_values[pname] for ptype, pname, _ in params if ptype in INT_TYPES]
    ptr_elems = _determine_ptr_elems(int_vals, ptr_size_override)

    if seed is not None:
        torch.manual_seed(seed)

    tensor_inputs = {}
    reference_inputs = {}
    output_specs = []
    kernel_call_args = []
    argtypes = []

    print("[buffers]")
    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            if dtype.is_floating_point:
                tensor = torch.randn(ptr_elems, device="cuda", dtype=dtype)
            else:
                tensor = torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_()
            tensor_inputs[pname] = tensor
            reference_inputs[pname] = tensor
            if not is_const:
                output_specs.append((pname, ptype))
            kernel_call_args.append(ctypes.c_void_p(tensor.data_ptr()))
            argtypes.append(ctypes.c_void_p)
            role = "input" if is_const else "output"
            eb = tensor.element_size()
            print(
                f"  {pname:>10s} : {ptype:<16s} [{role:>6s}] "
                f"{ptr_elems} elems  ({ptr_elems * eb / 1024 / 1024:.1f} MB)"
            )
        elif ptype in SUPPORTED_TYPES:
            _, ctype = SUPPORTED_TYPES[ptype]
            val = dim_values[pname]
            reference_inputs[pname] = val
            kernel_call_args.append(ctype(val))
            argtypes.append(ctype)
            print(f"  {pname:>10s} : {ptype:<16s} = {val}")

    lib.solve.restype = None
    lib.solve.argtypes = argtypes

    total_ptr_bytes = sum(t.nelement() * t.element_size() for t in tensor_inputs.values())

    return {
        "backend": "cuda",
        "signature": [
            {"type": ptype, "name": pname, "is_const": is_const}
            for ptype, pname, is_const in params
        ],
        "callable": lambda: lib.solve(*kernel_call_args),
        "tensor_inputs": tensor_inputs,
        "reference_inputs": reference_inputs,
        "output_specs": output_specs,
        "ptr_elems": ptr_elems,
        "total_ptr_bytes": total_ptr_bytes,
    }



def _validate_outputs(kernel_tensors, ref_tensors, output_params, atol, rtol):
    """Compare kernel and reference output tensors. Returns (all_pass, md_rows)."""
    PREVIEW = 8
    print(f"\n[validate] {len(output_params)} output tensor(s)\n")

    all_pass = True
    md_rows = []
    for pname, ptype in output_params:
        kt = kernel_tensors[pname].float()
        rt = ref_tensors[pname].float()

        match = torch.allclose(kt, rt, atol=atol, rtol=rtol)
        if not match:
            all_pass = False

        max_diff = (kt - rt).abs().max().item()
        mean_diff = (kt - rt).abs().mean().item()
        rel_err = ((kt - rt).abs() / rt.abs().clamp(min=1e-8)).mean().item()

        status_str = _color("PASS" if match else "FAIL", match)
        print(f"  [{status_str}]  {pname}  ({ptype})")
        print(f"         max |delta|   = {max_diff:.6e}")
        print(f"         mean |delta|  = {mean_diff:.6e}")
        print(f"         mean rel  = {rel_err:.6e}")

        mismatch_info = ""
        if not match:
            diff_mask = ~torch.isclose(kt, rt, atol=atol, rtol=rtol)
            bad_idx = diff_mask.nonzero(as_tuple=True)[0]
            n_bad = bad_idx.numel()
            print(f"         mismatches: {n_bad} / {kt.numel()}")
            if n_bad > 0:
                idx = bad_idx[0].item()
                print(f"         first bad   @ idx={idx}:  kernel={kt[idx].item():.6f}  ref={rt[idx].item():.6f}")
                mismatch_info = f"{n_bad}/{kt.numel()} (first @ {idx}: kernel={kt[idx].item():.6f} ref={rt[idx].item():.6f})"
            else:
                mismatch_info = f"{n_bad}/{kt.numel()}"

        k_preview = kernel_tensors[pname][:PREVIEW].float().cpu().tolist()
        r_preview = ref_tensors[pname][:PREVIEW].float().cpu().tolist()
        print(f"         kernel[:{PREVIEW}] = {_fmt_vals(k_preview)}")
        print(f"         ref   [:{PREVIEW}] = {_fmt_vals(r_preview)}")
        print()

        md_rows.append({
            "name": pname, "type": ptype, "pass": match,
            "max_diff": max_diff, "mean_diff": mean_diff, "rel_err": rel_err,
            "mismatch_info": mismatch_info,
            "kernel_preview": k_preview, "ref_preview": r_preview,
        })

    return all_pass, md_rows



def _build_md(solution_file, ref_file, gpu_name, arch, dim_values, ptr_elems,
              atol, rtol, validation_passed, md_rows):
    result_str = "ALL PASS" if validation_passed else "FAILED"
    lines = [
        "# Correctness Check",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| **Kernel** | {os.path.basename(solution_file)} |",
        f"| **Reference** | {os.path.basename(ref_file)} |",
        f"| **GPU** | {gpu_name} |",
        f"| **Arch** | {arch} |",
        f"| **Dims** | {dim_values} |",
        f"| **Buf/ptr** | {ptr_elems} elems |",
        f"| **Tolerance** | atol={atol}  rtol={rtol} |",
        f"| **Result** | **{result_str}** |",
        "",
        "## Output Tensors",
        "",
        "| Tensor | Type | Pass | Max |Δ| | Mean |Δ| | Mean Rel | Mismatches |",
        "|--------|------|:----:|---------:|----------:|---------:|------------|",
    ]
    for r in md_rows:
        pass_str = "✓" if r["pass"] else "✗"
        lines.append(
            f"| {r['name']} | {r['type']} | {pass_str} "
            f"| {r['max_diff']:.4e} | {r['mean_diff']:.4e} | {r['rel_err']:.4e} "
            f"| {r['mismatch_info'] or '—'} |"
        )
    lines.append("")
    PREVIEW = len(md_rows[0]["kernel_preview"]) if md_rows else 0
    if md_rows and PREVIEW:
        lines.append("## Value Previews")
        lines.append("")
        for r in md_rows:
            lines.append(f"### {r['name']}")
            lines.append("")
            lines.append(f"| | {' | '.join(str(i) for i in range(PREVIEW))} |")
            lines.append(f"|---|{'|'.join(['---:'] * PREVIEW)}|")
            lines.append(f"| kernel | {' | '.join(f'{v:.4f}' for v in r['kernel_preview'])} |")
            lines.append(f"| ref    | {' | '.join(f'{v:.4f}' for v in r['ref_preview'])} |")
            lines.append("")
    return "\n".join(lines)


def run(solution_file, ref_file, dim_values, ptr_size_override, arch, atol, rtol, seed, output_dir=""):
    if not ref_file:
        print("[error] --ref is required for correctness checking.", file=sys.stderr)
        sys.exit(1)

    ref_mod = load_reference(ref_file)
    ref_fn = ref_mod.reference
    _atol = float(getattr(ref_mod, "atol", atol))
    _rtol = float(getattr(ref_mod, "rtol", rtol))
    print(f"[reference] {ref_file}  (atol={_atol}, rtol={_rtol})\n")

    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    state = _setup_cuda(solution_file, dim_values, ptr_size_override, arch, seed=seed)

    if not state["output_specs"]:
        print("\n[warn] No output tensors detected. Nothing to validate.", file=sys.stderr)

    ref_inputs = {
        name: clone_value(value) for name, value in state["reference_inputs"].items()
    }

    print("\n[kernel]    running ... ", end="", flush=True)
    state["callable"]()
    torch.cuda.synchronize()
    print("done")

    print("[reference] running ... ", end="", flush=True)
    ref_fn(**ref_inputs)
    torch.cuda.synchronize()
    print("done")

    kernel_outputs = {
        name: tensor for name, tensor in state["tensor_inputs"].items()
        if name in {spec[0] for spec in state["output_specs"]}
    }
    ref_outputs = {
        name: tensor for name, tensor in ref_inputs.items()
        if isinstance(tensor, torch.Tensor) and name in kernel_outputs
    }

    validation_passed, md_rows = _validate_outputs(
        kernel_outputs,
        ref_outputs,
        state["output_specs"],
        _atol,
        _rtol,
    )

    print("=" * 60)
    print(f"  Target     : {os.path.basename(solution_file)}")
    print(f"  Backend    : cuda")
    print(f"  Reference  : {os.path.basename(ref_file)}")
    print(f"  GPU        : {gpu_name}")
    print(f"  Arch       : {arch}")
    print(f"  Dims       : {dim_values}")
    print(f"  Buf/ptr    : {state['ptr_elems']} elems")
    print(f"  Tolerance  : atol={_atol}  rtol={_rtol}")
    print("-" * 60)
    result_str = "ALL PASS" if validation_passed else "FAILED"
    print(f"  Result     : {_color(result_str, validation_passed)}")
    print("=" * 60)

    if output_dir and md_rows:
        md = _build_md(solution_file, ref_file, gpu_name, arch, dim_values,
                       state["ptr_elems"], _atol, _rtol, validation_passed, md_rows)
        _write_md_out(output_dir, md)

    if not validation_passed:
        sys.exit(1)



def main():
    parser = argparse.ArgumentParser(
        description="CUDA kernel correctness checker",
        epilog=(
            "Dimension args: pass --NAME=VALUE for each shape/scalar arg.\n"
            "The kernel must expose extern \"C\" void solve(...).\n"
            "ref.py must define `reference(**kwargs)` and may set module-level atol/rtol."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("solution_file", help="Path to CUDA kernel file (.cu)")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference .py file for validation")
    parser.add_argument("--ptr-size", type=int, default=0, help="Override element count for all pointer buffers")
    parser.add_argument("--arch", type=str, default="", help="GPU arch, e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input tensors (default: 42)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write correctness.md")

    args, unknown = parser.parse_known_args()

    dim_values = {}
    for item in unknown:
        if item.startswith("--") and "=" in item:
            key, val = item[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{item}'", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch(args.gpu)

    run(
        solution_file=args.solution_file,
        ref_file=args.ref,
        dim_values=dim_values,
        ptr_size_override=args.ptr_size,
        arch=arch,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
