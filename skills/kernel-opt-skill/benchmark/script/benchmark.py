#!/usr/bin/env python3
"""Benchmark CUDA kernel vs reference (PyTorch or CUTLASS).

Compares performance of a custom CUDA kernel against a reference implementation
using CUDA event timing and nsight-python hardware metrics.

The reference .py must define `reference(**kwargs)`.
The solution .cu must expose `extern "C" void solve(...)`.

Usage:
    python benchmark.py solution.cu --ref=ref.py --output-dir=./out --M=1024 --N=1024
    python benchmark.py solution.cu --ref=ref.py --output-dir=./out --skip-nsight --M=4096 --K=4096 --N=4096
"""

import argparse
import ctypes
import importlib.util
import os
import re
import statistics
import sys
from pathlib import Path

import torch

try:
    import nsight
except ImportError:
    print("nsight-python is not installed. Install with: pip install nsight-python", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Type tables for parsing extern "C" void solve(...)
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float*":         torch.float32,
    "double*":        torch.float64,
    "int*":           torch.int32,
    "long*":          torch.int64,
    "short*":         torch.int16,
    "char*":          torch.int8,
    "unsigned char*": torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*":  getattr(torch, "uint32", torch.int32),
}

_CTYPE_MAP = {
    "float*":          ctypes.c_void_p,
    "double*":         ctypes.c_void_p,
    "unsigned char*":  ctypes.c_void_p,
    "unsigned short*": ctypes.c_void_p,
    "unsigned int*":   ctypes.c_void_p,
    "char*":           ctypes.c_void_p,
    "short*":          ctypes.c_void_p,
    "long*":           ctypes.c_void_p,
    "int*":            ctypes.c_void_p,
    "int":             ctypes.c_int,
    "long":            ctypes.c_long,
    "size_t":          ctypes.c_size_t,
    "unsigned int":    ctypes.c_uint,
    "unsigned short":  ctypes.c_ushort,
    "unsigned char":   ctypes.c_ubyte,
    "char":            ctypes.c_char,
    "short":           ctypes.c_short,
}

_INT_TYPES = {"int", "long", "size_t", "unsigned int"}

# ---------------------------------------------------------------------------
# Helpers (self-contained, no cross-skill imports)
# ---------------------------------------------------------------------------

def _detect_arch(device_index=0):
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device_index)
        return f"sm_{major}{minor}"
    return "sm_80"


def _parse_signature(cu_file):
    with open(cu_file, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{', src)
    if not m:
        raise ValueError(f'Cannot find \'extern "C" void solve(...)\' in {cu_file}')
    raw = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())
    params = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        is_const = "const" in token
        clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
        matched = False
        for key in sorted(_CTYPE_MAP, key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            hit = re.match(rf"({base})\s+(\w+)", clean)
            if hit:
                params.append((key, hit.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token.strip()}'")
    return params


def _load_reference(ref_file):
    spec = importlib.util.spec_from_file_location("_ref", ref_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reference"):
        raise AttributeError(f"'{ref_file}' must define reference(**kwargs)")
    return mod


def _setup_kernel(cu_file, dim_values, ptr_size_override, arch, seed):
    """Load pre-compiled .so, allocate CUDA buffers, return callable + inputs."""
    params = _parse_signature(cu_file)

    so_path = os.path.splitext(cu_file)[0] + (".dll" if os.name == "nt" else ".so")
    if not os.path.exists(so_path):
        sys.exit(f"[error] .so not found: {so_path}\n"
                 f"        Compile first: nvcc -shared -std=c++17 -arch={arch} "
                 f"-O3 -Xcompiler -fPIC -o {so_path} {cu_file}")
    lib = ctypes.CDLL(so_path)

    for ptype, pname, _ in params:
        if ptype in _INT_TYPES and pname not in dim_values:
            raise ValueError(f"Missing dimension --{pname}=<value>")

    int_vals = [dim_values[n] for t, n, _ in params if t in _INT_TYPES]
    if ptr_size_override > 0:
        ptr_elems = ptr_size_override
    elif len(int_vals) == 0:
        ptr_elems = 1024 * 1024
    elif len(int_vals) == 1:
        ptr_elems = int_vals[0]
    else:
        sv = sorted(int_vals, reverse=True)
        ptr_elems = sv[0] * sv[1]
    ptr_elems = min(ptr_elems, 256 * 1024 * 1024)

    if seed is not None:
        torch.manual_seed(seed)

    tensors, ref_inputs, call_args, argtypes = {}, {}, [], []
    for ptype, pname, is_const in params:
        if ptype in _DTYPE_MAP:
            dtype = _DTYPE_MAP[ptype]
            t = (torch.randn(ptr_elems, device="cuda", dtype=dtype)
                 if dtype.is_floating_point
                 else torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_())
            tensors[pname] = t
            ref_inputs[pname] = t
            call_args.append(ctypes.c_void_p(t.data_ptr()))
            argtypes.append(ctypes.c_void_p)
        else:
            ctype = _CTYPE_MAP[ptype]
            val = dim_values[pname]
            ref_inputs[pname] = val
            call_args.append(ctype(val))
            argtypes.append(ctype)

    lib.solve.restype = None
    lib.solve.argtypes = argtypes

    return {
        "callable": lambda: lib.solve(*call_args),
        "tensors": tensors,
        "ref_inputs": ref_inputs,
        "output_names": [n for t, n, c in params if t in _DTYPE_MAP and not c],
    }


def _check_correctness(sol_tensors, ref_inputs_snapshot, ref_fn, output_names,
                       atol=1e-4, rtol=1e-3):
    """Run ref on cloned inputs and compare outputs with torch.allclose."""
    cloned = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in ref_inputs_snapshot.items()}
    ref_fn(**cloned)
    torch.cuda.synchronize()

    all_pass = True
    for name in output_names:
        if name not in cloned or not isinstance(cloned[name], torch.Tensor):
            continue
        ok = torch.allclose(sol_tensors[name].float(), cloned[name].float(),
                            atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False
    return all_pass

# ---------------------------------------------------------------------------
# NSight metrics
# ---------------------------------------------------------------------------

BENCH_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

METRIC_LABELS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":  "SM Throughput (% peak)",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":"Memory Throughput (% peak)",
    "dram__bytes.sum.per_second":                        "DRAM Bandwidth (bytes/s)",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved Occupancy (%)",
}

# module-level singletons — survive nsight re-entry
_sol_state = None
_ref_fn = None
_ref_inputs = None


def _init_solution(cu_file, dim_values, ptr_size, arch, seed):
    global _sol_state
    if _sol_state is None:
        _sol_state = _setup_kernel(cu_file, dim_values, ptr_size, arch, seed)
    return _sol_state


def _init_ref(ref_file, cu_file, dim_values, ptr_size, arch, seed):
    global _ref_fn, _ref_inputs
    if _ref_fn is None:
        mod = _load_reference(ref_file)
        _ref_fn = mod.reference
        state = _init_solution(cu_file, dim_values, ptr_size, arch, seed)
        _ref_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in state["ref_inputs"].items()
        }
    return _ref_fn, _ref_inputs


def _ref_call():
    _ref_fn(**_ref_inputs)

# ---------------------------------------------------------------------------
# Timing & profiling
# ---------------------------------------------------------------------------

def time_kernel(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0), times


def profile_nsight(fn, warmup):
    metrics = list(dict.fromkeys(BENCH_METRICS))

    @nsight.analyze.kernel(
        metrics=metrics,
        runs=1,
        output="quiet",
        output_csv=False,
        clock_control="none",
        cache_control="all",
    )
    def _run(n_warmup):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        with nsight.annotate("benchmark"):
            fn()

    return _run(warmup).to_dataframe()


def _get_metric(df, name):
    for _, row in df.iterrows():
        if row.get("Metric") == name:
            return row.get("AvgValue")
    return None


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2e}" if abs(v) > 1e6 else f"{v:.4f}"
    return str(v)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(solution_file, ref_file, dim_values, arch,
                 sol_ms, sol_std, sol_df,
                 ref_ms, ref_std, ref_df,
                 correctness_pass):
    gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    correctness_str = "PASS" if correctness_pass else "FAIL"

    lines = [
        "# Benchmark Report",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Solution** | `{os.path.basename(solution_file)}` |",
        f"| **Reference** | `{os.path.basename(ref_file)}` |",
        f"| **GPU** | {gpu} |",
        f"| **Arch** | {arch} |",
        f"| **Dims** | {dim_values} |",
        f"| **Correctness** | {correctness_str} |",
        "",
        "## Timing (CUDA Events)",
        "",
        "| Metric | Solution | Reference |",
        "|--------|----------:|----------:|",
        f"| Execution Time (ms) | {sol_ms:.4f} | {ref_ms:.4f} |",
        f"| Std dev (ms)        | {sol_std:.4f} | {ref_std:.4f} |",
        "",
    ]

    if sol_df is not None and ref_df is not None:
        lines += [
            "## Hardware Metrics (nsight-python)",
            "",
            "| Metric | Solution | Reference |",
            "|--------|----------:|----------:|",
        ]
        for metric, label in METRIC_LABELS.items():
            sv = _get_metric(sol_df, metric)
            rv = _get_metric(ref_df, metric)
            lines.append(f"| {label} | {_fmt(sv)} | {_fmt(rv)} |")
        lines.append("")
    elif sol_df is not None or ref_df is not None:
        lines.append("> *nsight profiling only succeeded for one side — no comparison table.*")
        lines.append("")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA kernel vs reference (PyTorch/CUTLASS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("solution_file", help="Path to CUDA kernel file (.cu)")
    parser.add_argument("--ref", type=str, required=True,
                        help="Path to reference .py defining reference(**kwargs)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output files")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations (default: 20)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timing iterations for CUDA event measurement (default: 100)")
    parser.add_argument("--ptr-size", type=int, default=0,
                        help="Override element count for pointer buffers")
    parser.add_argument("--arch", type=str, default="",
                        help="GPU arch e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--atol", type=float, default=1e-4, help="Correctness atol (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Correctness rtol (default: 1e-3)")
    parser.add_argument("--skip-nsight", action="store_true",
                        help="Skip nsight hardware profiling; only run CUDA event timing")

    args, unknown = parser.parse_known_args()

    dim_values = {}
    for item in unknown:
        if item.startswith("--") and "=" in item:
            key, val = item[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{item}'", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else _detect_arch(args.gpu)
    solution_file = str(Path(args.solution_file).resolve())
    ref_file = str(Path(args.ref).resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # nsight re-launches the whole process to inject the profiler;
    # skip all non-profiling work in those child processes
    under_nsight = (
        "CUDA_INJECTION64_PATH" in os.environ
        or "NV_NSIGHT_INJECTION64_PATH" in os.environ
        or any(k.startswith("NV_NSIGHT_") for k in os.environ)
    )

    if not under_nsight:
        print(f"[benchmark] solution  : {solution_file}")
        print(f"[benchmark] reference : {ref_file}")
        print(f"[benchmark] arch      : {arch}")
        print(f"[benchmark] dims      : {dim_values}")
        print()

    state = _init_solution(solution_file, dim_values, args.ptr_size, arch, args.seed)
    sol_fn = state["callable"]
    _init_ref(ref_file, solution_file, dim_values, args.ptr_size, arch, args.seed)

    if not under_nsight:
        # correctness check
        sol_fn()
        torch.cuda.synchronize()
        print("[correctness] checking...")
        ok = _check_correctness(
            sol_tensors=state["tensors"],
            ref_inputs_snapshot=state["ref_inputs"],
            ref_fn=_ref_fn,
            output_names=state["output_names"],
            atol=args.atol,
            rtol=args.rtol,
        )
        print(f"[correctness] {'PASS' if ok else 'FAIL'}\n")

        # CUDA event timing
        print(f"[timing] solution  ({args.warmup} warmup, {args.iters} iters)...")
        sol_ms, sol_std, sol_times = time_kernel(sol_fn, args.warmup, args.iters)
        print(f"[timing] solution  : {sol_ms:.4f} ms ± {sol_std:.4f} ms")

        print(f"[timing] reference ({args.warmup} warmup, {args.iters} iters)...")
        ref_ms, ref_std, ref_times = time_kernel(_ref_call, args.warmup, args.iters)
        print(f"[timing] reference : {ref_ms:.4f} ms ± {ref_std:.4f} ms")

    # nsight profiling — triggers re-entry; runs in both parent and child,
    # but only captures metrics when nsight.is_injected() is True
    sol_df = ref_df = None
    if not args.skip_nsight:
        if not under_nsight:
            print("\n[nsight] profiling solution...")
        try:
            sol_df = profile_nsight(sol_fn, args.warmup)
        except Exception as exc:
            print(f"[nsight] solution profiling failed: {exc}", file=sys.stderr)

        if not under_nsight:
            print("[nsight] profiling reference...")
        try:
            ref_df = profile_nsight(_ref_call, args.warmup)
        except Exception as exc:
            print(f"[nsight] reference profiling failed: {exc}", file=sys.stderr)

    if under_nsight:
        return 0

    report = build_report(
        solution_file=solution_file,
        ref_file=ref_file,
        dim_values=dim_values,
        arch=arch,
        sol_ms=sol_ms,
        sol_std=sol_std,
        sol_df=sol_df,
        ref_ms=ref_ms,
        ref_std=ref_std,
        ref_df=ref_df,
        correctness_pass=ok,
    )
    report_path = output_dir / "benchmark.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[benchmark] report    -> {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
