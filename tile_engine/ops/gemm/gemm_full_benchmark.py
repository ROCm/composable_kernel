#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Full GEMM benchmark sweep driven through the Dispatcher bridge.

Phases:
  Phase 1: Compile all kernels (parallel, returns .so paths only -- no GPU)
  Phase 2: Load problems (M, N, K shapes)
  Phase 3: Benchmark via subprocess isolation, distributed across all visible
           GPUs (one device-pinned worker per GPU, batched, fault-isolated)

Tile Engine generates NO binaries here: it expands its sweep config into shared
``GemmKernelConfig`` objects and hands them to the dispatcher, which codegens +
compiles each into a .so. Each kernel runs in a disposable worker subprocess so
a GPU fault (or ctypes' inability to unload a .so) takes down only one worker.

Unlike the serial-GPU design inherited from grouped_conv, Phase 3 here fans the
work out across every visible GPU in parallel: each device runs its own stream of
disposable worker subprocesses pinned with ``HIP_VISIBLE_DEVICES``, so an N-GPU
box benchmarks roughly N times faster while keeping per-batch fault isolation.

Examples:
    # Default: gemm_universal variant, its CI sweep config + example problems,
    # auto-detect and use all visible GPUs.
    python gemm_full_benchmark.py

    # Explicit variant + full sweep config on 4 GPUs:
    python gemm_full_benchmark.py --variant gemm_universal \
        configs/default_config.json --devices 4 --csv out.csv

When no config is given the driver uses the chosen variant's
``configs/default_ci_config.json`` (a small CI-sized sweep);
``configs/default_config.json`` is the full sweep, and the JSON used by nightly
tests is intended to drop into the same ``configs/`` directory.
"""

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "common"
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_COMMON_DIR))
sys.path.insert(0, str(_THIS_DIR))

from gemm_utils import setup_multiple_gemm_dispatchers, expand_sweep  # noqa: E402
from smi_utils import detect_gpu_ids  # noqa: E402

# Config layout. The bridged regular-GEMM path (gemm_universal) keeps its sweep
# configs in this op's flat ``configs/`` directory (matching the fmha/grouped_conv
# bridge convention): default_ci_config.json (small CI sweep), default_config.json
# (full sweep), user_provided_config.json, example_problems.json. The other,
# not-yet-bridged variants still live in their own per-variant ``configs/`` dirs;
# they are registered so ``--variant`` can select them once their bridge lands.
VARIANT_CONFIGS = {
    "gemm_universal": "configs",
    "gemm_multi_d": "gemm_multi_d/configs",
    "gemm_multi_abd": "gemm_multi_abd/configs",
    "gemm_preshuffle": "gemm_preshuffle/configs",
    "grouped_gemm": "grouped_gemm/configs",
}
DEFAULT_VARIANT = "gemm_universal"

CI_CONFIG_NAME = "default_ci_config.json"
EXAMPLE_PROBLEMS_NAME = "example_problems.json"

# Map the driver's --variant (a configs-dir selector) onto the single codegen/
# runtime variant token understood by expand_sweep / unified_gemm_codegen /
# GemmKernelConfig.variant. Every --variant choice must have an entry here.
CODEGEN_VARIANT = {
    "gemm_universal": "standard",
    "gemm_multi_d": "multi_d",
    "gemm_multi_abd": "multi_abd",
    "gemm_preshuffle": "preshuffle",
    "grouped_gemm": "grouped",
}

# Some variants only support a subset of dtypes/layouts. The preshuffle op
# (tile_engine gemm_preshuffle) supports fp16/bf16/fp8/bf8 and rcr ONLY.
VARIANT_SUPPORTED_DTYPES = {
    "gemm_preshuffle": ("fp16", "bf16", "fp8", "bf8"),
}
VARIANT_SUPPORTED_LAYOUTS = {
    "gemm_preshuffle": ("rcr",),
}

# Fallback problem set if a variant ships no example_problems.json.
DEFAULT_PROBLEMS = [
    {"M": 1024, "N": 1024, "K": 1024},
    {"M": 2048, "N": 2048, "K": 2048},
    {"M": 4096, "N": 4096, "K": 4096},
    {"M": 257, "N": 257, "K": 257},
]

SUPPORTED_DTYPES = ("fp16", "bf16", "fp8", "bf8")
# Row-major C only: ck_tile's universal GEMM rejects column-major C at build.
# The 4-char codes (rcrr, ...) are the multi_abd A,B,E,D layouts; TE gemm_multi_abd
# only supports rcrr today.
SUPPORTED_LAYOUTS = ("rcr", "rrr", "crr", "ccr", "rcrr", "rrrr", "crrr", "ccrr")


def detect_devices():
    """Return a list of visible GPU id strings (best-effort)."""
    return detect_gpu_ids()


def resolve_devices(spec):
    """Resolve --devices into a concrete list of device id strings.

    spec is None (auto: all visible), an int count, or a comma-list of ids.
    A bare digit is a *count*, not an id; to target one specific id use the
    comma form, e.g. "5,".
    """
    detected = detect_devices()
    if spec is None:
        return detected
    spec = str(spec).strip()
    if "," in spec:
        return [s.strip() for s in spec.split(",") if s.strip() != ""]
    if spec.isdigit():
        n = int(spec)
        if n <= 0:
            return detected
        # Treat a bare integer as a device *count*: take the first n detected ids.
        # If the environment explicitly restricts visibility (HIP/CUDA_VISIBLE_DEVICES),
        # do not invent additional ids beyond what's visible.
        if len(detected) >= n:
            return detected[:n]
        if os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ):
            return detected
        return [str(i) for i in range(n)]
    return [spec]


def resolve_configs(args):
    """Resolve positional configs -> concrete list of config paths."""
    if args.configs:
        return args.configs
    cfg = _THIS_DIR / VARIANT_CONFIGS[args.variant] / CI_CONFIG_NAME
    return [str(cfg)]


def load_problems(path, variant):
    if path:
        with open(path) as f:
            data = json.load(f)
        return data["problems"] if isinstance(data, dict) else data
    example = _THIS_DIR / VARIANT_CONFIGS[variant] / EXAMPLE_PROBLEMS_NAME
    if example.exists():
        with open(example) as f:
            data = json.load(f)
        return data["problems"] if isinstance(data, dict) else data
    return DEFAULT_PROBLEMS


def _run_batch_on_device(device_id, unit, args, worker_path, base_env):
    """Run one (problem, kernel-batch) unit in a device-pinned subprocess.

    Returns (rows, lines, n_fail) where rows are dicts ready for the CSV writer,
    lines are formatted strings to print, and n_fail counts failures.
    """
    prob_idx, prob_dict, batch = unit
    M, N, K = prob_dict["M"], prob_dict["N"], prob_dict["K"]

    def _item(cfg, lib):
        it = {"so_path": str(lib), "problem": prob_dict, "kernel_name": cfg.name}
        # N2: carry the multi_abd layout + per-group element-wise ops from the
        # config object so the worker's numpy reference uses the real ops/layout
        # instead of parsing (and defaulting) the kernel name.
        if getattr(cfg, "variant", "standard") == "multi_abd":
            it["layout4"] = cfg.layout4
            it["a_elementwise_op"] = cfg.a_elementwise_op
            it["b_elementwise_op"] = cfg.b_elementwise_op
            it["cde_elementwise_op"] = cfg.cde_elementwise_op
        return it

    items = [_item(cfg, lib) for _, cfg, lib in batch]
    payload = json.dumps(
        {"items": items, "verify": args.verify, "verify_tol": args.verify_tol}
    )

    env = base_env.copy()
    env["HIP_VISIBLE_DEVICES"] = str(device_id)

    rows, lines, n_fail = [], [], 0
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        stdout_bytes, _ = proc.communicate(
            input=payload.encode("utf-8"),
            timeout=args.kernel_timeout * len(batch),
        )

        reported = set()
        for line in stdout_bytes.decode("utf-8").strip().split("\n"):
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                lines.append(
                    f"  [gpu{device_id}] Warning: bad result line: {line[:50]}"
                )
                n_fail += 1
                continue
            bidx = result.get("idx", 0)
            _, cfg, _ = batch[bidx]
            reported.add(bidx)
            if result.get("ok", False):
                status = "OK" if result.get("non_zero", 0) > 0 else "ZERO"
                mismatch = False
                if args.verify and "verified" in result:
                    if result["verified"]:
                        status = "VERIFY"
                    else:
                        status = "MISMATCH"
                        mismatch = True
                extra = f" rel={result['max_rel']:.2e}" if "max_rel" in result else ""
                lines.append(
                    f"  [gpu{device_id}] {cfg.name:<58} {result['ms']:>10.3f} "
                    f"{result['tflops']:>10.2f} {status:>8}{extra}"
                )
                rows.append(
                    {
                        "kernel": cfg.name,
                        "problem_idx": prob_idx,
                        "M": M,
                        "N": N,
                        "K": K,
                        "device": device_id,
                        "latency_ms": result["ms"],
                        "tflops": result["tflops"],
                        "non_zero": result.get("non_zero", 0),
                        "max_rel": result.get("max_rel", ""),
                        "verified": result.get("verified", ""),
                    }
                )
                if mismatch:
                    n_fail += 1
            else:
                lines.append(f"  [gpu{device_id}] {cfg.name:<58} FAILED")
                lines.append(f"    Error: {result.get('error', 'unknown')[:100]}")
                n_fail += 1

        missing = set(range(len(batch))) - reported
        if missing or proc.returncode != 0:
            if proc.returncode != 0:
                lines.append(f"  [gpu{device_id}] worker exited code {proc.returncode}")
            for idx in sorted(missing):
                _, cfg, _ = batch[idx]
                lines.append(f"  [gpu{device_id}] {cfg.name:<58} MISSING (crash)")
            n_fail += len(missing)

    except subprocess.TimeoutExpired:
        lines.append(f"  [gpu{device_id}] batch timeout ({len(batch)} kernels)")
        try:
            proc.kill()
            proc.communicate(timeout=5)
        except Exception:
            pass
        n_fail += len(batch)
    except Exception as e:
        lines.append(f"  [gpu{device_id}] batch error: {e}")
        try:
            if proc and proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        n_fail += len(batch)

    return rows, lines, n_fail


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Benchmark Sweep (via Dispatcher)"
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="TE sweep config JSON files (default: variant's default_ci_config.json)",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=tuple(VARIANT_CONFIGS),
        help="GEMM variant (selects the configs/ directory)",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="GPU arch (e.g. gfx942/gfx950). Auto-detected via rocminfo when "
        "omitted; never silently defaulted to a specific GPU.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=SUPPORTED_DTYPES,
        help=f"Input dtype (supported: {', '.join(SUPPORTED_DTYPES)})",
    )
    parser.add_argument(
        "--layout",
        default="rcr",
        choices=SUPPORTED_LAYOUTS,
        help=f"A/B/C layout (supported: {', '.join(SUPPORTED_LAYOUTS)})",
    )
    # Multi-ABD only: per-group element-wise ops + tensor counts. These override
    # (are merged into) any multi_abd_config block in the config JSON so a single
    # non-PassThrough kernel can be driven + verified without editing configs.
    # Valid ops mirror the Old-TE gemm_multi_abd instance builder.
    _MABD_OPS = ("PassThrough", "AddScale", "MultiDMultiply", "MultiDAdd")
    parser.add_argument("--multi-abd-num-a", type=int, default=None,
                        help="multi_abd: number of A tensors (default: config/2)")
    parser.add_argument("--multi-abd-num-b", type=int, default=None,
                        help="multi_abd: number of B tensors (default: config/2)")
    parser.add_argument("--multi-abd-num-d", type=int, default=None,
                        help="multi_abd: number of D tensors (default: config/2)")
    parser.add_argument("--multi-abd-a-op", default=None, choices=_MABD_OPS,
                        help="multi_abd: A element-wise op (default: config/PassThrough)")
    parser.add_argument("--multi-abd-b-op", default=None, choices=_MABD_OPS,
                        help="multi_abd: B element-wise op (default: config/PassThrough)")
    parser.add_argument("--multi-abd-cde-op", default=None, choices=_MABD_OPS,
                        help="multi_abd: CDE element-wise op (default: config/PassThrough)")
    parser.add_argument("--problems", default=None, help="JSON file of M,N,K problems")
    parser.add_argument("--csv", type=str, default="gemm_results.csv")
    parser.add_argument("--workers", type=int, default=8, help="Parallel build workers")
    parser.add_argument(
        "--devices",
        default=None,
        help="GPUs to use: int count (e.g. 4) or comma-list of ids (e.g. 0,2,5); "
        "for one specific id use the comma form (e.g. 5,) since a bare digit is "
        "a count; default auto-detects all visible",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Kernels per subprocess (overhead vs fault isolation)",
    )
    parser.add_argument(
        "--kernel-timeout", type=int, default=30, help="Per-kernel timeout (s)"
    )
    parser.add_argument(
        "--max-kernels", type=int, default=0, help="Limit to first N kernels (0=all)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check each kernel's output against an fp32 numpy reference "
        "(global max|out-ref|/max|ref|); a mismatch counts as a failure",
    )
    parser.add_argument(
        "--verify-tol",
        type=float,
        default=2e-2,
        help="Relative tolerance for --verify (default 2e-2, suits fp16)",
    )
    args = parser.parse_args()

    config_paths = resolve_configs(args)
    devices = resolve_devices(args.devices)

    # ========================================================================
    # Phase 1: Compile kernels (parallel, no GPU)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile kernels")
    print(f"{'=' * 80}")
    print(f"  Variant: {args.variant}")
    print(f"  Configs: {', '.join(config_paths)}")

    if args.variant == "grouped_gemm":
        print(
            "  ERROR: grouped_gemm is not supported by this driver; "
            "use tile_engine/ops/gemm/grouped_gemm/grouped_gemm_benchmark.py"
        )
        return 1
    codegen_variant = CODEGEN_VARIANT[args.variant]
    # Per-variant dtype/layout guards (e.g. preshuffle is rcr-only, no fp32).
    ok_dtypes = VARIANT_SUPPORTED_DTYPES.get(args.variant)
    if ok_dtypes and args.dtype not in ok_dtypes:
        print(
            f"  ERROR: variant {args.variant} supports dtypes {ok_dtypes}, "
            f"got {args.dtype!r}"
        )
        return 1
    ok_layouts = VARIANT_SUPPORTED_LAYOUTS.get(args.variant)
    if ok_layouts and args.layout not in ok_layouts:
        print(
            f"  ERROR: variant {args.variant} supports layouts {ok_layouts}, "
            f"got {args.layout!r}"
        )
        return 1
    # Multi-ABD needs the 4-char (A,B,E,D) layout; if the user left the 3-char
    # default in place, extend it (D defaults to the C/E layout).
    sweep_layout = args.layout
    if codegen_variant == "multi_abd" and len(sweep_layout) == 3:
        sweep_layout = sweep_layout + sweep_layout[2]
    # multi_abd supports only the 'rcrr' layout today; reject anything else up
    # front instead of silently building an unsupported/divergent kernel.
    if codegen_variant == "multi_abd" and sweep_layout != "rcrr":
        raise SystemExit(
            f"multi_abd supports only the 'rcrr' layout today, got {sweep_layout!r}"
        )

    # Multi-ABD element-wise ops / tensor counts: CLI overrides win over the
    # config; otherwise expand_sweep falls back to any multi_abd_config block in
    # the JSON and finally to the Old-TE 2/2/2 all-PassThrough default.
    mabd_kwargs = {}
    if codegen_variant == "multi_abd":
        if args.multi_abd_num_a is not None:
            mabd_kwargs["num_a_tensors"] = args.multi_abd_num_a
        if args.multi_abd_num_b is not None:
            mabd_kwargs["num_b_tensors"] = args.multi_abd_num_b
        if args.multi_abd_num_d is not None:
            mabd_kwargs["num_d_tensors"] = args.multi_abd_num_d
        if args.multi_abd_a_op is not None:
            mabd_kwargs["a_elementwise_op"] = args.multi_abd_a_op
        if args.multi_abd_b_op is not None:
            mabd_kwargs["b_elementwise_op"] = args.multi_abd_b_op
        if args.multi_abd_cde_op is not None:
            mabd_kwargs["cde_elementwise_op"] = args.multi_abd_cde_op

    all_configs = []
    for cfg_path in config_paths:
        all_configs.extend(
            expand_sweep(
                cfg_path,
                args.arch,
                dtype=args.dtype,
                layout=sweep_layout,
                variant=codegen_variant,
                mabd_cli_overrides=(mabd_kwargs or None),
                **mabd_kwargs,
            )
        )

    if args.max_kernels > 0:
        all_configs = all_configs[: args.max_kernels]

    print(f"  Expanded configs: {len(all_configs)}")
    print(f"  Build workers: {args.workers}")

    t0 = time.perf_counter()
    # CRITICAL: returns Path objects only, does NOT load any .so.
    lib_paths = setup_multiple_gemm_dispatchers(
        all_configs, verbose=True, max_workers=args.workers
    )
    build_time = time.perf_counter() - t0

    built_kernels = [
        (cfg, lib) for cfg, lib in zip(all_configs, lib_paths) if lib is not None
    ]

    # Dedupe by .so path (distinct configs can map to the same physical kernel).
    seen_libs = set()
    unique_kernels = []
    duplicate_count = 0
    for cfg, lib in built_kernels:
        lib_key = str(lib.resolve())
        if lib_key not in seen_libs:
            seen_libs.add(lib_key)
            unique_kernels.append((cfg, lib))
        else:
            duplicate_count += 1
    built_kernels = unique_kernels

    print(
        f"\n  Built {len(all_configs)} configs -> {len(built_kernels)} unique kernels "
        f"({duplicate_count} duplicates filtered) in {build_time:.0f}s"
    )

    if not built_kernels:
        print("  ERROR: No kernels built successfully")
        return 1

    # ========================================================================
    # Phase 2: Load problems
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 2: Load test problems")
    print(f"{'=' * 80}")

    problems = load_problems(args.problems, args.variant)
    print(f"  Problems: {len(problems)}")
    print(
        f"  Total measurements: {len(built_kernels)} x {len(problems)} = "
        f"{len(built_kernels) * len(problems)}"
    )

    # ========================================================================
    # Phase 3: Benchmark across all visible GPUs (subprocess isolation, batched)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 3: Benchmark (multi-GPU, subprocess isolation, batched)")
    print(f"{'=' * 80}")
    print(f"  Devices: {len(devices)} -> {', '.join(devices)}")
    print(f"  Batch size: {args.batch_size} kernels per subprocess")
    print(f"  Timeout: {args.kernel_timeout}s per kernel\n")

    csv_path = Path(args.csv)
    csv_fields = [
        "kernel",
        "problem_idx",
        "M",
        "N",
        "K",
        "device",
        "latency_ms",
        "tflops",
        "non_zero",
        "max_rel",
        "verified",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    worker_path = _THIS_DIR / "run_one_gemm_kernel.py"
    base_env = os.environ.copy()
    base_env["GEMM_PYPATH"] = os.pathsep.join(
        [str(_DISPATCHER_ROOT / "python"), str(_THIS_DIR)]
    )

    # Build a single work queue of (prob_idx, prob_dict, kernel-batch) units and
    # fan them out across device-pinned worker threads.
    work_q = queue.Queue()
    for prob_idx, prob in enumerate(problems):
        prob_dict = {"M": int(prob["M"]), "N": int(prob["N"]), "K": int(prob["K"])}
        for start in range(0, len(built_kernels), args.batch_size):
            end = min(start + args.batch_size, len(built_kernels))
            batch = [
                (start + j, cfg, lib)
                for j, (cfg, lib) in enumerate(built_kernels[start:end])
            ]
            work_q.put((prob_idx, prob_dict, batch))

    io_lock = threading.Lock()
    stats = {"measurements": 0, "failures": 0}
    bench_t0 = time.perf_counter()

    def device_thread(device_id):
        while True:
            try:
                unit = work_q.get_nowait()
            except queue.Empty:
                return
            rows, lines, n_fail = _run_batch_on_device(
                device_id, unit, args, worker_path, base_env
            )
            with io_lock:
                for ln in lines:
                    print(ln)
                for row in rows:
                    writer.writerow(row)
                csv_file.flush()
                stats["measurements"] += len(rows)
                stats["failures"] += n_fail
            work_q.task_done()

    threads = [
        threading.Thread(target=device_thread, args=(d,), daemon=True) for d in devices
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    bench_time = time.perf_counter() - bench_t0
    csv_file.close()

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Build time: {build_time:.0f}s")
    print(f"  Benchmark time: {bench_time:.0f}s")
    print(f"  Total time: {build_time + bench_time:.0f}s")
    print(f"  Devices used: {len(devices)}")
    print(f"  Successful measurements: {stats['measurements']}")
    print(f"  Failed measurements: {stats['failures']}")
    print(f"  Output: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
