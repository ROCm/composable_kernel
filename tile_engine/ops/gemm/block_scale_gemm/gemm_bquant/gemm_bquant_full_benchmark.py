#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Full BQuant GEMM benchmark sweep driven through the Dispatcher bridge.

Mirrors tile_engine/ops/gemm/gemm_full_benchmark.py (the regular-GEMM bridge
driver from PR #8997) but adapted for BQuantGrouped GEMM.

Phases:
  Phase 1: Compile all kernels (parallel, returns .so paths only -- no GPU)
  Phase 2: Load problems (M, N, K, quant_group_k shapes)
  Phase 3: Benchmark via subprocess isolation, distributed across all visible
           GPUs (one device-pinned worker per GPU, batched, fault-isolated)

Tile Engine generates NO binaries here: it expands its sweep config into
BQuantKernelConfig objects and hands them to the dispatcher, which codegens +
compiles each into a .so. Each kernel runs in a disposable worker subprocess so
a GPU fault takes down only one worker.

Examples:
    # Default: fp8 variant, CI sweep config, auto-detect all visible GPUs.
    python gemm_bquant_full_benchmark.py

    # Explicit config + dtype on 2 GPUs:
    python gemm_bquant_full_benchmark.py configs/default_config.json \\
        --dtype fp8 --arch gfx950 --devices 2 --csv bquant_out.csv

When no config is given the driver uses configs/default_ci_config.json (a small
CI-sized sweep); configs/default_config.json is the full sweep.
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
# Dispatcher python utilities are three levels up from here:
# gemm_bquant/ -> block_scale_gemm/ -> gemm/ -> tile_engine/ops/ -> ...
# -> projects/composablekernel/dispatcher/python
_COMMON_DIR = _THIS_DIR.parents[2] / "common"
_DISPATCHER_ROOT = _THIS_DIR.parents[5] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_COMMON_DIR))
sys.path.insert(0, str(_THIS_DIR))

from grouped_gemm_bquant_utils import (  # noqa: E402
    setup_multiple_bquant_dispatchers,
    expand_bquant_sweep,
)
from smi_utils import detect_gpu_ids  # noqa: E402

# Dispatcher-schema configs live alongside the legacy tile-engine configs.
# The "dispatcher_*" files use the flat tile_configs/quant_groups format
# consumed by expand_bquant_sweep(); the legacy "default_*" files use the
# tile_config/trait_config range format consumed by GemmBQuantKernelBuilder.
CI_CONFIG_NAME        = "dispatcher_ci_config.json"
EXAMPLE_PROBLEMS_NAME = "example_problems.json"

SUPPORTED_DTYPES = ("fp8", "bf8", "fp8i4", "bf8i4", "mx_bf16bf16", "mx_bf16bf8", "mx_bf16fp4")

DEFAULT_PROBLEMS = [
    {"M": 128,  "N": 128,  "K": 128,  "quant_group_k": 128},
    {"M": 1024, "N": 1024, "K": 1024, "quant_group_k": 128},
    {"M": 4096, "N": 4096, "K": 4096, "quant_group_k": 128},
    {"M": 257,  "N": 257,  "K": 256,  "quant_group_k": 128},
]


def detect_devices():
    """Return a list of visible GPU id strings (best-effort)."""
    return detect_gpu_ids()


def resolve_devices(spec):
    """Resolve --devices into a concrete list of device id strings."""
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
        if len(detected) >= n:
            return detected[:n]
        if os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"):
            return detected
        return [str(i) for i in range(n)]
    return [spec]


def resolve_config(args):
    """Resolve positional configs -> concrete list of config paths."""
    if args.configs:
        return args.configs
    cfg = _THIS_DIR / "configs" / CI_CONFIG_NAME
    return [str(cfg)]


def load_problems(path):
    """Load BQuant problem shapes from a JSON file or return defaults."""
    if path:
        with open(path) as f:
            data = json.load(f)
        return data["problems"] if isinstance(data, dict) else data
    example = _THIS_DIR / "configs" / EXAMPLE_PROBLEMS_NAME
    if example.exists():
        with open(example) as f:
            data = json.load(f)
        return data["problems"] if isinstance(data, dict) else data
    return DEFAULT_PROBLEMS


def _run_batch_on_device(device_id, unit, args, worker_path, base_env):
    """Run one (problem, kernel-batch) unit in a device-pinned subprocess."""
    prob_idx, prob_dict, batch = unit
    M = prob_dict["M"]
    N = prob_dict["N"]
    K = prob_dict["K"]

    items = [
        {"so_path": str(lib), "problem": prob_dict, "kernel_name": cfg.name}
        for _, cfg, lib in batch
    ]
    payload = json.dumps({"items": items})

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
                lines.append(f"  [gpu{device_id}] Warning: bad result line: {line[:50]}")
                n_fail += 1
                continue
            bidx = result.get("idx", 0)
            _, cfg, _ = batch[bidx]
            reported.add(bidx)
            if result.get("ok", False):
                status = "OK" if result.get("non_zero", 0) > 0 else "ZERO"
                lines.append(
                    f"  [gpu{device_id}] {cfg.name:<62} {result['ms']:>10.3f} "
                    f"{result['tflops']:>10.2f} {status:>6}"
                )
                rows.append({
                    "kernel": cfg.name,
                    "problem_idx": prob_idx,
                    "M": M, "N": N, "K": K,
                    "quant_group_k": prob_dict.get("quant_group_k", 128),
                    "device": device_id,
                    "latency_ms": result["ms"],
                    "tflops": result["tflops"],
                    "non_zero": result.get("non_zero", 0),
                })
            else:
                lines.append(f"  [gpu{device_id}] {cfg.name:<62} FAILED")
                lines.append(f"    Error: {result.get('error', 'unknown')[:100]}")
                n_fail += 1

        missing = set(range(len(batch))) - reported
        if missing or proc.returncode != 0:
            if proc.returncode != 0:
                lines.append(f"  [gpu{device_id}] worker exited code {proc.returncode}")
            for idx in sorted(missing):
                _, cfg, _ = batch[idx]
                lines.append(f"  [gpu{device_id}] {cfg.name:<62} MISSING (crash)")
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
    parser = argparse.ArgumentParser(description="BQuant GEMM Benchmark Sweep (via Dispatcher)")
    parser.add_argument(
        "configs",
        nargs="*",
        help="BQuant sweep config JSON files (default: configs/default_ci_config.json)",
    )
    parser.add_argument("--arch", default="gfx950", help="GPU architecture (default: gfx950)")
    parser.add_argument(
        "--dtype",
        default="fp8",
        choices=SUPPORTED_DTYPES,
        help=f"Primary input dtype to filter from config (supported: {', '.join(SUPPORTED_DTYPES)})",
    )
    parser.add_argument("--problems", default=None, help="JSON file of {M,N,K,quant_group_k} problems")
    parser.add_argument("--csv", type=str, default="gemm_bquant_results.csv")
    parser.add_argument("--workers", type=int, default=8, help="Parallel build workers")
    parser.add_argument(
        "--devices",
        default=None,
        help="GPUs to use: int count (e.g. 4) or comma-list (e.g. 0,2,5); "
        "bare digit is a count, not an id; default auto-detects all visible",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Kernels per subprocess (overhead vs fault isolation)",
    )
    parser.add_argument(
        "--kernel-timeout", type=int, default=60, help="Per-kernel timeout (s)"
    )
    parser.add_argument(
        "--max-kernels", type=int, default=0, help="Limit to first N kernels (0=all)"
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Run Phase 1 only (compile all kernels, skip benchmarking)",
    )
    args = parser.parse_args()

    config_paths = resolve_config(args)
    devices = resolve_devices(args.devices)

    # ========================================================================
    # Phase 1: Compile kernels (parallel, no GPU)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile BQuant kernels")
    print(f"{'=' * 80}")
    print(f"  Configs: {', '.join(config_paths)}")
    print(f"  Arch: {args.arch}")
    print(f"  Dtype filter: {args.dtype}")

    all_configs = []
    for cfg_path in config_paths:
        expanded = expand_bquant_sweep(cfg_path, gfx_arch=args.arch)
        # Apply dtype filter: include configs whose variant_key starts with the dtype prefix
        dtype_prefix = args.dtype.split("i")[0].split("_")[0]  # "fp8i4" -> "fp8", "mx_bf16bf16" -> "mx"
        filtered = [c for c in expanded if c.variant_key.startswith(dtype_prefix)]
        all_configs.extend(filtered if filtered else expanded)

    if args.max_kernels > 0:
        all_configs = all_configs[:args.max_kernels]

    print(f"  Expanded configs: {len(all_configs)}")
    print(f"  Build workers: {args.workers}")

    t0 = time.perf_counter()
    lib_paths = setup_multiple_bquant_dispatchers(
        all_configs, gfx_arch=args.arch, max_workers=args.workers
    )
    build_time = time.perf_counter() - t0

    built_kernels = [
        (cfg, lib) for cfg, lib in zip(all_configs, lib_paths) if lib is not None
    ]

    # Dedupe by .so path
    seen_libs: set = set()
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

    if args.compile_only:
        print("\n  --compile-only: skipping benchmark phases")
        return 0

    # ========================================================================
    # Phase 2: Load problems
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 2: Load BQuant test problems")
    print(f"{'=' * 80}")

    problems = load_problems(args.problems)
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
        "kernel", "problem_idx", "M", "N", "K", "quant_group_k",
        "device", "latency_ms", "tflops", "non_zero",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    worker_path = _THIS_DIR / "run_one_bquant_kernel.py"
    base_env = os.environ.copy()
    base_env["BQUANT_PYPATH"] = os.pathsep.join(
        [str(_DISPATCHER_ROOT / "python"), str(_THIS_DIR)]
    )

    work_q: queue.Queue = queue.Queue()
    for prob_idx, prob in enumerate(problems):
        prob_dict = {
            "M": int(prob["M"]), "N": int(prob["N"]), "K": int(prob["K"]),
            "quant_group_k": int(prob.get("quant_group_k", 128)),
        }
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
