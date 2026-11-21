#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Full BATCHED GEMM benchmark sweep driven through the Dispatcher bridge.

Batched counterpart of gemm_full_benchmark.py.

Phases:
  Phase 1: Compile all batched kernels (parallel, returns .so paths -- no GPU)
  Phase 2: Load batched problems (batch_count, M, N, K)
  Phase 3: Benchmark via subprocess isolation, fanned across all visible GPUs

Tile Engine generates NO binaries here: it expands its sweep config into shared
``BatchedGemmKernelConfig`` objects and hands them to the dispatcher, which
codegens + compiles each into a .so. Each kernel runs in a disposable worker
subprocess so a GPU fault takes down only one worker.

Stream-K: a sweep config with a ``streamk_config`` block expands to Stream-K
kernels, which fold the batch into their tile space (StreamKKernel); configs of
both kinds can be mixed in one run. Each Stream-K kernel x problem runs once per
``--budget-pct`` workgroup budget; the other kernels run once.

Examples:
    python batched_gemm_full_benchmark.py
    python batched_gemm_full_benchmark.py \
        batched_gemm/configs/default_config.json --devices 4 --csv out.csv

    # gfx950: batched Stream-K (compv3/compv4/mem, atomic/linear/tree) next to
    # the batched kernels, Stream-K at the full and at half the workgroup budget:
    python batched_gemm_full_benchmark.py batched_gemm/configs/default_config.json \
        batched_gemm/configs/streamk_config.json --dtype bf16 --verify --budget-pct 100,50

    # gfx1250: batched comp_tdm / comp_tdm_v2 / comp_tdm_v3 / comp_tdm_pc kernels and
    # Stream-K on comp_tdm, comp_tdm_v2 and comp_tdm_v3 (add
    # streamk_config_gfx1250.json for Stream-K on compv3/compv4/mem):
    python batched_gemm_full_benchmark.py batched_gemm/configs/tdm_config_gfx1250.json \
        batched_gemm/configs/streamk_tdm_config_gfx1250.json --arch gfx1250 --dtype bf16 --verify
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
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))
sys.path.insert(0, str(_COMMON_DIR))
sys.path.insert(0, str(_THIS_DIR))

from batched_gemm_utils import (  # noqa: E402
    BATCHED_SUPPORTED_DTYPES as SUPPORTED_DTYPES,
    BATCHED_SUPPORTED_LAYOUTS as SUPPORTED_LAYOUTS,
    BATCHED_VERIFY_TOL,
    config_variant,
    setup_multiple_batched_gemm_dispatchers,
    expand_sweep,
)
from gemm_utils import (  # noqa: E402
    STREAMK_BUDGET_ENV,
    add_streamk_args,
    streamk_budgets_arg,
    streamk_work_units,
    unique_by_name,
)
from smi_utils import detect_gpu_ids  # noqa: E402
from gemm_vector_fallback import VectorFallback, add_vector_fallback_arg  # noqa: E402

# The batched op keeps its sweep configs in tile_engine/ops/gemm/batched_gemm/configs.
CONFIG_DIR = _THIS_DIR / "batched_gemm" / "configs"
CI_CONFIG_NAME = "default_ci_config.json"
DEFAULT_CONFIG_NAME = "default_config.json"
EXAMPLE_PROBLEMS_NAME = "example_problems.json"

# Fallback batched problem set if none is supplied.
DEFAULT_PROBLEMS = [
    {"batch_count": 8, "M": 1024, "N": 1024, "K": 1024},
    {"batch_count": 4, "M": 2048, "N": 2048, "K": 2048},
    {"batch_count": 16, "M": 512, "N": 512, "K": 512},
    {"batch_count": 2, "M": 3840, "N": 4096, "K": 2048},
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
    return [str(CONFIG_DIR / CI_CONFIG_NAME)]


def load_problems(path):
    if path:
        with open(path) as f:
            data = json.load(f)
        return data["problems"] if isinstance(data, dict) else data
    example = CONFIG_DIR / EXAMPLE_PROBLEMS_NAME
    if example.exists():
        with open(example) as f:
            data = json.load(f)
        return data["problems"] if isinstance(data, dict) else data
    return DEFAULT_PROBLEMS


def _run_batch_on_device(device_id, unit, args, worker_path, base_env):
    """Run one (problem, kernel-batch, Stream-K budget) unit in a device-pinned
    subprocess; the budget is None for a batch of non-Stream-K kernels."""
    prob_idx, prob_dict, batch, budget_pct = unit
    bc, M, N, K = (
        prob_dict["batch_count"],
        prob_dict["M"],
        prob_dict["N"],
        prob_dict["K"],
    )

    items = [
        {
            "so_path": str(lib),
            "problem": prob_dict,
            "kernel_name": cfg.name,
            "arch": cfg.gfx_arch,
        }
        for _, cfg, lib in batch
    ]
    payload = json.dumps(
        {"items": items, "verify": args.verify, "verify_tol": args.verify_tol}
    )

    env = base_env.copy()
    env["HIP_VISIBLE_DEVICES"] = str(device_id)
    if budget_pct is not None:
        env[STREAMK_BUDGET_ENV] = str(budget_pct)
    budget_str = f"{budget_pct:>4}%" if budget_pct is not None else " " * 5

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
                extra = (
                    f" rel={result['max_rel']:.2e}" if "max_rel" in result else ""
                )
                lines.append(
                    f"  [gpu{device_id}] {cfg.name:<62} {budget_str} {result['ms']:>10.3f} "
                    f"{result['tflops']:>10.2f} {status:>8}{extra}"
                )
                rows.append(
                    {
                        "kernel": cfg.name,
                        "problem_idx": prob_idx,
                        "batch_count": bc,
                        "M": M,
                        "N": N,
                        "K": K,
                        "device": device_id,
                        "budget_pct": "" if budget_pct is None else budget_pct,
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
                lines.append(f"  [gpu{device_id}] {cfg.name:<62} FAILED")
                lines.append(f"    Error: {result.get('error', 'unknown')[:100]}")
                n_fail += 1

        missing = set(range(len(batch))) - reported
        if missing or proc.returncode != 0:
            if proc.returncode != 0:
                lines.append(
                    f"  [gpu{device_id}] worker exited code {proc.returncode}"
                )
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
    parser = argparse.ArgumentParser(
        description="Batched GEMM Benchmark Sweep (via Dispatcher)"
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="TE sweep config JSON files (default: batched_gemm/configs/default_ci_config.json; "
        "on gfx1250/MI400 pass batched_gemm/configs/default_ci_config_gfx1250.json for WMMA 16x16x32)",
    )
    # Default None so the bridge utilities auto-detect the actual GPU arch via
    # rocminfo (_resolve_arch); never hardcode gfx942 -- that would build an
    # incompatible kernel on gfx90a/gfx950/gfx1250 and launch it on the visible device.
    parser.add_argument(
        "--arch",
        default=None,
        help="GPU arch (gfx90a/gfx942/gfx950/gfx1250); default: auto-detect via rocminfo.",
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
    parser.add_argument(
        "--problems", default=None, help="JSON file of batched problems"
    )
    parser.add_argument("--csv", type=str, default="batched_gemm_results.csv")
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel build workers"
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="GPUs to use: int count or comma-list of ids; default auto-detect",
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
        "--max-kernels",
        type=int,
        default=0,
        help="Limit to first N kernels plus their vector-width variants (0=all)",
    )
    add_streamk_args(parser)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check each kernel's output against an fp32 numpy reference "
        "(per-batch A @ B); a mismatch counts as a failure",
    )
    parser.add_argument(
        "--verify-tol",
        type=float,
        default=None,
        help="Relative tolerance for --verify (default per dtype: "
        + ", ".join(f"{d} {t:g}" for d, t in BATCHED_VERIFY_TOL.items())
        + ")",
    )
    add_vector_fallback_arg(parser)
    args = parser.parse_args()
    if args.verify_tol is None:
        args.verify_tol = BATCHED_VERIFY_TOL[args.dtype]

    # --batch-size is the step of range(0, len(built_kernels), args.batch_size);
    # a zero step raises ValueError and a negative one silently yields no batches,
    # so reject non-positive values up front with a clear message.
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer (kernels per subprocess)")
    budget_pcts = streamk_budgets_arg(parser, args)

    config_paths = resolve_configs(args)
    devices = resolve_devices(args.devices)

    # ========================================================================
    # Phase 1: Compile kernels (parallel, no GPU)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile batched kernels")
    print(f"{'=' * 80}")
    print(f"  Configs: {', '.join(config_paths)}")

    problems = load_problems(args.problems)
    vfb = VectorFallback(
        problems, args.layout, args.dtype, "batched", args.no_vector_fallback,
        args.tune_c_vector_width,
    )

    all_configs = []
    for cfg_path in config_paths:
        streamk = config_variant(cfg_path) == "stream_k"
        all_configs.extend(
            expand_sweep(
                cfg_path,
                args.arch,
                dtype=args.dtype,
                layout=args.layout,
                reduction_strategies=(
                    [args.reduction_strategy]
                    if streamk and args.reduction_strategy
                    else None
                ),
                **vfb.expand_kwargs,
            )
        )

    vfb.report_rejects()

    # Positional configs can share kernels: dedupe so --max-kernels counts
    # distinct ones.
    all_configs = vfb.limit_base_kernels(unique_by_name(all_configs), args.max_kernels)

    print(f"  Expanded configs: {len(all_configs)}")
    print(f"  Build workers: {args.workers}")

    t0 = time.perf_counter()
    lib_paths = setup_multiple_batched_gemm_dispatchers(
        all_configs, verbose=True, max_workers=args.workers
    )
    build_time = time.perf_counter() - t0

    built_kernels = [
        (cfg, lib) for cfg, lib in zip(all_configs, lib_paths) if lib is not None
    ]
    vfb.report_builds(all_configs, lib_paths)

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

    pairs = vfb.pairs(problems, built_kernels)

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
        "batch_count",
        "M",
        "N",
        "K",
        "device",
        "budget_pct",
        "latency_ms",
        "tflops",
        "non_zero",
        "max_rel",
        "verified",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    worker_path = _THIS_DIR / "run_one_batched_gemm_kernel.py"
    base_env = os.environ.copy()
    base_env["GEMM_PYPATH"] = os.pathsep.join(
        [str(_DISPATCHER_ROOT / "python"), str(_THIS_DIR)]
    )

    work_q = queue.Queue()
    for unit in streamk_work_units(
        problems, pairs, built_kernels, args.batch_size, budget_pcts,
        ("batch_count", "M", "N", "K"),
    ):
        work_q.put(unit)

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
    return 1 if stats["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
