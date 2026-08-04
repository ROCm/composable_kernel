#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Full Multi-D GEMM benchmark sweep driven through the Dispatcher bridge.

Same 3-phase architecture as ``gemm_full_benchmark.py``. Multi-D keeps the
single-problem *shape* (one A/B/C, one M/N/K) but uses a *dedicated* C ABI --
its own entry point ``dispatcher_run_multi_d_gemm`` in
``multi_d_gemm_ctypes_lib.cpp`` (which also takes the D-pointer array + count),
not the regular ``dispatcher_run_gemm``. The bridge difference is
``variant="multi_d"`` threaded into ``expand_sweep`` (which makes the dispatcher
codegen the Multi-D launch and ``gemm_utils`` select that dedicated ctypes
source). The GPU worker uses ``MultiDGemmProblem`` / ``GpuMultiDGemmRunner`` for
the fused-D epilogue.

  Phase 1: Compile all kernels (parallel, returns .so paths only -- no GPU)
  Phase 2: Load problems (M, N, K shapes)
  Phase 3: Benchmark via subprocess isolation, distributed across all visible
           GPUs (one device-pinned worker per GPU, batched, fault-isolated)

Like the regular bridge driver, Phase 3 fans the (kernel x problem) work out
across every visible GPU in parallel: each device runs its own stream of
disposable worker subprocesses pinned with ``HIP_VISIBLE_DEVICES``, so an N-GPU
box benchmarks roughly N times faster while keeping per-batch fault isolation.

Examples:
    # Default config (gemm_multi_d/configs/default_config.json), all visible GPUs:
    python gemm_multi_d_full_benchmark.py

    # Explicit config on 4 GPUs with correctness checking:
    python gemm_multi_d_full_benchmark.py gemm_multi_d/configs/default_config.json \
        --devices 4 --verify --csv gemm_multi_d_results.csv
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

# Multi-D is a single variant; its sweep configs live in gemm_multi_d/configs/.
DEFAULT_CONFIG = _THIS_DIR / "gemm_multi_d" / "configs" / "default_config.json"

# Default problem set: squares plus a large-K skinny shape -- Multi-D's sweet
# spot is few output tiles with a long K reduction. Tiny problems (e.g. 257^3)
# have too few tiles to partition across CUs and the kernel reports them as
# unsupported (status -2), which the bridge surfaces gracefully.
DEFAULT_PROBLEMS = [
    {"M": 1024, "N": 1024, "K": 1024},
    {"M": 2048, "N": 2048, "K": 2048},
    {"M": 4096, "N": 4096, "K": 4096},
    {"M": 512, "N": 512, "K": 8192},
]

# Bridge surface for Multi-D. The dispatcher host path
# (multi_d_gemm_ctypes_lib.cpp) derives strides from the kernel's layouts and the
# worker (run_one_gemm_multi_d_kernel.py) reads dtype/layout off the kernel name,
# so all 4 A/B/C/D layouts are supported. dtype is fp16 only -- this matches
# Old-TE's Multi-D capability set (fp16 x {rcrr,rrrr,ccrr,crrr}); the bridge
# runner encodes fp16 natively and accumulates in fp32. Other dtypes (bf16 /
# fp8 / bf8 / int8) are intentionally out of scope for Multi-D parity.
SUPPORTED_DTYPES = ("fp16",)
SUPPORTED_LAYOUTS = ("rcrr", "rrrr", "ccrr", "crrr")


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
        return detected[:n] if len(detected) >= n else [str(i) for i in range(n)]
    return [spec]


def resolve_configs(args):
    """Resolve positional configs -> concrete list of config paths."""
    if args.configs:
        return args.configs
    return [str(DEFAULT_CONFIG)]


def load_problems(path):
    if not path:
        return DEFAULT_PROBLEMS
    with open(path) as f:
        data = json.load(f)
    # Accept either a bare list or {"problems": [...]}.
    return data["problems"] if isinstance(data, dict) else data


def _run_batch_on_device(device_id, unit, args, worker_path, base_env):
    """Run one (problem, kernel-batch) unit in a device-pinned subprocess.

    Returns (rows, lines, n_fail) where rows are dicts ready for the CSV writer,
    lines are formatted strings to print, and n_fail counts failures.
    """
    prob_idx, prob_dict, batch = unit
    M, N, K = prob_dict["M"], prob_dict["N"], prob_dict["K"]

    items = [
        {"so_path": str(lib), "problem": prob_dict, "kernel_name": cfg.name}
        for _, cfg, lib in batch
    ]
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
                lines.append(f"  [gpu{device_id}] Warning: bad result line: {line[:50]}")
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
        description="Multi-D GEMM Benchmark Sweep (via Dispatcher)"
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="TE sweep config JSON files (default: gemm_multi_d/configs/default_config.json)",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="GPU arch (gfx90a/gfx942/gfx950); default: auto-detect via rocminfo "
        "(no silent gfx942 default). Flows into expand_sweep, which resolves None.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=SUPPORTED_DTYPES,
        help=f"Input dtype (supported: {', '.join(SUPPORTED_DTYPES)})",
    )
    parser.add_argument(
        "--layout",
        default="rcrr",
        choices=SUPPORTED_LAYOUTS,
        help=f"A/B/C layout (supported: {', '.join(SUPPORTED_LAYOUTS)})",
    )
    parser.add_argument("--problems", default=None, help="JSON file of M,N,K problems")
    parser.add_argument("--csv", type=str, default="gemm_multi_d_results.csv")
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
        help="Relative tolerance for --verify (default 2e-2). MultiDMultiply "
        "products amplify fp16 rounding vs MultiDAdd, so the reported max_rel is "
        "larger for Multiply/num_d=2; measured combos stay well under this gate.",
    )
    args = parser.parse_args()

    config_paths = resolve_configs(args)
    devices = resolve_devices(args.devices)

    # ========================================================================
    # Phase 1: Compile kernels (parallel, no GPU)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile Multi-D kernels")
    print(f"{'=' * 80}")
    print(f"  Configs: {', '.join(config_paths)}")

    all_configs = []
    for cfg_path in config_paths:
        all_configs.extend(
            expand_sweep(
                cfg_path,
                args.arch,
                dtype=args.dtype,
                layout=args.layout,
                variant="multi_d",
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

    worker_path = _THIS_DIR / "run_one_gemm_multi_d_kernel.py"
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
