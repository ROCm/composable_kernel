#!/usr/bin/env python3
"""Full grouped convolution benchmark sweep.

Architecture mirrors FMHA's fmha_full_benchmark.py:
  Phase 1: Compile all kernels (parallel, returns .so paths only)
  Phase 2: Benchmark via subprocess isolation (serial GPU access)

Each kernel runs in a subprocess to avoid Python ctypes library loading limits.
Subprocess batching (default 20) balances overhead vs fault isolation.

Usage:
    python grouped_conv_full_benchmark.py configs/forward_2d.json --arch gfx950 \
        --problems forward_2d --csv results.csv

Available problem sets (one per variant x ndim, plus validation):
    - forward_2d, forward_3d
    - bwd_data_2d, bwd_data_3d
    - bwd_weight_2d, bwd_weight_3d
    - bwd_data_test_validation, bwd_weight_test_validation, validation_holdout
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_THIS_DIR))

from grouped_conv_utils import setup_multiple_grouped_conv_dispatchers  # noqa: E402
from grouped_conv_instance_builder import expand_sweep  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Grouped Conv Benchmark Sweep")
    parser.add_argument("configs", nargs="+", help="Config JSON files")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--problems", default="forward_2d")
    parser.add_argument("--csv", type=str, default="grouped_conv_results.csv")
    parser.add_argument("--workers", type=int, default=8, help="Parallel build workers")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Kernels per subprocess (balance overhead vs fault isolation)",
    )
    parser.add_argument(
        "--kernel-timeout",
        type=int,
        default=30,
        help="Per-kernel timeout in seconds",
    )
    parser.add_argument(
        "--max-kernels",
        type=int,
        default=0,
        help="Limit to first N kernels (0=all)",
    )
    args = parser.parse_args()

    # ========================================================================
    # Phase 1: Compile kernels (parallel)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile kernels")
    print(f"{'=' * 80}")

    all_configs = []
    for cfg_path in args.configs:
        all_configs.extend(expand_sweep(cfg_path, args.arch))

    if args.max_kernels > 0:
        all_configs = all_configs[: args.max_kernels]

    print(f"  Expanded configs: {len(all_configs)}")
    print(f"  Build workers: {args.workers}")

    t0 = time.perf_counter()
    # CRITICAL: This returns Path objects only, does NOT load .so files
    lib_paths = setup_multiple_grouped_conv_dispatchers(
        all_configs, verbose=True, max_workers=args.workers
    )
    build_time = time.perf_counter() - t0

    built_kernels = [
        (cfg, lib) for cfg, lib in zip(all_configs, lib_paths) if lib is not None
    ]

    # Deduplicate by library path - don't benchmark the same .so multiple times
    # This happens when multiple virtual configs (e.g., compv3/compv4/compv5) map to the same physical kernel
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

    sys.path.insert(0, str(_THIS_DIR / "problems"))

    # Map --problems value to (module, attribute) so the import is lazy
    # (avoids paying the cost of every problem set on every run).
    problem_sets = {
        # Training sets: one per (variant, ndim)
        "forward_2d":     ("forward_2d",     "PROBLEMS_FORWARD_2D"),
        "forward_3d":     ("forward_3d",     "PROBLEMS_FORWARD_3D"),
        "bwd_data_2d":    ("bwd_data_2d",    "PROBLEMS_BWD_DATA_2D"),
        "bwd_data_3d":    ("bwd_data_3d",    "PROBLEMS_BWD_DATA_3D"),
        "bwd_weight_2d":  ("bwd_weight_2d",  "PROBLEMS_BWD_WEIGHT_2D"),
        "bwd_weight_3d":  ("bwd_weight_3d",  "PROBLEMS_BWD_WEIGHT_3D"),
        # Validation sets
        "bwd_data_test_validation":   ("bwd_data_test_validation",   "VALIDATION_PROBLEMS_BWD_DATA"),
        "bwd_weight_test_validation": ("bwd_weight_test_validation", "VALIDATION_PROBLEMS_BWD_WEIGHT"),
        "validation_holdout":         ("validation_holdout",         "VALIDATION_PROBLEMS"),
    }

    if args.problems not in problem_sets:
        raise ValueError(
            f"Unknown problem set: {args.problems!r}. "
            f"Available: {sorted(problem_sets)}"
        )

    mod_name, attr = problem_sets[args.problems]
    problems = getattr(__import__(mod_name), attr)

    print(f"  Problems: {len(problems)}")
    print(
        f"  Total measurements: {len(built_kernels)} x {len(problems)} = {len(built_kernels) * len(problems)}"
    )

    # ========================================================================
    # Phase 3: Benchmark via subprocess (serial GPU, batched subprocess)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 3: Benchmark (subprocess isolation, batched)")
    print(f"{'=' * 80}")
    print(f"  Batch size: {args.batch_size} kernels per subprocess")
    print(f"  Timeout: {args.kernel_timeout}s per kernel")
    print()

    csv_path = Path(args.csv)
    csv_fields = [
        "kernel",
        "problem_idx",
        "N",
        "C",
        "K",
        "G",
        "Di",
        "Hi",
        "Wi",
        "Z",
        "Y",
        "X",
        "stride_d",
        "stride_h",
        "stride_w",
        "pad_d",
        "pad_h",
        "pad_w",
        "dilation_d",
        "dilation_h",
        "dilation_w",
        "latency_ms",
        "tflops",
        "non_zero",
    ]

    # Open CSV for writing
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    worker_path = _THIS_DIR / "run_one_grouped_conv_kernel.py"
    worker_env = os.environ.copy()
    # Worker needs both dispatcher/python (for dispatcher_common) and current dir (for grouped_conv_utils)
    worker_env["GCONV_PYPATH"] = os.pathsep.join(
        [str(_DISPATCHER_ROOT / "python"), str(_THIS_DIR)]
    )

    total_measurements = 0
    total_failures = 0
    bench_t0 = time.perf_counter()

    for prob_idx, prob in enumerate(problems):
        try:
            # All shape/ndim/feature support is enforced by the dispatcher.
            # Unsupported (kernel, problem) combinations must surface as loud
            # errors from the worker subprocess — do NOT pre-filter here.
            prob_Di = getattr(prob, "Di", 1)
            prob_Z = getattr(prob, "Z", 1)
            prob_ndim = 3 if (prob_Di > 1 or prob_Z > 1) else 2

            matching_kernels = built_kernels

            print(
                f"\nProblem [{prob_idx + 1}/{len(problems)}]: N={prob.N} C={prob.C} K={prob.K} H={prob.Hi} W={prob.Wi} (ndim={prob_ndim}D, {len(matching_kernels)} kernels)"
            )
            print(f"  {'Kernel':<60} {'Time(ms)':>10} {'TFLOPS':>10} {'Status':>10}")
            print(f"  {'-' * 95}")

            # Convert problem to dict once (with 3D support)
            prob_dict = {
            "N": prob.N,
            "C": prob.C,
            "K": prob.K,
            "G": prob.G,
            "Di": prob_Di,
            "Hi": prob.Hi,
            "Wi": prob.Wi,
            "Z": prob_Z,
            "Y": prob.Y,
            "X": prob.X,
            "stride_d": getattr(prob, "stride_d", 1),
            "stride_h": prob.stride_h,
            "stride_w": prob.stride_w,
            "pad_d": getattr(prob, "pad_d", 0),
            "pad_h": prob.pad_h,
            "pad_w": prob.pad_w,
            "dilation_d": getattr(prob, "dilation_d", 1),
            "dilation_h": getattr(prob, "dilation_h", 1),
            "dilation_w": getattr(prob, "dilation_w", 1),
            "direction": prob.direction,
            }

            # Process matching kernels in batches
            for batch_start in range(0, len(matching_kernels), args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(matching_kernels))
                batch = matching_kernels[batch_start:batch_end]

                # Build JSON payload for this batch
                items = []
                for cfg, lib_path in batch:
                    items.append(
                        {
                            "so_path": str(
                                lib_path
                            ),  # CRITICAL: Only pass string path, not loaded library
                            "problem": prob_dict,
                            "kernel_name": cfg.name,
                        }
                    )

                payload = json.dumps({"items": items})

                # Run subprocess with batch
                try:
                    proc = subprocess.Popen(
                        [sys.executable, str(worker_path)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        env=worker_env,
                    )
    
                    timeout_total = args.kernel_timeout * len(batch)
                    stdout_bytes, _ = proc.communicate(
                        input=payload.encode("utf-8"), timeout=timeout_total
                    )
    
                    # Track which batch indices were reported
                    reported_indices = set()
    
                    # Parse results (one JSON line per kernel)
                    for line in stdout_bytes.decode("utf-8").strip().split("\n"):
                        if not line:
                            continue
    
                        try:
                            result = json.loads(line)
                            batch_idx = result.get("idx", 0)
                            cfg, lib_path = batch[batch_idx]
                            reported_indices.add(batch_idx)
    
                            if result.get("ok", False):
                                status = "OK" if result.get("non_zero", 0) > 0 else "ZERO"
                                print(
                                    f"  {cfg.name:<60} {result['ms']:>10.3f} {result['tflops']:>10.2f} {status:>10}"
                                )
    
                                writer.writerow(
                                    {
                                        "kernel": cfg.name,
                                        "problem_idx": prob_idx,
                                        "N": prob.N,
                                        "C": prob.C,
                                        "K": prob.K,
                                        "G": prob.G,
                                        "Di": getattr(prob, "Di", 1),
                                        "Hi": prob.Hi,
                                        "Wi": prob.Wi,
                                        "Z": getattr(prob, "Z", 1),
                                        "Y": prob.Y,
                                        "X": prob.X,
                                        "stride_d": getattr(prob, "stride_d", 1),
                                        "stride_h": prob.stride_h,
                                        "stride_w": prob.stride_w,
                                        "pad_d": getattr(prob, "pad_d", 0),
                                        "pad_h": prob.pad_h,
                                        "pad_w": prob.pad_w,
                                        "dilation_d": getattr(prob, "dilation_d", 1),
                                        "dilation_h": getattr(prob, "dilation_h", 1),
                                        "dilation_w": getattr(prob, "dilation_w", 1),
                                        "latency_ms": result["ms"],
                                        "tflops": result["tflops"],
                                        "non_zero": result.get("non_zero", 0),
                                    }
                                )
                                csv_file.flush()
                                total_measurements += 1
                            else:
                                error_msg = result.get("error", "unknown")
                                # Show full error for debugging (first 100 chars)
                                print(f"  {cfg.name:<60} FAILED")
                                print(f"    Error: {error_msg[:100]}")
                                total_failures += 1
    
                        except json.JSONDecodeError:
                            print(f"  Warning: Could not parse result line: {line[:50]}")
                            total_failures += 1
    
                    # Check for missing results (worker crashed mid-batch or non-zero exit)
                    missing_indices = set(range(len(batch))) - reported_indices
                    if missing_indices or proc.returncode != 0:
                        if proc.returncode != 0:
                            print(f"  Worker exited with code {proc.returncode}")
                        if missing_indices:
                            print(f"  Missing results for {len(missing_indices)} kernel(s)")
                            for idx in sorted(missing_indices):
                                cfg, _ = batch[idx]
                                print(f"  {cfg.name:<60} MISSING (worker crash)")
                            total_failures += len(missing_indices)
    
                except subprocess.TimeoutExpired:
                    print(f"  Batch timeout after {args.kernel_timeout * len(batch)}s ({len(batch)} kernels)")
                    try:
                        proc.kill()
                        proc.communicate(timeout=5)
                    except:
                        pass
                    total_failures += len(batch)
                    # Log which kernels timed out
                    for idx, (cfg, _) in enumerate(batch):
                        print(f"    {cfg.name} - TIMEOUT")
    
                except Exception as e:
                    print(f"  Batch error: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        if proc and proc.poll() is None:
                            proc.kill()
                    except:
                        pass
                    total_failures += len(batch)
    
        except Exception as e:
            print(f"\n  PROBLEM ERROR: Problem {prob_idx} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Continuing to next problem...\n")
            # Count all kernels for this problem as failures
            if 'matching_kernels' in locals():
                total_failures += len(matching_kernels)

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
    print(f"  Successful measurements: {total_measurements}")
    print(f"  Failed measurements: {total_failures}")
    print(f"  Output: {csv_path}")


if __name__ == "__main__":
    main()
