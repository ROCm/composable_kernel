#!/usr/bin/env python3
"""Full GROUPED GEMM benchmark sweep driven through the Dispatcher bridge.

The grouped counterpart of gemm_full_benchmark.py. Same 3-phase architecture:
  Phase 1: Compile all grouped kernels (parallel, returns .so paths only -- no GPU)
  Phase 2: Load grouped problems (lists of (M, N, K) sub-problems)
  Phase 3: Benchmark via subprocess isolation (serial GPU, batched)

Tile Engine generates NO binaries here: it expands its grouped sweep config into
shared ``GemmKernelConfig(variant="grouped")`` objects and hands them to the
dispatcher, which codegens + compiles each into a grouped .so (built against
``grouped_gemm_ctypes_lib.cpp``). Each kernel runs in a disposable worker
subprocess so a GPU fault takes down only one worker.

Usage:
    python grouped_gemm_full_benchmark.py grouped_gemm/configs/default_config.json \
        --arch gfx942 --csv grouped_gemm_results.csv

A grouped "problem" is a batch of sub-problems run by one launch. The default set
uses uniform shapes at two group counts; override with --problems pointing at a
JSON file of [{"groups": [[M,N,K], ...]}, ...].
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

from gemm_utils import setup_multiple_gemm_dispatchers, expand_sweep  # noqa: E402

# Default grouped problem set: uniform groups at two batch sizes (Phase 3 parity).
DEFAULT_PROBLEMS = [
    {"groups": [[1024, 1024, 1024]] * 4},
    {"groups": [[2048, 2048, 2048]] * 4},
    {"groups": [[1024, 1024, 1024]] * 8},
    {"groups": [[512, 1024, 2048], [1024, 512, 1024], [2048, 2048, 512], [256, 768, 1024]]},
]


def load_problems(path):
    if not path:
        return DEFAULT_PROBLEMS
    with open(path) as f:
        data = json.load(f)
    # Accept either a bare list or {"problems": [...]}.
    return data["problems"] if isinstance(data, dict) else data


def _problem_dims(prob):
    """Aggregate (group_count, total_flops) for reporting."""
    groups = prob["groups"]
    flops = sum(2.0 * m * n * k for (m, n, k) in groups)
    return len(groups), flops


def main():
    parser = argparse.ArgumentParser(
        description="Grouped GEMM Benchmark Sweep (via Dispatcher)"
    )
    parser.add_argument("configs", nargs="+", help="TE grouped sweep config JSON files")
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--layout", default="rcr")
    parser.add_argument(
        "--problems", default=None, help="JSON file of grouped problems"
    )
    parser.add_argument("--csv", type=str, default="grouped_gemm_results.csv")
    parser.add_argument("--workers", type=int, default=8, help="Parallel build workers")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Kernels per subprocess (overhead vs fault isolation)",
    )
    parser.add_argument(
        "--kernel-timeout", type=int, default=60, help="Per-kernel timeout (s)"
    )
    parser.add_argument(
        "--max-kernels", type=int, default=0, help="Limit to first N kernels (0=all)"
    )
    args = parser.parse_args()

    # ========================================================================
    # Phase 1: Compile kernels (parallel, no GPU)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile grouped kernels")
    print(f"{'=' * 80}")

    all_configs = []
    for cfg_path in args.configs:
        all_configs.extend(
            expand_sweep(
                cfg_path,
                args.arch,
                dtype=args.dtype,
                layout=args.layout,
                variant="grouped",
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
    print("Phase 2: Load grouped test problems")
    print(f"{'=' * 80}")

    problems = load_problems(args.problems)
    print(f"  Problems: {len(problems)}")
    print(
        f"  Total measurements: {len(built_kernels)} x {len(problems)} = "
        f"{len(built_kernels) * len(problems)}"
    )

    # ========================================================================
    # Phase 3: Benchmark via subprocess (serial GPU, batched)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 3: Benchmark (subprocess isolation, batched)")
    print(f"{'=' * 80}")
    print(f"  Batch size: {args.batch_size} kernels per subprocess")
    print(f"  Timeout: {args.kernel_timeout}s per kernel\n")

    csv_path = Path(args.csv)
    csv_fields = [
        "kernel",
        "problem_idx",
        "group_count",
        "total_flops",
        "latency_ms",
        "tflops",
        "non_zero",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    worker_path = _THIS_DIR / "run_one_grouped_gemm_kernel.py"
    worker_env = os.environ.copy()
    worker_env["GEMM_PYPATH"] = os.pathsep.join(
        [str(_DISPATCHER_ROOT / "python"), str(_THIS_DIR)]
    )

    total_measurements = 0
    total_failures = 0
    bench_t0 = time.perf_counter()

    for prob_idx, prob in enumerate(problems):
        group_count, total_flops = _problem_dims(prob)
        print(
            f"\nProblem [{prob_idx + 1}/{len(problems)}]: groups={group_count} "
            f"flops={total_flops / 1e9:.1f}G ({len(built_kernels)} kernels)"
        )
        print(f"  {'Kernel':<60} {'Time(ms)':>10} {'TFLOPS':>10} {'Status':>10}")
        print(f"  {'-' * 95}")

        prob_dict = {"groups": [list(g) for g in prob["groups"]]}

        for batch_start in range(0, len(built_kernels), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(built_kernels))
            batch = built_kernels[batch_start:batch_end]

            items = [
                {
                    "so_path": str(lib_path),
                    "problem": prob_dict,
                    "kernel_name": cfg.name,
                    "dtype": args.dtype,
                    "layout": args.layout,
                }
                for cfg, lib_path in batch
            ]
            payload = json.dumps({"items": items})

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

                reported_indices = set()
                for line in stdout_bytes.decode("utf-8").strip().split("\n"):
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                        batch_idx = result.get("idx")
                        if not isinstance(batch_idx, int) or not (
                            0 <= batch_idx < len(batch)
                        ):
                            print(
                                f"  [warn] worker returned out-of-range idx "
                                f"{batch_idx!r}; skipping line"
                            )
                            continue
                        cfg, lib_path = batch[batch_idx]
                        reported_indices.add(batch_idx)

                        if result.get("ok", False):
                            status = "OK" if result.get("non_zero", 0) > 0 else "ZERO"
                            print(
                                f"  {cfg.name:<60} {result['ms']:>10.3f} "
                                f"{result['tflops']:>10.2f} {status:>10}"
                            )
                            writer.writerow(
                                {
                                    "kernel": cfg.name,
                                    "problem_idx": prob_idx,
                                    "group_count": group_count,
                                    "total_flops": total_flops,
                                    "latency_ms": result["ms"],
                                    "tflops": result["tflops"],
                                    "non_zero": result.get("non_zero", 0),
                                }
                            )
                            csv_file.flush()
                            total_measurements += 1
                        else:
                            print(f"  {cfg.name:<60} FAILED")
                            print(f"    Error: {result.get('error', 'unknown')[:100]}")
                            total_failures += 1
                    except json.JSONDecodeError:
                        print(f"  Warning: could not parse result line: {line[:50]}")
                        total_failures += 1

                missing_indices = set(range(len(batch))) - reported_indices
                if missing_indices or proc.returncode != 0:
                    if proc.returncode != 0:
                        print(f"  Worker exited with code {proc.returncode}")
                    for idx in sorted(missing_indices):
                        cfg, _ = batch[idx]
                        print(f"  {cfg.name:<60} MISSING (worker crash)")
                    total_failures += len(missing_indices)

            except subprocess.TimeoutExpired:
                print(f"  Batch timeout ({len(batch)} kernels)")
                try:
                    proc.kill()
                    proc.communicate(timeout=5)
                except Exception:
                    pass
                total_failures += len(batch)
            except Exception as e:
                print(f"  Batch error: {e}")
                try:
                    if proc and proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
                total_failures += len(batch)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
