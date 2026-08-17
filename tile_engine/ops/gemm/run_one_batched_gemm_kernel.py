#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Worker script for running BATCHED GEMM kernels in an isolated subprocess.

Batched counterpart of run_one_gemm_kernel.py:
- Receives kernel config + batched problem via stdin as JSON
- Loads the .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (one line per kernel, flushed)
- A GPU fault kills only this process; the parent driver can continue

Input JSON format:
    Single: {"so_path": "...",
             "problem": {"batch_count":.., "M":.., "N":.., "K":..},
             "kernel_name": "..."}
    Batch:  {"items": [{...}, ...]}

Optional top-level keys ``verify`` (bool) and ``verify_tol`` (float) enable an
fp32 numpy reference check (per-batch A @ B); when set, each OK result also
carries ``verified`` and ``max_rel``.
"""

import json
import os
import sys

# Add dispatcher python paths from environment (os.pathsep-separated).
gemm_pypath = os.environ.get("GEMM_PYPATH", "")
if gemm_pypath:
    for p in gemm_pypath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)

from batched_gemm_utils import (  # noqa: E402
    BatchedGemmProblem,
    GpuBatchedGemmRunner,
)
import numpy as np  # noqa: E402


def _run_one(idx, so_path, prob_dict, kernel_name, verify=False, verify_tol=2e-2):
    """Run a single batched kernel and emit its result as one JSON line.

    When ``verify`` is set, the batched output is checked against an fp32 numpy
    reference (per-batch ``A @ B``) using the global relative metric
    ``max|out - ref| / max|ref|``.
    """
    try:
        problem = BatchedGemmProblem.from_dict(prob_dict)

        # Cache host tensors per (batch, M, N, K) so batch mode doesn't
        # regenerate large inputs per kernel.
        cache = getattr(_run_one, "_ab_cache", {})
        key = (problem.batch_count, problem.M, problem.N, problem.K)
        if key not in cache:
            rng = np.random.RandomState(42)
            cache[key] = (
                (rng.randn(problem.batch_count, problem.M, problem.K) * 0.1).astype(
                    np.float32
                ),
                (rng.randn(problem.batch_count, problem.K, problem.N) * 0.1).astype(
                    np.float32
                ),
            )
            _run_one._ab_cache = cache
        A, B = cache[key]

        # CRITICAL: load the library ONLY inside this subprocess.
        runner = GpuBatchedGemmRunner(lib_path=so_path)
        result = runner.run(A, B, problem)

        if result.success:
            non_zero = (
                int(np.count_nonzero(result.output))
                if result.output is not None
                else 0
            )
            out = {
                "idx": idx,
                "ok": True,
                "ms": result.time_ms,
                "tflops": result.tflops,
                "non_zero": non_zero,
                "kernel": kernel_name,
            }
            if verify:
                ref = np.matmul(A.astype(np.float32), B.astype(np.float32))
                got = result.output.astype(np.float32)
                denom = float(np.max(np.abs(ref))) or 1.0
                max_rel = float(np.max(np.abs(got - ref)) / denom)
                out["max_rel"] = max_rel
                out["verified"] = bool(max_rel <= verify_tol)
            print(json.dumps(out), flush=True)
        else:
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "ok": False,
                        "error": f"kernel returned status {result.status}",
                        "kernel": kernel_name,
                    }
                ),
                flush=True,
            )

    except Exception as e:
        print(
            json.dumps(
                {"idx": idx, "ok": False, "error": str(e), "kernel": kernel_name}
            ),
            flush=True,
        )


def main():
    """Read JSON from stdin, run kernel(s), output results."""
    try:
        d = json.loads(sys.stdin.buffer.read())
    except Exception as e:
        print(
            json.dumps({"idx": 0, "ok": False, "error": f"JSON parse error: {e}"}),
            flush=True,
        )
        sys.exit(1)

    verify = bool(d.get("verify", False))
    verify_tol = float(d.get("verify_tol", 2e-2))

    if "items" in d:
        for i, item in enumerate(d["items"]):
            _run_one(
                i,
                item["so_path"],
                item["problem"],
                item.get("kernel_name", "unknown"),
                verify=verify,
                verify_tol=verify_tol,
            )
    else:
        _run_one(
            0,
            d["so_path"],
            d["problem"],
            d.get("kernel_name", "unknown"),
            verify=verify,
            verify_tol=verify_tol,
        )


if __name__ == "__main__":
    main()
