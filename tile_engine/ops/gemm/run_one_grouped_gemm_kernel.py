#!/usr/bin/env python3
"""Worker script for running GROUPED GEMM kernels in an isolated subprocess.

Mirrors run_one_gemm_kernel.py, but for the grouped variant:
- Receives a grouped kernel config + grouped problem via stdin as JSON
- Loads the .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (one line per kernel, flushed)
- A GPU fault kills only this process; the parent driver can continue

A grouped problem is a LIST of (M, N, K) sub-problems run by one grouped launch.

Input JSON format (dtype/layout default to fp16/rcr if absent):
    Single: {"so_path": "...", "problem": {"groups": [[M,N,K], ...]},
             "kernel_name": "...", "dtype": "fp16", "layout": "rcr"}
    Batch:  {"items": [{"so_path": "...", "problem": {...}, "kernel_name": "...",
             "dtype": "...", "layout": "..."}, ...]}

Output JSON format (one line per kernel):
    {"idx": 0, "ok": true, "ms": 0.123, "tflops": 456.7, "non_zero": 1,
     "group_count": 8, "kernel": "..."}
    {"idx": 1, "ok": false, "error": "...", "kernel": "..."}
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

from gemm_utils import GroupedGemmProblem, GpuGroupedGemmRunner  # noqa: E402
import numpy as np  # noqa: E402


def _run_one(idx, so_path, prob_dict, kernel_name, dtype="fp16", layout="rcr"):
    """Run a single grouped kernel and emit its result as one JSON line."""
    try:
        problem = GroupedGemmProblem.from_dict(prob_dict)

        # Operands are generated as fp32-ish floats; the runner casts them to the
        # per-dtype codec (fp16/bf16/fp8/bf8) and applies layout transposes.
        np.random.seed(42)
        A_list = []
        B_list = []
        for (M, N, K) in problem.groups:
            A_list.append((np.random.randn(M, K) * 0.1).astype(np.float32))
            B_list.append((np.random.randn(K, N) * 0.1).astype(np.float32))

        # CRITICAL: load the library ONLY inside this subprocess.
        runner = GpuGroupedGemmRunner(lib_path=so_path, dtype=dtype, layout=layout)
        result = runner.run(A_list, B_list, problem)

        if result.success:
            non_zero = sum(
                int(np.count_nonzero(o)) for o in result.outputs if o is not None
            )
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "ok": True,
                        "ms": result.time_ms,
                        "tflops": result.tflops,
                        "non_zero": 1 if non_zero > 0 else 0,
                        "group_count": problem.group_count,
                        "kernel": kernel_name,
                    }
                ),
                flush=True,
            )
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
    """Read JSON from stdin, run grouped kernel(s), output results."""
    try:
        d = json.loads(sys.stdin.buffer.read())
    except Exception as e:
        print(
            json.dumps({"idx": 0, "ok": False, "error": f"JSON parse error: {e}"}),
            flush=True,
        )
        sys.exit(1)

    if "items" in d:
        for i, item in enumerate(d["items"]):
            _run_one(
                i,
                item["so_path"],
                item["problem"],
                item.get("kernel_name", "unknown"),
                item.get("dtype", "fp16"),
                item.get("layout", "rcr"),
            )
    else:
        _run_one(
            0,
            d["so_path"],
            d["problem"],
            d.get("kernel_name", "unknown"),
            d.get("dtype", "fp16"),
            d.get("layout", "rcr"),
        )


if __name__ == "__main__":
    main()
