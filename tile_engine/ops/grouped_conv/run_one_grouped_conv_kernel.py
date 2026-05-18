#!/usr/bin/env python3
"""Worker script for running grouped conv kernels in isolated subprocess.

This mirrors FMHA's run_one_kernel.py design:
- Receives kernel config + problem via stdin as JSON
- Loads .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (flushed per-kernel)
- GPU fault kills only this process, parent can continue

Input JSON format:
    Single: {"so_path": "...", "problem": {...}, "kernel_name": "..."}
    Batch:  {"items": [{"so_path": "...", "problem": {...}, "kernel_name": "..."}, ...]}

Output JSON format (one line per kernel):
    {"idx": 0, "ok": true, "ms": 0.123, "tflops": 456.7}
    {"idx": 1, "ok": false, "error": "..."}
"""

import json
import os
import sys

# Add dispatcher python paths from environment (can be multiple paths separated by os.pathsep)
gconv_pypath = os.environ.get("GCONV_PYPATH", "")
if gconv_pypath:
    for p in gconv_pypath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)

from grouped_conv_utils import GroupedConvProblem, GpuGroupedConvRunner  # noqa: E402
import numpy as np  # noqa: E402


def _run_one(idx, so_path, prob_dict, kernel_name):
    """Run a single kernel and output result as JSON."""
    try:
        # Create problem from dict (include dilation and 3D if present)
        problem = GroupedConvProblem(
            N=prob_dict["N"],
            C=prob_dict["C"],
            K=prob_dict["K"],
            G=prob_dict["G"],
            Di=prob_dict.get("Di", 1),
            Hi=prob_dict["Hi"],
            Wi=prob_dict["Wi"],
            Z=prob_dict.get("Z", 1),
            Y=prob_dict["Y"],
            X=prob_dict["X"],
            stride_d=prob_dict.get("stride_d", 1),
            stride_h=prob_dict["stride_h"],
            stride_w=prob_dict["stride_w"],
            pad_d=prob_dict.get("pad_d", 0),
            pad_h=prob_dict["pad_h"],
            pad_w=prob_dict["pad_w"],
            dilation_d=prob_dict.get("dilation_d", 1),
            dilation_h=prob_dict.get("dilation_h", 1),
            dilation_w=prob_dict.get("dilation_w", 1),
            direction=prob_dict["direction"],
        )

        # Generate input/weight data based on direction using shape helpers
        # Direction determines what input_np and weight_np represent:
        #   forward:    input_np=X,    weight_np=W
        #   bwd_data:   input_np=dY,   weight_np=W
        #   bwd_weight: input_np=X,    weight_np=dY
        np.random.seed(42)
        if problem.direction == "bwd_data":
            # Runner expects (dY, W) for bwd_data
            input_shape = problem.output_shape()  # dY shape
            weight_shape = problem.weight_shape()  # W shape
        elif problem.direction == "bwd_weight":
            # Runner expects (X, dY) for bwd_weight
            input_shape = problem.input_shape()  # X shape
            weight_shape = problem.output_shape()  # dY shape
        else:  # forward
            # Runner expects (X, W) for forward
            input_shape = problem.input_shape()  # X shape
            weight_shape = problem.weight_shape()  # W shape

        input_data = (np.random.randn(*input_shape) * 0.1).astype(np.float16)
        weight_data = (np.random.randn(*weight_shape) * 0.1).astype(np.float16)

        # CRITICAL: Load library ONLY inside this subprocess
        runner = GpuGroupedConvRunner(lib_path=so_path)
        result = runner.run(input_data, weight_data, problem)

        if result.success:
            non_zero = (
                int(np.count_nonzero(result.output)) if result.output is not None else 0
            )
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "ok": True,
                        "ms": result.time_ms,
                        "tflops": result.tflops,
                        "non_zero": non_zero,
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
                        "error": result.error,
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

    if "items" in d:
        # Batch mode: run multiple kernels in this one subprocess
        for i, item in enumerate(d["items"]):
            _run_one(
                i, item["so_path"], item["problem"], item.get("kernel_name", "unknown")
            )
    else:
        # Single mode
        _run_one(0, d["so_path"], d["problem"], d.get("kernel_name", "unknown"))


if __name__ == "__main__":
    main()
