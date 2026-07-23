#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Worker script for running Multi-D GEMM kernels in an isolated subprocess.

Multi-D is a single-problem GEMM that fuses ``num_d`` extra D operands into the
epilogue: ``E = elementwise_op(A @ B, D0, D1, ...)``. It has its own C ABI
(``dispatcher_run_multi_d_gemm``) and runner (``GpuMultiDGemmRunner``); the
Multi-D specifics live entirely in the force-included kernel and
``multi_d_gemm_ctypes_lib.cpp``.

- Receives kernel config + problem via stdin as JSON
- Loads the .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (one line per kernel, flushed)
- A GPU fault kills only this process; the parent driver can continue

Input JSON format (only these fields are consumed):
    Single: {"so_path": "...", "problem": {"M":.., "N":.., "K":..},
             "kernel_name": "..."}
    Batch:  {"items": [ {single-item fields}, ... ]}

The D-tensor count is read off the .so (``dispatcher_get_num_d_tensors``) and the
element-wise op is parsed from ``kernel_name`` (``..._multid_<op>_d<num_d>``), so
no ``num_d`` / ``elementwise_op`` need be supplied -- any such keys are ignored.

Optional top-level keys ``verify`` (bool) and ``verify_tol`` (float) enable an
fp32 numpy reference check; when set, each OK result also carries ``verified``
and ``max_rel``.

Output JSON format (one line per kernel):
    {"idx": 0, "ok": true, "ms": 0.123, "tflops": 456.7, "non_zero": 1, "kernel": "..."}
    {"idx": 0, "ok": true, ..., "verified": true, "max_rel": 1.1e-3}   # with --verify
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

from gemm_utils import (  # noqa: E402
    MultiDGemmProblem,
    GpuMultiDGemmRunner,
)
import numpy as np  # noqa: E402


def _elementwise_op_from_kernel_name(name: str) -> str:
    """Extract the elementwise op token from a multi_d kernel name.

    Name ends with ``..._multid_<op>_d<num_d>`` (op = MultiDAdd / MultiDMultiply
    / PassThrough). Falls back to MultiDAdd.
    """
    parts = name.split("_")
    if "multid" in parts:
        i = parts.index("multid")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "MultiDAdd"


def _reference(op: str, ab: np.ndarray, Ds) -> np.ndarray:
    """fp32 reference of the fused multi_d epilogue: E = op(A@B, D0, D1, ...).

    MultiDAdd sums all D tensors onto the GEMM; MultiDMultiply multiplies them
    in; PassThrough ignores D. Matches ck_tile::reference_gemm_multiple_d.
    """
    out = ab.astype(np.float32).copy()
    if op == "MultiDMultiply":
        for d in Ds:
            out = out * d.astype(np.float32)
    elif op == "PassThrough":
        pass
    else:  # MultiDAdd (default)
        for d in Ds:
            out = out + d.astype(np.float32)
    return out


def _run_one(idx, so_path, prob_dict, kernel_name, verify=False, verify_tol=2e-2):
    """Run a single multi_d kernel and emit its result as one JSON line."""
    try:
        M = int(prob_dict["M"])
        N = int(prob_dict["N"])
        K = int(prob_dict["K"])

        # CRITICAL: load the library ONLY inside this subprocess. The runner reads
        # dtype + layout off the kernel name and the D-tensor count off the .so.
        runner = GpuMultiDGemmRunner(lib_path=so_path)
        num_d = runner.num_d_tensors  # authoritative: baked into the kernel
        problem = MultiDGemmProblem(M=M, N=N, K=K, num_d=num_d)

        # Cache host matrices so batch mode doesn't regenerate huge inputs per
        # kernel: A/B keyed by shape, Ds by (shape, num_d).
        cache = getattr(_run_one, "_abd_cache", {})
        key = (M, N, K, num_d)
        if key not in cache:
            rng = np.random.RandomState(42)
            A = (rng.randn(M, K) * 0.1).astype(np.float32)
            B = (rng.randn(K, N) * 0.1).astype(np.float32)
            Ds = [(rng.randn(M, N) * 0.1).astype(np.float32) for _ in range(num_d)]
            cache[key] = (A, B, Ds)
            _run_one._abd_cache = cache
        A, B, Ds = cache[key]

        result = runner.run(A, B, Ds, problem)

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
                # fp16 inputs: quantize A/B/D exactly as the device sees them so
                # the metric isolates compute error from input quantization.
                op = _elementwise_op_from_kernel_name(runner.kernel_name)
                Aq = A.astype(np.float16).astype(np.float32)
                Bq = B.astype(np.float16).astype(np.float32)
                Dq = [d.astype(np.float16).astype(np.float32) for d in Ds]
                ref = _reference(op, Aq @ Bq, Dq)
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
