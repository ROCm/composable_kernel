#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Worker script for running GEMM kernels in an isolated subprocess.

Mirrors grouped_conv's run_one_grouped_conv_kernel.py:
- Receives kernel config + problem via stdin as JSON
- Loads the .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (one line per kernel, flushed)
- A GPU fault kills only this process; the parent driver can continue

Input JSON format:
    Single: {"so_path": "...", "problem": {"M":.., "N":.., "K":..}, "kernel_name": "..."}
    Batch:  {"items": [{"so_path": "...", "problem": {...}, "kernel_name": "..."}, ...]}

Optional top-level keys ``verify`` (bool) and ``verify_tol`` (float) enable an
fp32 numpy reference check; when set, each OK result also carries ``verified``
and ``max_rel``.

Output JSON format (one line per kernel):
    {"idx": 0, "ok": true, "ms": 0.123, "tflops": 456.7, "non_zero": 1, "kernel": "..."}
    {"idx": 0, "ok": true, ..., "verified": true, "max_rel": 3.1e-4}   # with --verify
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

from gemm_utils import GemmProblem, GpuGemmRunner, GpuMultiABDRunner  # noqa: E402
import numpy as np  # noqa: E402


def _is_multi_abd(kernel_name):
    """A multi_abd kernel name carries the ``_multiabd_`` suffix (see codegen)."""
    return "_multiabd_" in (kernel_name or "")


def _run_one(
    idx,
    so_path,
    prob_dict,
    kernel_name,
    verify=False,
    verify_tol=2e-2,
    layout4=None,
    a_op=None,
    b_op=None,
    cde_op=None,
):
    """Run a single kernel and emit its result as one JSON line.

    When ``verify`` is set, the kernel output is checked against an fp32 numpy
    reference using the global relative metric ``max|out - ref| / max|ref|``;
    the emitted ``verified`` field then reflects correctness, not just liveness
    (``non_zero``).

    Standard path: reference is ``A @ B``. Multi-ABD path (B1): the reference is
    the full ``E = CDE( AB(As) @ BB(Bs), {Ds} )`` computed inside
    GpuMultiABDRunner (it owns the operands); the runner returns ``max_rel`` on
    the result. ``layout4`` and the per-group element-wise ops come from the
    config object (N2) so the runner never guesses the layout from the name.
    """
    try:
        problem = GemmProblem.from_dict(prob_dict)

        # Multi-ABD has a divergent (array-pointer) ABI and its own runner, which
        # generates its A/B/D operands internally (no shared A,B). Route by name.
        if _is_multi_abd(kernel_name):
            # CRITICAL: load the library ONLY inside this subprocess.
            runner = GpuMultiABDRunner(
                lib_path=so_path,
                layout4=layout4,
                a_elementwise_op=a_op,
                b_elementwise_op=b_op,
                cde_elementwise_op=cde_op,
            )
            # The multi_abd runner computes its own numpy reference (it owns the
            # operands), so verification flows through the runner, not the A@B
            # check below.
            result = runner.run(problem, verify=verify, verify_tol=verify_tol)
            A = B = None
        else:
            # Cache host matrices per shape so batch mode doesn't regenerate huge
            # inputs per kernel.
            cache = getattr(_run_one, "_ab_cache", {})
            key = (problem.M, problem.N, problem.K)
            if key not in cache:
                rng = np.random.RandomState(42)
                cache[key] = (
                    (rng.randn(problem.M, problem.K) * 0.1).astype(np.float32),
                    (rng.randn(problem.K, problem.N) * 0.1).astype(np.float32),
                )
                _run_one._ab_cache = cache
            A, B = cache[key]

            # CRITICAL: load the library ONLY inside this subprocess.
            runner = GpuGemmRunner(lib_path=so_path)
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
            if verify and _is_multi_abd(kernel_name):
                # Multi-ABD verifies against its own full multi-tensor reference
                # (E = CDE(AB(As) @ BB(Bs), {Ds})) computed inside the runner,
                # exercising the BRIDGE launch path (dispatcher_run_multi_abd ->
                # SelectedKernel::launch), not just the Old-TE --verify path.
                if result.max_rel is not None:
                    out["max_rel"] = result.max_rel
                    out["verified"] = bool(result.max_rel <= verify_tol)
            elif verify and A is not None and B is not None:
                ref = A.astype(np.float32) @ B.astype(np.float32)
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
                layout4=item.get("layout4"),
                a_op=item.get("a_elementwise_op"),
                b_op=item.get("b_elementwise_op"),
                cde_op=item.get("cde_elementwise_op"),
            )
    else:
        _run_one(
            0,
            d["so_path"],
            d["problem"],
            d.get("kernel_name", "unknown"),
            verify=verify,
            verify_tol=verify_tol,
            layout4=d.get("layout4"),
            a_op=d.get("a_elementwise_op"),
            b_op=d.get("b_elementwise_op"),
            cde_op=d.get("cde_elementwise_op"),
        )


if __name__ == "__main__":
    main()
