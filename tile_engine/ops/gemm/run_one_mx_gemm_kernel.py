#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Worker script for running MX GEMM kernels in an isolated subprocess.

Speaks the run_one_gemm_kernel.py protocol (stdin JSON in, one JSON line per
kernel out) so mx_gemm_full_benchmark.py reuses the gemm_full_benchmark.py
Phase 3 runner. The payload also carries top-level ``dtype`` (fp4 | fp8) and
``arch``.

Inputs are grid-exact fp4/fp8 values with random e8m0 block scales, so --verify
checks the per-32-K scale handling against the numpy microscaled reference
(global max|out - ref| / max|ref|). A kernel that rejects the problem shape
(rc -2, e.g. K not a multiple of tile_k without K padding) is reported as
``skipped`` rather than as a failure.
"""

import json
import os
import sys

gemm_pypath = os.environ.get("GEMM_PYPATH", "")
for p in gemm_pypath.split(os.pathsep):
    if p and p not in sys.path:
        sys.path.insert(0, p)

from mx_gemm_utils import GpuMxGemmRunner, MxGemmProblem  # noqa: E402
import numpy as np  # noqa: E402

# Scale bytes 124..129 are 2^-3..2^2: exercises the scales without overflowing
# the fp16 output.
SCALE_BYTES = (124, 130)

# (M, N, K, dtype) -> [A_deq, B_deq, A, B, scale_a, scale_b, (ref, denom) | None]
_inputs = {}


def _inputs_for(runner, prob):
    key = (prob.M, prob.N, prob.K, runner.dtype)
    if key not in _inputs:
        a_deq, b_deq, a, b, sa, sb = runner.make_inputs(prob, seed=42)
        rng = np.random.default_rng(43)
        sa[:] = rng.integers(*SCALE_BYTES, size=sa.shape, dtype=np.uint8)
        sb[:] = rng.integers(*SCALE_BYTES, size=sb.shape, dtype=np.uint8)
        _inputs[key] = [a_deq, b_deq, a, b, sa, sb, None]
    return _inputs[key]


def _run_one(idx, item, dtype, arch, verify, verify_tol):
    kernel_name = item.get("kernel_name", "unknown")
    out = {"idx": idx, "kernel": kernel_name}
    try:
        p = item["problem"]
        prob = MxGemmProblem(int(p["M"]), int(p["N"]), int(p["K"]))
        # CRITICAL: load the library ONLY inside this subprocess.
        runner = GpuMxGemmRunner(item["so_path"], dtype=dtype, arch=arch)
        cached = _inputs_for(runner, prob)
        a_deq, b_deq, a, b, sa, sb, ref = cached
        try:
            result = runner.run(prob, a, b, sa, sb)
        except RuntimeError as e:
            if "rc=-2" not in str(e):
                raise
            out.update(ok=False, skipped=True, error=str(e))
            print(json.dumps(out), flush=True)
            return

        got = np.asarray(result.C, dtype=np.float32)
        ms = result.time_ms
        out.update(
            ok=True,
            ms=ms,
            tflops=prob.flops / (ms * 1e9) if ms > 0 else 0.0,
            non_zero=int(np.count_nonzero(got)),
        )
        if verify:
            if ref is None:
                r = runner.reference(a_deq, b_deq, sa, sb, prob).astype(np.float32)
                ref = cached[6] = (r, float(np.max(np.abs(r))) or 1.0)
            max_rel = float(np.max(np.abs(got - ref[0])) / ref[1])
            out.update(max_rel=max_rel, verified=bool(max_rel <= verify_tol))
    except Exception as e:
        out.update(ok=False, error=str(e))
    print(json.dumps(out), flush=True)


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

    for i, item in enumerate(d.get("items", [d])):
        _run_one(
            i,
            item,
            d["dtype"],
            d.get("arch"),
            bool(d.get("verify", False)),
            float(d.get("verify_tol", 2e-2)),
        )


if __name__ == "__main__":
    main()
