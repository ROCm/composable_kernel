#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Worker script for running MX GEMM kernels in an isolated subprocess.

Speaks the run_one_gemm_kernel.py protocol (stdin JSON in, one JSON line per
kernel out) so mx_gemm_full_benchmark.py reuses the gemm_full_benchmark.py
Phase 3 runner. The payload also carries top-level ``dtype`` (fp4 | fp8) and
``arch``.

The payload's ``init`` picks the inputs. ``grid`` (default): grid-exact fp4/fp8
values with random e8m0 block scales. ``trig`` (fp4 only): example 56's default
bench data, see _trig_fp4. Either way --verify checks the per-32-K scale
handling against the numpy microscaled reference
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

from mx_gemm_utils import (  # noqa: E402
    GpuMxGemmRunner,
    MxGemmProblem,
    dequantize_fp4_packed,
)
import numpy as np  # noqa: E402

# Scale bytes 124..129 are 2^-3..2^2: exercises the scales without overflowing
# the fp16 output.
SCALE_BYTES = (124, 130)

# Midpoints between consecutive E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)

# (M, N, K, dtype, init) -> [A_deq | None, B_deq | None, A, B, scale_a, scale_b,
# (ref, denom) | None]; trig inputs dequantize only for --verify.
_inputs = {}


def _trig_fp4(rng, rows, K, use_cos):
    """Packed E2M1 bytes [rows, K//2] and E8M0 scales [rows, K//32] as example 56's
    -data=trig draws them: each code is sin (A) or cos (B) of the element's linear index
    row * K + k, as float, rounded to the nearest E2M1 value (ties to even); scales are
    2^-3..2^3 (bytes 124..130). The same codes, not the same scale bits."""
    data = np.empty((rows, K // 2), dtype=np.uint8)
    k = np.arange(K, dtype=np.float64)
    f = np.cos if use_cos else np.sin
    step = max(1, (1 << 22) // K)
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        x = f(np.arange(r0, r1, dtype=np.float64)[:, None] * K + k).astype(np.float32)
        a = np.abs(x)
        c = np.searchsorted(_E2M1_MIDPOINTS, a)
        c += (c % 2 == 1) & (a == _E2M1_MIDPOINTS[np.minimum(c, 6)])
        codes = c.astype(np.uint8) | (np.signbit(x).astype(np.uint8) << 3)
        data[r0:r1] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = rng.integers(124, 131, size=(rows, K // 32), dtype=np.uint8)
    return data, scales


def _inputs_for(runner, prob, init):
    key = (prob.M, prob.N, prob.K, runner.dtype, init)
    if key not in _inputs:
        if init == "trig":
            if runner.dtype != "fp4":
                raise ValueError("init trig supports fp4 only")
            rng = np.random.default_rng(43)
            a, sa = _trig_fp4(rng, prob.M, prob.K, use_cos=False)
            b, sb = _trig_fp4(rng, prob.N, prob.K, use_cos=True)
            a_deq = b_deq = None
        else:
            a_deq, b_deq, a, b, sa, sb = runner.make_inputs(prob, seed=42)
            rng = np.random.default_rng(43)
            sa[:] = rng.integers(*SCALE_BYTES, size=sa.shape, dtype=np.uint8)
            sb[:] = rng.integers(*SCALE_BYTES, size=sb.shape, dtype=np.uint8)
        _inputs[key] = [a_deq, b_deq, a, b, sa, sb, None]
    return _inputs[key]


def _run_one(idx, item, dtype, arch, verify, verify_tol, init="grid"):
    kernel_name = item.get("kernel_name", "unknown")
    out = {"idx": idx, "kernel": kernel_name}
    try:
        p = item["problem"]
        prob = MxGemmProblem(int(p["M"]), int(p["N"]), int(p["K"]))
        # CRITICAL: load the library ONLY inside this subprocess.
        runner = GpuMxGemmRunner(item["so_path"], dtype=dtype, arch=arch)
        cached = _inputs_for(runner, prob, init)
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
                if a_deq is None:
                    a_deq = cached[0] = dequantize_fp4_packed(a, prob.K)
                    b_deq = cached[1] = dequantize_fp4_packed(b, prob.K).T
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
            d.get("init", "grid"),
        )


if __name__ == "__main__":
    main()
