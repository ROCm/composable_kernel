#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Worker script for running BQuant GEMM kernels in an isolated subprocess.

Mirrors tile_engine/ops/gemm/run_one_gemm_kernel.py (PR #8997) but for
BQuantGrouped GEMM. No verify path is included (no TE/dispatcher parity check).

- Receives kernel config + problem via stdin as JSON
- Loads the .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (one line per kernel, flushed)
- A GPU fault kills only this process; the parent driver can continue

Input JSON format:
    Single: {"so_path": "...", "problem": {"M":.., "N":.., "K":.., "quant_group_k":..},
             "kernel_name": "..."}
    Batch:  {"items": [{...}, ...]}

Output JSON format (one line per kernel):
    {"idx": 0, "ok": true, "ms": 1.23, "tflops": 45.6, "non_zero": 1, "kernel": "..."}
    {"idx": 1, "ok": false, "error": "...", "kernel": "..."}
"""

import json
import os
import sys

# Add dispatcher python paths from environment (os.pathsep-separated).
_bquant_pypath = os.environ.get("BQUANT_PYPATH", "")
if _bquant_pypath:
    for p in _bquant_pypath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)

import numpy as np  # noqa: E402
from grouped_gemm_bquant_utils import (  # noqa: E402
    BQuantGemmProblem,
    BQuantGpuGemmRunner,
)


def _make_inputs(M, N, K, quant_group_k, rng):
    """Generate random fp8 A, B, float32 BQ inputs for a BQuant problem."""
    prob = BQuantGemmProblem(M=M, N=N, K=K, quant_group_k=quant_group_k)

    # A: [M, K] fp8 (e4m3)
    A = (rng.randn(M, K) * 0.1).astype(np.float32)
    # Quantise to fp8 range and cast via uint8 view to avoid numpy fp8 ABI issues.
    A = np.clip(A, -448.0, 448.0).astype(np.float32)

    # B: [K, N] fp8 (col-major means stride=K)
    B = (rng.randn(K, N) * 0.1).astype(np.float32)
    B = np.clip(B, -448.0, 448.0).astype(np.float32)

    # BQ scale: [QK_B, QN_B] float32, positive
    BQ = rng.uniform(0.5, 1.5, (prob.QK_B, prob.QN_B)).astype(np.float32)

    # Cast A and B to float8_e4m3fn (fp8) if available, else keep float32 for
    # type-compatibility (the ctypes lib copies raw bytes; dtype must match compiled type).
    try:
        A = A.astype(np.float8_e4m3fn)
        B = B.astype(np.float8_e4m3fn)
    except AttributeError:
        # numpy < 2.0: pass float32 as a stand-in so the buffer size matches.
        # On real hardware the kernel will be compiled for fp8_t; pass the right bytes.
        A = A.view(np.uint8)[::4].copy()  # rough byte-size match
        B = B.view(np.uint8)[::4].copy()

    return A, B, BQ, prob


def _run_one(idx, so_path, prob_dict, kernel_name):
    """Run a single BQuant kernel and emit its result as one JSON line."""
    try:
        M  = int(prob_dict["M"])
        N  = int(prob_dict["N"])
        K  = int(prob_dict["K"])
        qk = int(prob_dict.get("quant_group_k", 128))

        # Cache inputs per (M, N, K, qk) so batch mode doesn't regenerate for every kernel.
        cache = getattr(_run_one, "_input_cache", {})
        key = (M, N, K, qk)
        if key not in cache:
            rng = np.random.RandomState(42)
            cache[key] = _make_inputs(M, N, K, qk, rng)
            _run_one._input_cache = cache
        A, B, BQ, prob = cache[key]

        # CRITICAL: load the library ONLY inside this subprocess.
        runner = BQuantGpuGemmRunner(so_path)
        result = runner.run(A, B, BQ, prob)

        non_zero = int(np.count_nonzero(result.C)) if result.C is not None else 0
        tflops = 2.0 * M * N * K / (result.time_ms * 1e-3) / 1e12

        print(
            json.dumps({
                "idx": idx,
                "ok": True,
                "ms": result.time_ms,
                "tflops": tflops,
                "non_zero": non_zero,
                "kernel": kernel_name,
            }),
            flush=True,
        )

    except Exception as e:
        print(
            json.dumps({"idx": idx, "ok": False, "error": str(e), "kernel": kernel_name}),
            flush=True,
        )


def main():
    """Read JSON from stdin, run BQuant kernel(s), output results."""
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
            _run_one(i, item["so_path"], item["problem"], item.get("kernel_name", "unknown"))
    else:
        _run_one(0, d["so_path"], d["problem"], d.get("kernel_name", "unknown"))


if __name__ == "__main__":
    main()
