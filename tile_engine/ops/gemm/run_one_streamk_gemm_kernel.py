#!/usr/bin/env python3
"""Worker script for running Stream-K GEMM kernels in an isolated subprocess.

Stream-K is a single-problem GEMM with the same C ABI as regular GEMM, so this
worker is identical in shape to ``run_one_gemm_kernel.py`` and reuses
``GemmProblem`` / ``GpuGemmRunner`` unchanged -- the Stream-K specifics live
entirely inside the force-included kernel and ``streamk_gemm_ctypes_lib.cpp``.

- Receives kernel config + problem via stdin as JSON
- Loads the .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (one line per kernel, flushed)
- A GPU fault kills only this process; the parent driver can continue

Input JSON format:
    Single: {"so_path": "...", "problem": {"M":.., "N":.., "K":..}, "kernel_name": "..."}
    Batch:  {"items": [{"so_path": "...", "problem": {...}, "kernel_name": "..."}, ...]}

Optional top-level keys ``verify`` (bool) and ``verify_tol`` (float) enable an
fp32 numpy reference check; when set, each OK result also carries ``verified``
and ``max_rel``. Stream-K's Atomic reduction does multiple fp16 atomic-adds (one
per K-split partial), so it is inherently noisier than a single fp32->fp16 store;
the default gate tolerance (2e-2) is loose enough to pass while still catching
gross errors.

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
    GemmProblem,
    GpuGemmRunner,
    _dtype_from_kernel_name,
    _fp32_to_bf16_u16,
    _bf16_u16_to_fp32,
    _fp32_to_fp8_u8,
    _fp8_u8_to_fp32,
    _fp32_to_bf8_u8,
    _bf8_u8_to_fp32,
)
import numpy as np  # noqa: E402


def _run_one(idx, so_path, prob_dict, kernel_name, verify=False, verify_tol=2e-2):
    """Run a single kernel and emit its result as one JSON line.

    When ``verify`` is set, the kernel output is checked against an fp32 numpy
    reference (``A @ B``) using the global relative metric
    ``max|out - ref| / max|ref|``; the emitted ``verified`` field then reflects
    correctness, not just liveness (``non_zero``).
    """
    try:
        problem = GemmProblem.from_dict(prob_dict)

        np.random.seed(42)
        A = (np.random.randn(problem.M, problem.K) * 0.1).astype(np.float32)
        B = (np.random.randn(problem.K, problem.N) * 0.1).astype(np.float32)

        # CRITICAL: load the library ONLY inside this subprocess. The runner reads
        # dtype + layout off the kernel name and arranges/encodes A/B accordingly.
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
            if verify:
                # Reference uses the SAME quantized inputs the device sees, per the
                # kernel's dtype (bf16/fp8/bf8 bit-quantization vs fp16), so the
                # metric isolates compute error from input quantization. The dtype
                # comes from the kernel name and the quantizers are the same module
                # helpers GpuGemmRunner uses to build the device buffers, so host
                # and device see identical inputs.
                kdt = _dtype_from_kernel_name(runner.kernel_name)
                if kdt == "bf16":
                    Aq = _bf16_u16_to_fp32(_fp32_to_bf16_u16(A))
                    Bq = _bf16_u16_to_fp32(_fp32_to_bf16_u16(B))
                elif kdt == "fp8":
                    Aq = _fp8_u8_to_fp32(_fp32_to_fp8_u8(A))
                    Bq = _fp8_u8_to_fp32(_fp32_to_fp8_u8(B))
                elif kdt == "bf8":
                    Aq = _bf8_u8_to_fp32(_fp32_to_bf8_u8(A))
                    Bq = _bf8_u8_to_fp32(_fp32_to_bf8_u8(B))
                else:  # fp16
                    Aq = A.astype(np.float16).astype(np.float32)
                    Bq = B.astype(np.float16).astype(np.float32)
                ref = Aq @ Bq
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
