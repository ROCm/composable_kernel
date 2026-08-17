#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-GPU worker for the batched-contraction bridge benchmark.

Reads one JSON job from stdin:
  {"so_path": "...", "kernel_name": "...", "dtype": "fp16",
   "problem": {"g_dims":[...],"m_dims":[...],"n_dims":[...],"k_dims":[...],"k_batch":1},
   "verify": true, "verify_tol": 2e-2}

Runs the kernel via the ctypes bridge, optionally verifies against an fp32 numpy
reference, and prints a JSON result to stdout. Isolated per process so a bad kernel
cannot take down the driver.
"""

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_DISP = _HERE.parents[3] / "dispatcher"
sys.path.insert(0, str(_DISP / "python"))

from batched_contraction_utils import (  # noqa: E402
    BatchedContractionProblem,
    GpuBatchedContractionRunner,
)


def main() -> int:
    job = json.loads(sys.stdin.read())
    so_path = job["so_path"]
    dtype = job.get("dtype", "fp16")
    prob = BatchedContractionProblem.from_dict(job["problem"])
    verify = job.get("verify", False)
    tol = float(job.get("verify_tol", 2e-2))
    expect_name = job.get("kernel_name")
    num_d = int(job.get("num_d_tensors", 0))
    elementwise = job.get("elementwise", "PassThrough")

    out = {"so_path": so_path, "kernel_name": None, "time_ms": None,
           "status": "ok", "max_rel": None, "name_match": None}
    try:
        runner = GpuBatchedContractionRunner(
            so_path, dtype=dtype, num_d_tensors=num_d, elementwise=elementwise
        )
        out["kernel_name"] = runner.kernel_name
        if expect_name is not None:
            out["name_match"] = (runner.kernel_name == expect_name)
            # The name-parity anchor guarantees we benchmarked exactly the
            # requested config. A mismatch means a stale/wrong .so, so fail the
            # job instead of silently counting it as a pass.
            if not out["name_match"]:
                out["status"] = "name_mismatch"
                print(json.dumps(out))
                return 1

        rng = np.random.default_rng(0)
        npd = runner.np_dtype
        A = rng.uniform(-2, 2, (prob.G, prob.M, prob.K)).astype(npd)
        B = rng.uniform(-2, 2, (prob.G, prob.N, prob.K)).astype(npd)
        res = runner.run(A, B, prob)
        out["time_ms"] = res.time_ms

        if verify:
            ref = runner.reference(A, B, prob, Ds=res.Ds).astype(np.float32)
            egpu = res.E.astype(np.float32)
            max_rel = float(np.max(np.abs(egpu - ref)) / (np.max(np.abs(ref)) + 1e-6))
            out["max_rel"] = max_rel
            if max_rel > tol:
                out["status"] = "verify_failed"
    except Exception as e:  # noqa: BLE001
        out["status"] = f"error: {e}"

    print(json.dumps(out))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
