# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the tile_engine GEMM sweep driver's --verify and exit code.

* ``GpuGemmRunner.reference`` builds the fp32 reference from the values the
  kernel reads (the host encoding run() uses), so an exact kernel verifies for
  every dtype. Against unquantized inputs fp8/bf8 input rounding alone exceeds
  the default tolerance and every kernel was reported as MISMATCH.
* ``GpuBatchedGemmRunner`` shares it, and the GEMM, stream-K and batched
  workers use it.
* A sweep with failed measurements exits 1.

The ctypes lib is replaced by a host model of an exact kernel.
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))

import gemm_full_benchmark as drv  # noqa: E402
import run_one_batched_gemm_kernel  # noqa: E402
import run_one_gemm_kernel  # noqa: E402
import run_one_streamk_gemm_kernel  # noqa: E402
from batched_gemm_utils import BatchedGemmProblem, GpuBatchedGemmRunner  # noqa: E402
from gemm_utils import (  # noqa: E402
    GemmProblem,
    GpuGemmRunner,
    _bf8_u8_to_fp32,
    _bf16_u16_to_fp32,
    _fp8_u8_to_fp32,
    _fp32_to_bf16_u16,
)

DEFAULT_TOL = 2e-2
PROBLEM = {"M": 64, "N": 96, "K": 256}
BATCHED_PROBLEM = {"batch_count": 3, **PROBLEM}


def _native(h, ocp):
    return h.astype(np.float32)


# How the device reads each host input buffer, per kernel dtype.
DEVICE_DECODE = {
    "fp16": _native,
    "fp32": _native,
    "bf16": lambda h, ocp: _bf16_u16_to_fp32(h),
    "fp8": _fp8_u8_to_fp32,
    "bf8": _bf8_u8_to_fp32,
}
# Unit roundoff of the C the kernel stores (fp8/bf8 store fp16): with exact
# compute, only this rounding is left between C and the reference (fp32: plus
# the summation order of the host matmul).
C_EPS = {"bf16": 2.0**-8, "fp32": 2.0**-20}


def _c_eps(dtype):
    return C_EPS.get(dtype, 2.0**-11)


# fp8/bf8 in both the FNUZ (gfx942) and OCP (gfx950, gfx12) formats.
CASES = [("fp16", None), ("bf16", None), ("fp32", None)] + [
    (d, o) for d in ("fp8", "bf8") for o in (False, True)
]


class _ExactKernelLib:
    """Stands in for the plain or batched ctypes lib: computes C from the host
    buffers it is given in the kernel's layout, as an exact kernel would."""

    def __init__(self, dtype, layout, use_ocp):
        self.dtype, self.layout, self.use_ocp = dtype, layout, use_ocp

    def run(self, A_h, B_h, C_h, *dims, **kw):
        def mat(X, lay):
            return X if lay == "r" else np.swapaxes(X, -1, -2)

        dec = DEVICE_DECODE[self.dtype]
        A = mat(dec(A_h, self.use_ocp), self.layout[0])
        B = mat(dec(B_h, self.use_ocp), self.layout[1])
        C = mat(A @ B, self.layout[2])
        # bf16 C is a uint16 bit pattern; fp8/bf8 store fp16.
        C_h[...] = _fp32_to_bf16_u16(C) if self.dtype == "bf16" else C
        return 0, 1.0


def _runner(dtype, use_ocp, layout="rcr", cls=GpuGemmRunner):
    r = object.__new__(cls)
    r._kernel_name = f"gemm_{dtype}_{layout}_compv3_cshuffle_intrawave_128x128x64"
    r._use_ocp = use_ocp
    r.lib = _ExactKernelLib(dtype, layout, use_ocp)
    return r


def _inputs(problem=PROBLEM):
    rng = np.random.RandomState(42)
    lead = (problem["batch_count"],) if "batch_count" in problem else ()
    M, N, K = problem["M"], problem["N"], problem["K"]
    A = (rng.randn(*lead, M, K) * 0.1).astype(np.float32)
    B = (rng.randn(*lead, K, N) * 0.1).astype(np.float32)
    return A, B


def _max_rel(got, ref):
    return float(np.max(np.abs(got.astype(np.float32) - ref)) / np.max(np.abs(ref)))


RUNNERS = {
    "gemm": (GpuGemmRunner, GemmProblem, PROBLEM),
    "batched": (GpuBatchedGemmRunner, BatchedGemmProblem, BATCHED_PROBLEM),
}


@pytest.mark.parametrize("kind", RUNNERS)
@pytest.mark.parametrize("layout", ["rcr", "rrr", "crr", "ccr"])
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_reference_matches_an_exact_kernel(dtype, use_ocp, layout, kind):
    cls, problem_cls, problem = RUNNERS[kind]
    runner = _runner(dtype, use_ocp, layout, cls)
    A, B = _inputs(problem)
    out = runner.run(A, B, problem_cls.from_dict(problem)).output
    assert _max_rel(out, runner.reference(A, B)) < _c_eps(dtype)


@pytest.mark.parametrize("dtype,use_ocp", [c for c in CASES if c[0] in ("fp8", "bf8")])
def test_unquantized_reference_mismatches_fp8_bf8(dtype, use_ocp):
    # The bug the reference fixes: input rounding alone exceeds the tolerance.
    runner = _runner(dtype, use_ocp)
    A, B = _inputs()
    out = runner.run(A, B, GemmProblem.from_dict(PROBLEM)).output
    assert _max_rel(out, A @ B) > DEFAULT_TOL


def test_reference_follows_the_fp8_format():
    A, B = _inputs()
    fnuz = _runner("fp8", False).reference(A, B)
    ocp = _runner("fp8", True).reference(A, B)
    # OCP E4M3 has one more exponent step than FNUZ, so the values differ.
    assert not np.array_equal(fnuz, ocp)


# ------------------------------------------------------------------ workers


WORKERS = {
    "gemm": (run_one_gemm_kernel, GpuGemmRunner, PROBLEM),
    "streamk": (run_one_streamk_gemm_kernel, GpuGemmRunner, PROBLEM),
    "batched": (run_one_batched_gemm_kernel, GpuBatchedGemmRunner, BATCHED_PROBLEM),
}


@pytest.mark.parametrize("kind", WORKERS)
@pytest.mark.parametrize("dtype,use_ocp", CASES)
def test_worker_verifies_an_exact_kernel(monkeypatch, capsys, kind, dtype, use_ocp):
    worker, cls, problem = WORKERS[kind]
    runner = _runner(dtype, use_ocp, cls=cls)
    monkeypatch.setattr(worker, cls.__name__, lambda **kw: runner)
    if hasattr(worker._run_one, "_ab_cache"):
        monkeypatch.delattr(worker._run_one, "_ab_cache")
    worker._run_one(0, "fake.so", problem, runner.kernel_name, verify=True)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] and out["verified"], out
    assert out["max_rel"] < _c_eps(dtype)


# ------------------------------------------------------------- exit code


def _main(monkeypatch, capsys, tmp_path, n_fail):
    lib = tmp_path / "libfake.so"
    lib.write_bytes(b"")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemm_full_benchmark.py",
            "--arch",
            "gfx950",
            "--max-kernels",
            "1",
            "--csv",
            str(tmp_path / "out.csv"),
        ],
    )
    monkeypatch.setattr(drv, "resolve_devices", lambda spec: ["0"])
    monkeypatch.setattr(
        drv, "setup_multiple_gemm_dispatchers", lambda cfgs, **kw: [lib] * len(cfgs)
    )
    monkeypatch.setattr(drv, "_run_batch_on_device", lambda *a, **kw: ([], [], n_fail))
    rc = drv.main()
    return rc, capsys.readouterr().out


@pytest.mark.parametrize("n_fail,rc", [(0, 0), (1, 1)])
def test_sweep_exit_code_reflects_failures(monkeypatch, capsys, tmp_path, n_fail, rc):
    got, out = _main(monkeypatch, capsys, tmp_path, n_fail)
    assert got == rc
    assert "BENCHMARK COMPLETE" in out


DRIVERS = [
    "gemm_full_benchmark.py",
    "batched_gemm_full_benchmark.py",
    "gemm_multi_d_full_benchmark.py",
    "streamk_gemm_full_benchmark.py",
    "grouped_gemm_full_benchmark.py",
    "block_scale_gemm/gemm_bquant/gemm_bquant_full_benchmark.py",
]


@pytest.mark.parametrize("path", DRIVERS)
def test_every_gemm_sweep_driver_returns_its_failures(path):
    # The other drivers share the summary tail; their last return must depend
    # on the failure count instead of being a constant 0.
    tree = ast.parse((GEMM_TE_DIR / path).read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    last = main.body[-1]
    assert isinstance(last, ast.Return)
    assert not isinstance(last.value, ast.Constant), ast.unparse(last)
    assert "fail" in ast.unparse(last.value)
