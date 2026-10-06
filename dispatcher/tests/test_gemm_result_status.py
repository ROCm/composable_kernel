#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only checks for the explicit "unsupported" run status.

The ctypes run() entry point returns -3 when the selected kernel rejects the
problem (IsSupportedArgument throws "... not supported ..." inside run()). That is not a
numerical failure: the kernel never launched. GemmResult and friends expose it
as ``unsupported`` (``success`` stays False for backward compatibility), and the
search-space sweep counts it as a skip without verifying the output.

-1 (host/HIP/launch error) and -2 (no suitable kernel) stay real failures.
The ctypes library is mocked, so no GPU or built .so is needed.

Run: python3 -m pytest tests/test_gemm_result_status.py -v
"""

import argparse
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
BINDINGS_DIR = DISPATCHER_DIR / "bindings" / "ctypes"
STATUS_ERROR = -1
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402

from gemm_utils import (  # noqa: E402
    STATUS_NO_KERNEL,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    GemmProblem,
    GemmResult,
    GpuGemmRunner,
    GroupedGemmResult,
    MultiDGemmResult,
)


def _result(cls, status):
    if cls is GroupedGemmResult:
        return cls(outputs=[], time_ms=0.0, status=status, tflops=0.0, kernel_name="k")
    return cls(
        output=np.zeros((1, 1)), time_ms=0.0, status=status, tflops=0.0, kernel_name="k"
    )


class TestResultStatus(unittest.TestCase):
    def test_status_values(self):
        self.assertEqual((STATUS_OK, STATUS_UNSUPPORTED, STATUS_NO_KERNEL), (0, -3, -2))

    def test_properties(self):
        for cls in (GemmResult, GroupedGemmResult, MultiDGemmResult):
            with self.subTest(cls=cls.__name__):
                ok = _result(cls, STATUS_OK)
                self.assertTrue(ok.success)
                self.assertFalse(ok.unsupported)

                rej = _result(cls, STATUS_UNSUPPORTED)
                self.assertFalse(rej.success, "success must stay False")
                self.assertTrue(rej.unsupported)

                for code in (STATUS_ERROR, STATUS_NO_KERNEL):
                    bad = _result(cls, code)
                    self.assertFalse(bad.success)
                    self.assertFalse(bad.unsupported)

    def test_bindings_return_unsupported_code(self):
        # Each binding feeding these results maps a "not supported" launch
        # exception to STATUS_UNSUPPORTED, not to the error/no-kernel codes.
        for lib in (
            "gemm_ctypes_lib.cpp",
            "grouped_gemm_ctypes_lib.cpp",
            "multi_d_gemm_ctypes_lib.cpp",
            "gemm_multi_abd_ctypes_lib.cpp",
        ):
            with self.subTest(lib=lib):
                src = (BINDINGS_DIR / lib).read_text()
                tail = src[src.index('find("not supported")') :]
                first_return = re.search(r"return\s+(-?\d+)\s*;", tail)
                self.assertEqual(first_return.group(1), "-3", lib)


def _fake_runner(status, time_ms=0.0):
    """GpuGemmRunner whose ctypes lib is a mock returning ``status``."""
    runner = GpuGemmRunner.__new__(GpuGemmRunner)
    runner.lib = mock.Mock()
    runner.lib.run.return_value = (status, time_ms)
    runner._kernel_name = "gemm_fp16_rcr_compv3_cshuffle_intrawave"
    runner._use_ocp = None
    return runner


class TestRunnerMockedLib(unittest.TestCase):
    def setUp(self):
        self.A = np.ones((8, 16), dtype=np.float32)
        self.B = np.ones((16, 4), dtype=np.float32)
        self.problem = GemmProblem(M=8, N=4, K=16)

    def test_unsupported_status(self):
        res = _fake_runner(STATUS_UNSUPPORTED).run(self.A, self.B, self.problem)
        self.assertIsInstance(res, GemmResult)
        self.assertEqual(res.status, STATUS_UNSUPPORTED)
        self.assertTrue(res.unsupported)
        self.assertFalse(res.success)
        self.assertEqual(res.tflops, 0.0)

    def test_ok_status(self):
        res = _fake_runner(STATUS_OK, 1.0).run(self.A, self.B, self.problem)
        self.assertTrue(res.success)
        self.assertFalse(res.unsupported)

    def test_no_kernel_status_is_not_unsupported(self):
        res = _fake_runner(STATUS_NO_KERNEL).run(self.A, self.B, self.problem)
        self.assertFalse(res.success)
        self.assertFalse(res.unsupported)


class TestSearchSpaceCountsSkip(unittest.TestCase):
    """The sweep loop counts STATUS_UNSUPPORTED as a skip and never verifies it."""

    @classmethod
    def setUpClass(cls):
        import test_gemm_search_space as sweep

        cls.sweep = sweep

    def _args(self):
        return argparse.Namespace(
            arch="gfx942",
            variant="standard",
            dtypes="fp16",
            layouts="rcr",
            budget=0,
            seed=1,
            size=64,
            groups=1,
            warmup=0,
            repeat=1,
            json=None,
            elementwise_op=None,
        )

    def _run(self, statuses):
        sweep = self.sweep
        cfgs = [SimpleNamespace(name=f"k{i}") for i in range(len(statuses))]
        sos = [Path(f"k{i}.so") for i in range(len(statuses))]
        by_name = dict(zip((c.name for c in cfgs), statuses))
        verified = []

        def make_runner(variant, cfg, so):
            return SimpleNamespace(name=cfg.name)

        def invoke(variant, runner, ops):
            res = _result(GemmResult, by_name[runner.name])
            res.time_ms = 1.0
            return res

        def verify(variant, cfg, runner, result, ops):
            verified.append(cfg.name)
            return 0.0

        with mock.patch.object(
            sweep, "expand_sweep", return_value=cfgs
        ), mock.patch.object(
            sweep, "setup_multiple_gemm_dispatchers", return_value=sos
        ), mock.patch.object(
            sweep, "_make_runner", side_effect=make_runner
        ), mock.patch.object(sweep, "_invoke", side_effect=invoke), mock.patch.object(
            sweep, "_verify", side_effect=verify
        ), mock.patch("builtins.print"):
            rc = sweep.run(self._args())
        return rc, verified

    def test_unsupported_is_skipped_not_failed(self):
        rc, verified = self._run([STATUS_OK, STATUS_UNSUPPORTED])
        self.assertEqual(rc, 0)
        self.assertEqual(verified, ["k0"], "unsupported kernel must not be verified")

    def test_real_failure_still_fails(self):
        for code in (STATUS_ERROR, STATUS_NO_KERNEL):
            with self.subTest(status=code):
                rc, verified = self._run([STATUS_OK, code])
                self.assertEqual(rc, 1)
                self.assertEqual(verified, ["k0"])

    def test_all_unsupported_is_not_green(self):
        rc, verified = self._run([STATUS_UNSUPPORTED, STATUS_UNSUPPORTED])
        self.assertEqual(rc, 1)
        self.assertEqual(verified, [])


if __name__ == "__main__":
    unittest.main()
