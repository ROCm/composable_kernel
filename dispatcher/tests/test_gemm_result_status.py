#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only checks for the explicit "unsupported" run status.

The ctypes run() entry point returns -3 when the selected kernel rejects the
problem (IsSupportedArgument throws "... not supported ..." inside run()). That is not a
numerical failure: the kernel never launched. GemmResult and friends expose it
as ``unsupported`` (``success`` stays False for backward compatibility), and the
search-space sweep counts it as a skip without verifying the output. When HIP
sees no usable GPU the sweep runs nothing and fails once with the cause.

-1 (host/HIP/launch error) and -2 (no suitable kernel) stay real failures.
The ctypes library is mocked, so no GPU or built .so is needed.

Run: python3 -m pytest tests/test_gemm_result_status.py -v
"""

import argparse
import json
import re
import sys
import tempfile
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

    def _args(self, json_path=None):
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
            json=json_path,
            elementwise_op=None,
        )

    def _run(self, statuses, gpu_problem=None, json_path=None, hip_error="",
             build_errors=None):
        sweep = self.sweep
        cfgs = [SimpleNamespace(name=f"k{i}") for i in range(len(statuses))]
        sos = [Path(f"k{i}.so") for i in range(len(statuses))]
        by_name = dict(zip((c.name for c in cfgs), statuses))
        verified = []
        self.ran = []

        def make_runner(variant, cfg, so):
            self.ran.append(cfg.name)
            return SimpleNamespace(name=cfg.name)

        def invoke(variant, runner, ops):
            res = _result(GemmResult, by_name[runner.name])
            res.time_ms = 1.0
            return res

        def verify(variant, cfg, runner, result, ops):
            verified.append(cfg.name)
            return 0.0

        def build(configs, verbose=True, max_workers=None, errors=None):
            # Stands in for a build in which nothing compiled. Writing through
            # `errors` also pins that the sweep passes the out-parameter at all
            # -- drop it there and this raises on the None.
            for cfg in configs:
                errors[cfg.name] = build_errors[cfg.name]
            return [None] * len(configs)

        build_patch = (
            mock.patch.object(sweep, "setup_multiple_gemm_dispatchers", side_effect=build)
            if build_errors is not None
            else mock.patch.object(
                sweep, "setup_multiple_gemm_dispatchers", return_value=sos)
        )

        with mock.patch.object(
            sweep, "expand_sweep", return_value=cfgs
        ), build_patch, mock.patch.object(
            sweep, "_make_runner", side_effect=make_runner
        ), mock.patch.object(sweep, "_invoke", side_effect=invoke), mock.patch.object(
            sweep, "_verify", side_effect=verify
        ), mock.patch.object(
            sweep, "hip_device_problem", return_value=gpu_problem
        ), mock.patch.object(
            sweep, "hip_last_error", return_value=hip_error
        ), mock.patch("builtins.print") as printed:
            rc = sweep.run(self._args(json_path))
        # Kept so tests can assert on the summary the operator actually reads,
        # not just on the JSON.
        self.printed = "\n".join(
            " ".join(str(a) for a in c.args) for c in printed.call_args_list
        )
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

    def test_no_gpu_fails_without_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            rc, verified = self._run(
                [STATUS_OK, STATUS_OK], gpu_problem="no /dev/kfd", json_path=out
            )
            data = json.loads(out.read_text())
        self.assertEqual(rc, 1)
        self.assertEqual(self.ran, [], "no kernel may run without a GPU")
        self.assertEqual(verified, [])
        self.assertEqual(data["gpu_problem"], "no /dev/kfd")
        self.assertEqual(data["n_not_run"], 2)
        self.assertEqual([k["status"] for k in data["kernels"]], ["not_run"] * 2)

    def test_pending_hip_error_is_recorded_with_the_run_failure(self):
        # A bare "status=-1" is what made develop #2687 unreadable: it collapses
        # every hipMalloc/hipMemcpy/launch failure and every non-"not supported"
        # C++ exception into one integer. When HIP left an error pending, name it.
        hip_error = "hipErrorIllegalAddress: an illegal memory access was encountered"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            rc, _ = self._run([STATUS_ERROR], json_path=out, hip_error=hip_error)
            data = json.loads(out.read_text())
        self.assertEqual(rc, 1)
        err = data["kernels"][0]["error"]
        # Use assertIn rather than assertEqual: the human-readable description
        # alongside "status=-1" may be revised without breaking the contract.
        self.assertIn("status=-1", err)
        self.assertIn(hip_error, err)

    def test_run_failure_without_a_pending_hip_error_stays_bare(self):
        # When no HIP error is pending, "status=-1" must still appear; the
        # empty string must not produce "status=-1 ()" and the description
        # must not accidentally include HIP-error text.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            self._run([STATUS_ERROR], json_path=out, hip_error="")
            data = json.loads(out.read_text())
        err = data["kernels"][0]["error"]
        self.assertIn("status=-1", err)
        self.assertNotIn("HIP error:", err)   # no pending error → not appended

    def test_build_failure_reason_reaches_the_json_and_the_summary(self):
        # develop #2700: every kernel failed to build and the report said only
        # "build_fail " with nothing after it, so the lane was undiagnosable.
        reason = "Compile failed: clang++: not found"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            rc, verified = self._run(
                [STATUS_OK, STATUS_OK],
                json_path=out,
                build_errors={"k0": reason, "k1": reason},
            )
            data = json.loads(out.read_text())
        self.assertEqual(rc, 1)
        self.assertEqual(verified, [], "nothing built, so nothing is verified")
        self.assertEqual(data["n_build_fail"], 2)
        # Per kernel in the JSON -- consumers and artifact diffs are per kernel.
        self.assertEqual([k["error"] for k in data["kernels"]], [reason, reason])

    def test_identical_build_failures_print_the_reason_once(self):
        # The whole point of grouping: 64 copies of one compiler message is the
        # wrong output, and the message is the only part that matters.
        reason = "Compile failed: clang++: not found"
        self._run(
            [STATUS_OK] * 3,
            build_errors={f"k{i}": reason for i in range(3)},
        )
        self.assertEqual(self.printed.count(reason), 1, self.printed)
        self.assertIn("[3x] k0 (+2 more)", self.printed)
        self.assertIn("1 distinct reason", self.printed)

    def test_distinct_build_failures_stay_apart(self):
        self._run(
            [STATUS_OK] * 2,
            build_errors={"k0": "Compile failed: A", "k1": "Link failed: B"},
        )
        self.assertIn("Compile failed: A", self.printed)
        self.assertIn("Link failed: B", self.printed)
        self.assertIn("2 distinct reasons", self.printed)

    def test_reasonless_build_failure_still_says_so(self):
        # The fallback text must not read as an empty reason, which is exactly
        # the output this whole change exists to remove.
        self._run([STATUS_OK], build_errors={"k0": ""})
        self.assertIn("no reason reported by the build", self.printed)


class TestGroupBuildFailures(unittest.TestCase):
    """_group_build_failures is pure, so test it directly."""

    @classmethod
    def setUpClass(cls):
        import test_gemm_search_space as sweep

        cls.group = staticmethod(sweep._group_build_failures)

    def test_groups_by_reason_in_first_seen_order(self):
        groups = self.group([
            {"name": "a", "status": "build_fail", "error": "X"},
            {"name": "b", "status": "build_fail", "error": "Y"},
            {"name": "c", "status": "build_fail", "error": "X"},
        ])
        self.assertEqual(list(groups), ["X", "Y"])
        self.assertEqual(groups["X"], ["a", "c"])

    def test_ignores_every_other_status(self):
        groups = self.group([
            {"name": "a", "status": "pass"},
            {"name": "b", "status": "run_fail", "error": "status=-1"},
            {"name": "c", "status": "not_run"},
        ])
        self.assertEqual(groups, {})


if __name__ == "__main__":
    unittest.main()
