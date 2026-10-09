# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU unit tests for the batched GEMM Old-TE benchmark driver: batch_count
pass-through and result extraction from mixed stdout. No GPU is required."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import batched_gemm_benchmark as bgb  # noqa: E402

_RESULT = {
    "name": "k",
    "problem": {"batch_count": 96, "m": 64, "n": 64, "k": 64},
    "perf_result": {"latency(ms)": 1.5, "tflops(TFlops)": 2.0, "bandwidth(GB/s)": 3.0},
}
_KERNEL = Path(
    "benchmark_batched_gemm_fp16_rcr_compv3_cshuffle_intrawave"
    "_False_False_False_64x64x32_2x2x1_16x16x32"
)


class TestExtractResultJson(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(bgb.extract_result_json(json.dumps(_RESULT)), _RESULT)

    def test_verify_messages_around_json(self):
        text = (
            "For k Relative error threshold is 0.001 Absolute error threshold is 0.001\n"
            "The verification result is:correct\n"
            + json.dumps(_RESULT, indent=1)
            + "\ntrailing {not json}\n"
        )
        self.assertEqual(bgb.extract_result_json(text), _RESULT)

    def test_nested_objects_are_not_the_result(self):
        # The problem/perf_result sub-objects must not be returned on their own
        self.assertEqual(bgb.extract_result_json(json.dumps(_RESULT))["name"], "k")

    def test_no_result(self):
        for text in ("", "Verification failed, skip kernel: k", '{"a": 1}'):
            self.assertIsNone(bgb.extract_result_json(text), text)

    def test_parse_json_file_mixed_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            path.write_text("The verification result is:correct\n" + json.dumps(_RESULT))
            parsed = bgb.GemmBenchmark(tmp).parse_json_file(path)
        self.assertEqual(
            (parsed["time_ms"], parsed["tflops"], parsed["bandwidth_gb_s"]),
            (1.5, 2.0, 3.0),
        )


class TestBatchCount(unittest.TestCase):
    def _sweep(self, problem_sizes, **kw):
        driver = bgb.GemmBenchmark(".")
        driver.launch_attempted = driver.launch_failed = 0
        cmds = []

        def fake_run(cmd, **_):
            cmds.append(cmd)
            return mock.Mock(returncode=0, stdout=json.dumps(_RESULT), stderr="")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            bgb.subprocess, "run", side_effect=fake_run
        ), mock.patch.object(driver, "discover_kernels", return_value=[_KERNEL]):
            driver.build_dir = Path(tmp)
            best = driver.benchmark_sweep(problem_sizes, **kw)
        batch_args = [
            [a for a in cmd if a.startswith("-batch_count=")] for cmd in cmds
        ]
        return batch_args, best

    def test_default_batch_count(self):
        args, best = self._sweep([(64, 64, 64)])
        self.assertEqual(args, [["-batch_count=8"]])
        self.assertEqual(list(best), ["b8_m64_n64_k64_splitk1"])

    def test_batch_count_sweep(self):
        args, _ = self._sweep([(64, 64, 64)], batch_counts=[96, 8192])
        self.assertEqual(args, [["-batch_count=96"], ["-batch_count=8192"]])

    def test_explicit_bmnk_pins_batch(self):
        args, best = self._sweep(
            [(512, 64, 64, 64), (64, 64, 64)], batch_counts=[3]
        )
        self.assertEqual(args, [["-batch_count=512"], ["-batch_count=3"]])
        self.assertEqual(
            sorted(best), ["b3_m64_n64_k64_splitk1", "b512_m64_n64_k64_splitk1"]
        )

    def test_cli_batch_count(self):
        seen = {}

        def fake_sweep(self, problem_sizes, **kw):
            seen.update(problems=problem_sizes, batch_counts=kw["batch_counts"])
            return {}

        argv = [
            "prog",
            ".",
            "--problem-sizes",
            "64,64,64",
            "7,32,32,32",
            "--batch-count",
            "96",
            "256",
        ]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            sys, "argv", argv + ["--best", os.path.join(tmp, "best.txt")]
        ), mock.patch.object(
            bgb.GemmBenchmark, "benchmark_sweep", fake_sweep
        ), mock.patch("builtins.print"):
            bgb.main()
        self.assertEqual(seen["problems"], [(64, 64, 64), (7, 32, 32, 32)])
        self.assertEqual(seen["batch_counts"], [96, 256])

    def test_cli_rejects_bad_problem(self):
        for extra in (
            ["--problem-sizes", "1,2"],
            ["--problem-sizes", "0,64,64,64"],
            ["--problem-sizes", "4,64,-2,64"],
            ["--problem-sizes", "64,0,64"],
            ["--batch-count", "0"],
            ["--batch-count", "8", "-1"],
        ):
            with mock.patch.object(sys, "argv", ["prog", "."] + extra), mock.patch(
                "builtins.print"
            ):
                self.assertEqual(bgb.main(), 1, msg=extra)


if __name__ == "__main__":
    unittest.main()
