# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""AQuant configuration fidelity and public-runner packing, without a GPU.

The ABI tests use ctypes callbacks in place of a loaded shared library. They
exercise the real runner and low-level wrapper, but do not execute HIP or GEMM.
"""

import ctypes
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_aquant_utils import (  # noqa: E402
    AQuantDispatcherLib,
    AQuantGemmProblem,
    AQuantGpuGemmRunner,
    expand_aquant_sweep,
)
from unified_gemm_aquant_codegen import (  # noqa: E402
    AQuantKernelHeaderGenerator,
    _build_specs,
)


class TestAQuantSweepFidelity(unittest.TestCase):
    def setUp(self):
        self.config = {
            "variant_keys": ["fp8", "bf8"],
            "layouts": ["ccr"],
            "pipeline": "compv3",
            "scheduler": "intrawave",
            "preshuffle_aquant": True,
            "epilogues": ["default", "cshuffle"],
            "tile_configs": [{
                "tile_m": 128, "tile_n": 128, "tile_k": 128,
                "warp_m": 1, "warp_n": 4, "warp_k": 1,
                "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 128,
            }],
            "quant_groups": [{"quant_group_m": 1, "quant_group_n": 1,
                              "quant_group_k": 128}],
        }

    def expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.json"
            path.write_text(json.dumps(self.config))
            return expand_aquant_sweep(str(path), gfx_arch="gfx1250")

    def test_ccr_preshuffle_keeps_requested_epilogue_and_row_major_aq(self):
        specs = _build_specs(self.config)
        configs = self.expand()
        self.assertEqual(len(specs), 4)
        self.assertEqual(len(configs), 4)
        self.assertEqual({s.name for s in specs}, {c.name for c in configs})
        for spec in specs:
            header = " ".join(AQuantKernelHeaderGenerator().generate(spec).split())
            self.assertIn("using ALayout = ck_tile::tensor_layout::gemm::ColumnMajor;", header)
            self.assertIn("using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;", header)
            self.assertIn("using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;", header)
            self.assertIn("APreshuffleQuant = true;", header)
            self.assertEqual("using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<" in header,
                             spec.epilogue == "default")
            self.assertEqual("using GemmEpilogue = ck_tile::CShuffleEpilogue<" in header,
                             spec.epilogue == "cshuffle")
        for config in configs:
            self.assertEqual(config.pipeline, "compv3")
            self.assertEqual(config.to_codegen_config()["epilogues"], [config.epilogue])

    def test_explicit_pipeline_survives_without_preshuffle(self):
        self.config["preshuffle_aquant"] = False
        configs = self.expand()
        self.assertEqual(len(configs), 4)
        self.assertTrue(all(c.pipeline_key == "compv3" for c in configs))
        self.assertEqual({s.name for s in _build_specs(self.config)},
                         {c.name for c in configs})

    def test_singular_epilogue_and_legacy_default(self):
        del self.config["epilogues"]
        self.config["epilogue"] = "default"
        self.assertEqual({c.epilogue for c in self.expand()}, {"default"})
        del self.config["epilogue"]
        self.assertEqual({c.epilogue for c in self.expand()}, {"cshuffle"})

    def test_duplicates_and_unsupported_epilogue(self):
        self.config["epilogues"] = ["default", "default", "unsupported"]
        self.assertEqual(len(self.expand()), 2)


class TestAQuantPublicRunnerPacking(unittest.TestCase):
    @staticmethod
    def input_array(rows, cols, storage, salt, dtype):
        # Distinct logical patterns expose transpositions on rectangular arrays.
        x = ((np.arange(rows * cols).reshape(rows, cols) * 7 + salt) % 251).astype(dtype)
        if storage == "fortran":
            return np.asfortranarray(x)
        if storage == "strided":
            backing = np.zeros((rows * 2, cols * 3), dtype=dtype)
            view = backing[::2, ::3]
            view[:] = x
            return view
        return x

    def test_public_runner_through_ctypes_abi(self):
        for layout in ("rcr", "rrr", "crr", "ccr"):
            for storage in ("contiguous", "fortran", "strided"):
                for M, N, K, group_k in ((3, 5, 12, 4), (7, 2, 10, 3)):
                    with self.subTest(layout=layout, storage=storage, shape=(M, N, K)):
                        self.check_packing(layout, storage, M, N, K, group_k)

    def check_packing(self, layout, storage, M, N, K, group_k):
        problem = AQuantGemmProblem(M=M, N=N, K=K, quant_group_k=group_k, k_batch=2)
        QK = problem.QK_A
        # uint8 carries the one-byte fp8/bf8 storage without requiring a codec.
        A = self.input_array(M, K, storage, 3, np.uint8)
        B = self.input_array(K, N, storage, 17, np.uint8)
        AQ = self.input_array(M, QK, storage, 31, np.float32)
        originals = [x.copy() for x in (A, AQ, B)]
        observed = {}
        sentinel = np.arange(M * N, dtype=np.float16).reshape(M, N) + np.float16(0.5)

        def run(a, aq, b, c, m, n, k, sa, saq, sb, sc, qk, kb, elapsed):
            observed["args"] = (m, n, k, sa, saq, sb, sc, qk, kb)
            observed["A"] = ctypes.string_at(a, M * K)
            observed["AQ"] = ctypes.string_at(aq, M * QK * 4)
            observed["B"] = ctypes.string_at(b, K * N)
            ctypes.memmove(c, sentinel.ctypes.data, sentinel.nbytes)
            elapsed[0] = 1.25
            return 0

        name = ctypes.create_string_buffer(b"cpu_abi_aquant")
        callbacks = SimpleNamespace(
            dispatcher_initialize=ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0),
            dispatcher_run_aquant_gemm=ctypes.CFUNCTYPE(
                ctypes.c_int, *AQuantDispatcherLib._RUN_ARGTYPES)(run),
            dispatcher_get_kernel_name=ctypes.CFUNCTYPE(ctypes.c_void_p)(
                lambda: ctypes.addressof(name)),
            dispatcher_get_kernel_count=ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 1),
            dispatcher_cleanup=ctypes.CFUNCTYPE(None)(lambda: None),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "callback_only.so"
            path.touch()
            with patch("quant_bridge_base.ctypes.CDLL", return_value=callbacks):
                runner = AQuantGpuGemmRunner(path, layout=layout)
                result = runner.run(A, AQ, B, problem)
                self.assertEqual(runner._lib.get_kernel_count(), 1)
                runner._lib.cleanup()

        self.assertEqual(observed["args"], (
            M, N, K, K if layout[0] == "r" else M, QK,
            N if layout[1] == "r" else K, N, QK, 2))
        self.assertEqual(observed["A"], A.tobytes(order="C" if layout[0] == "r" else "F"))
        self.assertEqual(observed["B"], B.tobytes(order="C" if layout[1] == "r" else "F"))
        self.assertEqual(observed["AQ"], AQ.tobytes(order="C"))
        np.testing.assert_array_equal(result.C, sentinel)
        self.assertEqual(result.C.dtype, np.float16)
        self.assertEqual(result.time_ms, 1.25)
        self.assertEqual(result.kernel_name, "cpu_abi_aquant")
        for array, original in zip((A, AQ, B), originals):
            np.testing.assert_array_equal(array, original)


if __name__ == "__main__":
    unittest.main()
