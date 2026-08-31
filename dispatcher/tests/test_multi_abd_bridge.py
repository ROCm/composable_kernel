#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the multi_abd GEMM bridge.

The Tile Engine -> Dispatcher multi_abd bridge relies on one hard invariant:
``GemmKernelConfig.name`` must reproduce, byte-for-byte, the kernel stem that
``unified_gemm_codegen.py`` bakes into the generated kernel (and that the .so
reports at runtime). For multi_abd that stem carries the 4-char (A,B,E,D)
layout plus a ``_multiabd_a{na}_b{nb}_d{nd}_{aop}_{bop}_{cdeop}`` suffix, so
distinct tensor counts / element-wise ops can never collapse onto one kernel.

These tests exercise only pure host-side logic (name generation, the codegen
JSON projection, and the shipped configs/*.json). No GPU, hipcc, or build is
required, so the suite runs green in CPU-only CI.

Run: python3 -m pytest tests/test_multi_abd_bridge.py -v
"""

import ctypes
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
REPO_ROOT = DISPATCHER_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    MultiABDDispatcherLib,
    _output_dtype,
    _dtype_from_kernel_name,
)

# The shipped multi_abd sweep configs the bridge codegens from.
_CONFIG_DIR = (
    REPO_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "gemm_multi_abd"
    / "configs"
)


def _make_config(**overrides):
    """A canonical multi_abd config; overrides tweak individual fields."""
    kw = dict(
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_c="fp16",
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
        layout_d="row",
        variant="multi_abd",
        num_a_tensors=2,
        num_b_tensors=2,
        num_d_tensors=2,
    )
    kw.update(overrides)
    return GemmKernelConfig(**kw)


class TestMultiAbdName(unittest.TestCase):
    """The multi_abd kernel-name contract (the byte-parity invariant)."""

    def test_name_carries_multiabd_suffix(self):
        cfg = _make_config()
        name = cfg.name
        # 4-char (A,B,E,D) layout, not the 3-char C layout.
        self.assertIn("_rcrr_", name)
        # multiabd tensor-count + op suffix, exactly as codegen emits it.
        self.assertTrue(
            name.endswith(
                "_multiabd_a2_b2_d2_PassThrough_PassThrough_PassThrough"
            ),
            name,
        )

    def test_full_stem_is_stable(self):
        # Pin the entire stem so any drift in the naming scheme is caught.
        cfg = _make_config(
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
        )
        expected = (
            "gemm_fp16_rcrr_compv4_cshuffle_intrawave"
            "_True_True_True_False"
            "_128x128x32_2x2x1_32x32x16"
            "_multiabd_a2_b2_d2_PassThrough_PassThrough_PassThrough"
        )
        self.assertEqual(cfg.name, expected)

    def test_tensor_counts_change_the_name(self):
        # Distinct tensor counts must not collapse onto one kernel name.
        base = _make_config().name
        more = _make_config(num_a_tensors=3, num_d_tensors=1).name
        self.assertNotEqual(base, more)
        self.assertIn("_a3_", more)
        self.assertIn("_d1_", more)

    def test_elementwise_ops_change_the_name(self):
        base = _make_config().name
        scaled = _make_config(cde_elementwise_op="AddScale").name
        self.assertNotEqual(base, scaled)
        self.assertTrue(scaled.endswith("_PassThrough_PassThrough_AddScale"))

    def test_dtype_recovers_from_name(self):
        # The runner reads the input dtype straight out of the compiled .so
        # name, so every dtype the bridge builds must round-trip.
        for dtype in ("fp16", "bf16"):
            cfg = _make_config(
                dtype_a=dtype, dtype_b=dtype, dtype_c=_output_dtype(dtype),
            )
            self.assertEqual(_dtype_from_kernel_name(cfg.name), dtype)

    def test_name_carries_four_char_layout(self):
        # multi_abd uses the 4-char (A,B,E,D) layout code in the stem; the D
        # char must reflect layout_d independently of the C layout.
        for la, lb, lc, ld in (
            ("row", "col", "row", "row"),
            ("row", "row", "row", "col"),
            ("col", "col", "row", "row"),
        ):
            cfg = _make_config(
                layout_a=la, layout_b=lb, layout_c=lc, layout_d=ld,
            )
            self.assertIn(f"_{cfg.layout4}_", cfg.name)


class TestMultiAbdCodegenJson(unittest.TestCase):
    """The codegen JSON projection must carry the multi_abd block."""

    def test_codegen_json_has_multi_abd_block(self):
        cfg = _make_config(
            num_a_tensors=2, num_b_tensors=2, num_d_tensors=2,
            cde_elementwise_op="AddScale",
        )
        j = cfg.to_codegen_json()
        self.assertIn("multi_abd_config", j)
        mabd = j["multi_abd_config"]
        self.assertEqual(mabd["num_a_tensors"], 2)
        self.assertEqual(mabd["num_b_tensors"], 2)
        self.assertEqual(mabd["num_d_tensors"], 2)
        self.assertEqual(mabd["cde_elementwise_op"], "AddScale")

    def test_codegen_json_core_blocks(self):
        j = _make_config().to_codegen_json()
        self.assertIn("tile_config", j)
        self.assertIn("trait_config", j)
        # dispatcher wave_* is projected onto codegen warp_* (warps per block).
        self.assertEqual(j["tile_config"]["warp_m"], [2])


class TestMultiAbdShippedConfigs(unittest.TestCase):
    """The configs/*.json the bridge codegens from must be well-formed sweeps."""

    def test_config_dir_exists(self):
        self.assertTrue(_CONFIG_DIR.is_dir(), _CONFIG_DIR)

    def test_configs_are_valid_sweeps(self):
        configs = sorted(_CONFIG_DIR.glob("*.json"))
        self.assertGreater(len(configs), 0, "no multi_abd configs shipped")
        for path in configs:
            with self.subTest(config=path.name):
                with open(path) as f:
                    data = json.load(f)
                self.assertIn("tile_config", data)
                self.assertIn("trait_config", data)
                tc = data["tile_config"]
                for key in (
                    "tile_m", "tile_n", "tile_k",
                    "warp_m", "warp_n", "warp_k",
                    "warp_tile_m", "warp_tile_n", "warp_tile_k",
                ):
                    self.assertIn(key, tc, f"{path.name} missing {key}")
                tr = data["trait_config"]
                for key in ("pipeline", "scheduler", "epilogue"):
                    self.assertIn(key, tr, f"{path.name} missing {key}")


class TestMultiAbdAbiMarshalling(unittest.TestCase):
    """GPU-free tests of the C-ABI marshalling in ``MultiABDDispatcherLib``.

    The ``.so`` is mocked (``ctypes.CDLL`` patched), so no generated kernel or
    GPU is needed. These cover the highest-risk-but-previously-untested surface:
    the declared ABI, the pointer/stride array packing, the positional argument
    order, and status forwarding.
    """

    def _make_lib(self, status=0):
        fake = mock.MagicMock()
        fake.dispatcher_run_multi_abd.return_value = status
        with mock.patch("ctypes.CDLL", return_value=fake):
            lib = MultiABDDispatcherLib(Path("/nonexistent/multi_abd.so"))
        return lib, fake

    def test_run_abi_argtypes(self):
        # The declared argtypes must match, in count and key types, the
        # positional arguments run() actually passes (19).
        _, fake = self._make_lib()
        argt = fake.dispatcher_run_multi_abd.argtypes
        self.assertEqual(len(argt), 19)
        for i in range(3):  # A/B/D host-pointer arrays
            self.assertEqual(argt[i], ctypes.POINTER(ctypes.c_void_p))
        self.assertEqual(argt[3], ctypes.c_void_p)  # E host pointer
        for i in (4, 5, 6):  # A/B/D leading-stride arrays
            self.assertEqual(argt[i], ctypes.POINTER(ctypes.c_int64))
        self.assertEqual(argt[-1], ctypes.POINTER(ctypes.c_float))  # time_ms

    def test_run_marshals_pointer_arrays_counts_and_order(self):
        lib, fake = self._make_lib(status=0)
        na, nb, nd = 2, 2, 2
        as_arrays = [np.zeros(4, np.float16) for _ in range(na)]
        bs_arrays = [np.zeros(4, np.float16) for _ in range(nb)]
        ds_arrays = [np.zeros(4, np.float16) for _ in range(nd)]
        e = np.zeros(4, np.float16)
        status, _ = lib.run(
            as_arrays, bs_arrays, ds_arrays, e,
            M=2, N=2, K=2, elem_a=2, elem_b=2, elem_d=2, elem_e=2,
            stride_as=[2, 2], stride_bs=[2, 2], stride_ds=[2, 2], stride_e=2,
        )
        self.assertEqual(status, 0)
        fake.dispatcher_run_multi_abd.assert_called_once()
        args = fake.dispatcher_run_multi_abd.call_args[0]
        self.assertEqual(len(args), 19)
        # A/B/D host-pointer arrays carry exactly one entry per tensor.
        self.assertEqual(len(args[0]), na)
        self.assertEqual(len(args[1]), nb)
        self.assertEqual(len(args[2]), nd)
        # Tensor counts occupy positions 12/13/14; M/N/K positions 15/16/17.
        self.assertEqual((args[12], args[13], args[14]), (na, nb, nd))
        self.assertEqual((args[15], args[16], args[17]), (2, 2, 2))

    def test_run_forwards_nonzero_status(self):
        # A thin shim must surface the C error code (e.g. -3 tensor-count
        # mismatch) verbatim rather than swallowing it.
        lib, fake = self._make_lib(status=-3)
        a = [np.zeros(4, np.float16)]
        status, _ = lib.run(
            a, a, a, np.zeros(4, np.float16),
            M=1, N=1, K=1, elem_a=2, elem_b=2, elem_d=2, elem_e=2,
            stride_as=[1], stride_bs=[1], stride_ds=[1], stride_e=1,
        )
        self.assertEqual(status, -3)


class TestMultiAbdGfx1250Config(unittest.TestCase):
    """The gfx1250 (MI400) multi_abd sweep config must be WMMA-shaped.

    gfx1250 runs WMMA, not MFMA: the fp16/bf16 warp tile is 16x16x32, and the
    32x32x16 MFMA tile shipped in ``default_ci_config.json`` never instantiates
    a runnable kernel there. ``default_ci_config_gfx1250.json`` is the arch
    variant that swaps in the WMMA warp tile while keeping every other sweep
    axis identical, so the bridge codegens kernels that actually launch on MI400.
    """

    _GFX1250_CONFIG = _CONFIG_DIR / "default_ci_config_gfx1250.json"

    def test_gfx1250_config_exists(self):
        self.assertTrue(
            self._GFX1250_CONFIG.is_file(),
            f"missing gfx1250 multi_abd config: {self._GFX1250_CONFIG}",
        )

    def test_gfx1250_config_uses_wmma_warp_tile(self):
        with open(self._GFX1250_CONFIG) as f:
            tc = json.load(f)["tile_config"]
        # gfx1250 fp16/bf16 WMMA warp tile is 16x16x32 (NOT the 32x32x16 MFMA
        # tile the CDNA CI config ships); a stale MFMA tile here would silently
        # produce zero runnable kernels on MI400.
        self.assertEqual(tc["warp_tile_m"]["values"], [16])
        self.assertEqual(tc["warp_tile_n"]["values"], [16])
        self.assertEqual(tc["warp_tile_k"]["values"], [32])

    def test_gfx1250_config_wave_triple_is_supported(self):
        # The swept warps-per-block triple must be one gfx1250 actually supports,
        # or expand_sweep gates the whole sweep away to an empty kernel set.
        sys.path.insert(
            0, str(REPO_ROOT / "tile_engine" / "ops" / "gemm")
        )
        from gemm_validation_utils import (  # noqa: E402
            WARP_SUPPORTED_COMBINATIONS,
        )

        self.assertIn("gfx1250", WARP_SUPPORTED_COMBINATIONS)
        with open(self._GFX1250_CONFIG) as f:
            tc = json.load(f)["tile_config"]
        for wm in tc["warp_m"]["values"]:
            for wn in tc["warp_n"]["values"]:
                for wk in tc["warp_k"]["values"]:
                    self.assertIn(
                        [wm, wn, wk],
                        WARP_SUPPORTED_COMBINATIONS["gfx1250"],
                        f"wave triple [{wm},{wn},{wk}] not supported on gfx1250",
                    )


if __name__ == "__main__":
    unittest.main()
