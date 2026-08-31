#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the multi_d GEMM bridge.

Multi-D GEMM fuses one or more extra D operands into the epilogue
(E = elementwise_op(A@B, D0, D1, ...)). Its kernel therefore differs from plain
GEMM by a 4-char (A,B,C,D) layout code plus a
``_multid_{elementwise_op}_d{num_d_tensors}`` name suffix. That suffix is the
byte-parity invariant tying config -> codegen -> the compiled kernel name the
runtime reports, so distinct D-counts / element-wise ops never collapse onto one
kernel. The same two knobs are surfaced in the codegen JSON's
``multi_d_config`` block, which is where unified_gemm_codegen expands them.

Everything under test is pure host-side logic (name generation, the codegen
JSON projection, and the shipped configs/*.json). No GPU, hipcc, or dispatcher
build is required.

Run: python3 -m pytest projects/composablekernel/dispatcher/tests/test_multi_d_bridge.py -v
"""

import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
REPO_ROOT = DISPATCHER_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    _output_dtype,
    _dtype_from_kernel_name,
)

_CONFIG_DIR = (
    REPO_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "gemm_multi_d"
    / "configs"
)


def _make_config(**overrides):
    kw = dict(
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_c="fp16",
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
        d_layout="row",
        variant="multi_d",
        elementwise_op="MultiDAdd",
        num_d_tensors=1,
    )
    kw.update(overrides)
    return GemmKernelConfig(**kw)


class TestMultiDName(unittest.TestCase):
    """The multi_d kernel-name contract (the byte-parity invariant)."""

    def test_name_carries_multid_suffix(self):
        cfg = _make_config()
        self.assertTrue(
            cfg.name.endswith("_multid_MultiDAdd_d1"), cfg.name
        )

    def test_name_uses_four_char_layout(self):
        # multi_d appends the D layout char after the 3-char A,B,C code.
        for la, lb, lc, ld in (
            ("row", "col", "row", "row"),
            ("row", "row", "row", "col"),
            ("col", "col", "row", "row"),
        ):
            cfg = _make_config(
                layout_a=la, layout_b=lb, layout_c=lc, d_layout=ld,
            )
            four = cfg.layout + ("r" if ld == "row" else "c")
            self.assertIn(f"_{four}_", cfg.name)

    def test_full_stem_is_stable(self):
        cfg = _make_config(
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
            elementwise_op="MultiDAdd", num_d_tensors=2,
        )
        expected = (
            "gemm_fp16_rcrr_compv4_cshuffle_intrawave"
            "_True_True_True_False"
            "_128x128x32_2x2x1_32x32x16"
            "_multid_MultiDAdd_d2"
        )
        self.assertEqual(cfg.name, expected)

    def test_d_count_changes_the_name(self):
        one = _make_config(num_d_tensors=1).name
        two = _make_config(num_d_tensors=2).name
        self.assertNotEqual(one, two)
        self.assertTrue(one.endswith("_d1"))
        self.assertTrue(two.endswith("_d2"))

    def test_elementwise_op_changes_the_name(self):
        add = _make_config(elementwise_op="MultiDAdd").name
        mul = _make_config(elementwise_op="MultiDMultiply").name
        self.assertNotEqual(add, mul)
        self.assertIn("_multid_MultiDMultiply_", mul)

    def test_dtype_recovers_from_name(self):
        for dtype in ("fp16", "bf16"):
            cfg = _make_config(
                dtype_a=dtype, dtype_b=dtype, dtype_c=_output_dtype(dtype),
            )
            self.assertEqual(_dtype_from_kernel_name(cfg.name), dtype)


class TestMultiDCodegenJson(unittest.TestCase):
    """The codegen JSON must carry the multi_d_config block."""

    def test_codegen_json_has_multi_d_block(self):
        cfg = _make_config(elementwise_op="MultiDAdd", num_d_tensors=2)
        j = cfg.to_codegen_json()
        self.assertIn("multi_d_config", j)
        md = j["multi_d_config"]
        self.assertEqual(md["elementwise_ops"], ["MultiDAdd"])
        self.assertEqual(md["num_d_tensors"], [2])

    def test_codegen_json_core_blocks(self):
        j = _make_config().to_codegen_json()
        self.assertIn("tile_config", j)
        self.assertIn("trait_config", j)
        # dispatcher wave_* is projected onto codegen warp_* (warps per block).
        self.assertEqual(j["tile_config"]["warp_m"], [2])


class TestMultiDShippedConfigs(unittest.TestCase):
    """The configs/*.json the bridge codegens from must be well-formed sweeps."""

    def test_config_dir_exists(self):
        self.assertTrue(_CONFIG_DIR.is_dir(), _CONFIG_DIR)

    def test_configs_are_valid_sweeps(self):
        configs = sorted(_CONFIG_DIR.glob("*.json"))
        self.assertGreater(len(configs), 0, "no multi_d configs shipped")
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


class TestMultiDGfx1250Config(unittest.TestCase):
    """The gfx1250 (MI400) CI config must use WMMA warp tiles.

    gfx1250 has no MFMA units: it runs the RDNA4 WMMA path whose fp16/bf16 warp
    tile is 16x16x32 (see arch_specs_generated.py). The merged multi_d bridge
    (#9308) shipped only an MFMA CI config (32x32x16, valid on gfx942/gfx950),
    which the kernel reports as unsupported (status -2/-1) on gfx1250. The
    gfx1250 CI config therefore pins the WMMA warp tile so the sweep produces
    kernels that actually run on MI400.
    """

    _GFX1250_CONFIG = _CONFIG_DIR / "default_ci_config_gfx1250.json"

    def test_gfx1250_config_exists(self):
        self.assertTrue(self._GFX1250_CONFIG.is_file(), self._GFX1250_CONFIG)

    def test_gfx1250_config_uses_wmma_warp_tile(self):
        with open(self._GFX1250_CONFIG) as f:
            tc = json.load(f)["tile_config"]
        # WMMA (RDNA4) fp16/bf16 warp tile, not MFMA (CDNA) 32x32x16.
        self.assertEqual(tc["warp_tile_m"]["values"], [16])
        self.assertEqual(tc["warp_tile_n"]["values"], [16])
        self.assertEqual(tc["warp_tile_k"]["values"], [32])

    def test_gfx1250_config_keeps_multi_d_block(self):
        # gfx1250 enablement must not drop the D-fusion knobs that make this a
        # multi_d config rather than a plain GEMM config.
        with open(self._GFX1250_CONFIG) as f:
            data = json.load(f)
        self.assertIn("multi_d_config", data)
        md = data["multi_d_config"]
        self.assertIn("elementwise_ops", md)
        self.assertIn("num_d_tensors", md)


if __name__ == "__main__":
    unittest.main()
