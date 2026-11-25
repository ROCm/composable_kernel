#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the preshuffle GEMM bridge.

The preshuffle bridge pre-permutes the B operand, so its kernel differs from
plain GEMM by a ``_preshuffle`` name token (plus ``_permuteN`` when the
permute-N B-shuffle is selected). That token is the byte-parity invariant tying
config -> codegen -> the compiled kernel name the runtime reports, so a
preshuffle kernel can never collapse onto its plain-GEMM sibling. The
``permute_n`` knob is also surfaced at the top level of the codegen JSON, which
is where unified_gemm_codegen selects shuffle_b_permuteN vs shuffle_b.

Everything under test is pure host-side logic (name generation, the codegen
JSON projection, and the shipped configs/*.json). No GPU, hipcc, or dispatcher
build is required.

Run: python3 -m pytest tests/test_preshuffle_bridge.py -v
"""

import json
import sys
import unittest
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
REPO_ROOT = DISPATCHER_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    setup_multiple_gemm_dispatchers,
    _output_dtype,
    _dtype_from_kernel_name,
    _layout_from_kernel_name,
)

_CONFIG_DIR = (
    REPO_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "gemm_preshuffle"
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
        variant="preshuffle",
    )
    kw.update(overrides)
    return GemmKernelConfig(**kw)


class TestPreshuffleName(unittest.TestCase):
    """The preshuffle kernel-name contract (the byte-parity invariant)."""

    def test_name_gains_preshuffle_suffix(self):
        cfg = _make_config()
        self.assertTrue(cfg.name.endswith("_preshuffle"), cfg.name)

    def test_name_is_plain_gemm_plus_suffix(self):
        # The preshuffle name must be exactly the plain-GEMM name +
        # "_preshuffle", so it shares codegen but never collides.
        common = dict(
            dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32",
            layout_a="row", layout_b="row", layout_c="row",
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
        )
        plain = GemmKernelConfig(variant="standard", **common)
        pre = GemmKernelConfig(variant="preshuffle", **common)
        self.assertEqual(pre.name, plain.name + "_preshuffle")

    def test_permute_n_appends_token(self):
        # permute_n selects the shuffle_b_permuteN pipeline; it must be visible
        # in the kernel name so the two shuffles never collapse.
        base = _make_config(permute_n=False)
        permuted = _make_config(permute_n=True)
        self.assertFalse(base.name.endswith("_permuteN"))
        self.assertTrue(permuted.name.endswith("_preshuffle_permuteN"), permuted.name)
        self.assertNotEqual(base.name, permuted.name)

    def test_permute_n_build_hard_fails(self):
        # A permute_n=True config must be rejected at build time (the permuteN
        # pipeline is not bridged yet), before any codegen/compile happens.
        from gemm_utils import setup_multiple_gemm_dispatchers
        cfg = _make_config(permute_n=True, gfx_arch="gfx1250")
        with self.assertRaises(ValueError):
            setup_multiple_gemm_dispatchers([cfg])

    def test_dtype_and_layout_recover_from_name(self):
        for dtype in ("fp16", "bf16"):
            for la, lb, lc in (
                ("row", "col", "row"),
                ("row", "row", "row"),
                ("col", "col", "row"),
            ):
                cfg = _make_config(
                    dtype_a=dtype, dtype_b=dtype, dtype_c=_output_dtype(dtype),
                    layout_a=la, layout_b=lb, layout_c=lc,
                )
                name = cfg.name
                self.assertEqual(_dtype_from_kernel_name(name), dtype)
                self.assertEqual(_layout_from_kernel_name(name), cfg.layout)


class TestPreshuffleCodegenJson(unittest.TestCase):
    """The codegen JSON must surface the top-level permute_n knob."""

    def test_permute_n_in_codegen_json(self):
        self.assertEqual(_make_config(permute_n=True).to_codegen_json()["permute_n"], True)
        self.assertEqual(_make_config(permute_n=False).to_codegen_json()["permute_n"], False)

    def test_codegen_json_core_blocks(self):
        j = _make_config().to_codegen_json()
        self.assertIn("tile_config", j)
        self.assertIn("trait_config", j)
        # dispatcher wave_* is projected onto codegen warp_* (warps per block).
        self.assertEqual(j["tile_config"]["warp_m"], [2])


class TestPreshuffleShippedConfigs(unittest.TestCase):
    """The configs/*.json the bridge codegens from must be well-formed sweeps."""

    def test_config_dir_exists(self):
        self.assertTrue(_CONFIG_DIR.is_dir(), _CONFIG_DIR)

    def test_configs_are_valid_sweeps(self):
        configs = sorted(_CONFIG_DIR.glob("*.json"))
        self.assertGreater(len(configs), 0, "no preshuffle configs shipped")
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


class TestPreshufflePersistentLaunch(unittest.TestCase):
    """Persistent vs non-persistent launch parity (correctness regression guard).

    Old-TE ships ``persistent: [true, false]`` for the preshuffle op, so BOTH
    are legitimate, must-support configs. On gfx942 both verify against an fp32
    reference at rel ~4e-4 (see PR9307/coder/persistent_bug_fix.md). The
    host-launch invariant that makes the non-persistent kernel correct is the
    grid choice: a persistent kernel loops over tiles from an occupancy-sized
    grid (``MaxOccupancyGridSize``); a non-persistent kernel needs a full
    one-block-per-output-tile grid (``GridSize``). If the codegen ever emitted
    the occupancy grid for a non-persistent kernel, only a subset of output
    tiles would be written and the result would be silently wrong (the failure
    mode originally reported for this PR: rel ~1.4). These tests pin that
    invariant at the codegen level so a regression is caught without a GPU.
    """

    @staticmethod
    def _gen():
        sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
        import unified_gemm_codegen as ugc

        gen = ugc.CKTileKernelGenerator("fp16", "rcr")
        tile = ugc.TileConfig(
            tile_m=128, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        )

        def make(persistent):
            trait = ugc.TraitConfig(
                pipeline="preshufflev2", epilogue="default", scheduler="default",
                pad_m=False, pad_n=False, pad_k=False, persistent=persistent,
            )
            return ugc.KernelConfig(
                tile=tile, trait=trait, variant=ugc.GemmVariant.PRESHUFFLE,
                preshuffle=True, permute_n=True,
            )

        return gen, make

    def test_non_persistent_uses_full_gridsize(self):
        gen, make = self._gen()
        code = gen._launch_function(make(persistent=False))
        self.assertIn("GemmKernel::GridSize(args.M, args.N, args.k_batch)", code)
        self.assertNotIn("MaxOccupancyGridSize", code)

    def test_persistent_uses_occupancy_gridsize(self):
        gen, make = self._gen()
        code = gen._launch_function(make(persistent=True))
        self.assertIn("MaxOccupancyGridSize", code)

    def test_persistent_flag_reaches_kernel_traits(self):
        # UsePersistentKernel selects the device kernel entry point (tile-looping
        # vs blockIdx). It is emitted as a compile-time constant in the selected-
        # kernel struct and must track the trait for both values, otherwise the
        # host grid and the device entry point disagree.
        gen, make = self._gen()
        false_src = gen._selected_kernel_struct(make(False), "k_np")
        true_src = gen._selected_kernel_struct(make(True), "k_p")
        self.assertIn("UsePersistentKernel = false", false_src)
        self.assertIn("UsePersistentKernel = true", true_src)


class TestPreshuffleConfigsCoverPersistentFalse(unittest.TestCase):
    """The shipped configs must exercise persistent=False (an Old-TE config).

    Dropping persistent=False from the sweep would hide the non-persistent path
    rather than validate it; Old-TE runs both, so the bridge must too.
    """

    def test_ci_and_default_configs_include_persistent_false(self):
        for name in ("default_ci_config.json", "default_config.json"):
            path = _CONFIG_DIR / name
            if not path.exists():
                continue
            with self.subTest(config=name):
                with open(path) as f:
                    data = json.load(f)
                vals = data["trait_config"]["persistent"]["values"]
                self.assertIn(False, vals, f"{name} must sweep persistent=False")
                self.assertIn(True, vals, f"{name} must sweep persistent=True")


class TestPreshuffleGfx1250(unittest.TestCase):
    def test_arch_spec_survives_regeneration(self):
        sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
        from arch_specs_generated import PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS
        specs = json.loads((DISPATCHER_DIR / "codegen" / "arch_specs.json").read_text())
        expected = {"fp16_fp16_fp32": [[16, 16, 32]],
                    "bf16_bf16_fp32": [[16, 16, 32]],
                    "fp8_fp8_fp32": [[16, 16, 64]],
                    "bf8_bf8_fp32": [[16, 16, 64]]}
        self.assertEqual(specs["preshuffle_warp_tile_combos"]["gfx1250"], expected)
        self.assertEqual(PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS["gfx1250"], expected)

    def test_wave32_lds_distribution_rejects_oversized_k(self):
        sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
        from unified_gemm_codegen import is_preshuffle_config_valid
        args = dict(tile_m=128, tile_n=128, tile_k=512, warp_m=2, warp_n=2,
                    warp_k=1, warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
                    datatype="fp16")
        self.assertTrue(is_preshuffle_config_valid(**args, warp_size=64))
        self.assertFalse(is_preshuffle_config_valid(**args, warp_size=32))

    def test_ci_configs_generate_all_epilogues_and_persistent_modes(self):
        sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
        from gemm_utils import expand_sweep
        from unified_gemm_codegen import UnifiedGemmCodegen, GemmVariant
        for dtype in ("fp16", "bf16"):
            configs = expand_sweep(str(_CONFIG_DIR / "default_ci_config.json"),
                                   arch="gfx1250", dtype=dtype, variant="preshuffle")
            self.assertEqual(len(configs), 4)
            self.assertEqual({(c.epilogue, c.persistent) for c in configs},
                             {("default", False), ("default", True),
                              ("cshuffle", False), ("cshuffle", True)})
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for index, config in enumerate(configs):
                    config_file = root / f"{index}.json"
                    config_file.write_text(json.dumps(config.to_codegen_json()))
                    gen = UnifiedGemmCodegen(root / str(index), datatype=dtype,
                                            layout="rcr", gpu_target="gfx1250",
                                            config_file=config_file,
                                            variants=[GemmVariant.PRESHUFFLE])
                    result = gen.generate_all(parallel=False)
                    self.assertEqual(result["failed"], [])
                    self.assertEqual(len(result["kernels"]), 1)
                    self.assertEqual(Path(result["kernels"][0]).stem, config.name)


if __name__ == "__main__":
    unittest.main()
