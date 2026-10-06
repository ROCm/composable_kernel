# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only negative tests: grouped rowcolquant/tensorquant tile-engine builders
reject async/TDM pipelines and the tdm epilogue on every arch.

Both builders hardcode a CompV3 pipeline with a CShuffle epilogue, so without
the guard such a request would emit a CompV3/CShuffle kernel under a name that
advertises a different pipeline/epilogue.
"""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))

_OPS = {
    "rowcolquant": "GroupedRowColQuantGemmKernelBuilder",
    "tensorquant": "GroupedTensorQuantGemmKernelBuilder",
}
_ARCHES = ("gfx90a", "gfx942", "gfx950", "gfx1201", "gfx1250")
_BAD_PIPELINES = ("comp_async", "comp_tdm", "comp_tdm_v2")
_TILE = {
    "tile_m": 128,
    "tile_n": 128,
    "tile_k": 64,
    "warp_m": 2,
    "warp_n": 2,
    "warp_k": 1,
    "warp_tile_m": 32,
    "warp_tile_n": 32,
    "warp_tile_k": 16,
}
_TILE_STR = "128x128x64_2x2x1_32x32x16"


def _op_dir(op):
    return os.path.join(_HERE, f"grouped_gemm_{op}")


def _builder_path(op):
    return os.path.join(_op_dir(op), f"grouped_gemm_{op}_instance_builder.py")


def _load_module(op):
    spec = importlib.util.spec_from_file_location(
        f"grouped_gemm_{op}_instance_builder_under_test", _builder_path(op)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MODS = {op: _load_module(op) for op in _OPS}


def _load_config(op):
    with open(os.path.join(_op_dir(op), "configs", "default_ci_config.json")) as f:
        return json.load(f)


def _make_builder(op, tmpdir, config, gpu_target="gfx942"):
    cfg_path = os.path.join(tmpdir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    cls = getattr(_MODS[op], _OPS[op])
    return cls(f"grouped_gemm_{op}", tmpdir, gpu_target, "fp8", "rcr", cfg_path)


def _combo(pipeline, epilogue):
    return (pipeline, epilogue, "intrawave", False, False, False, False)


class TestGroupedQuantBuilderRejects(unittest.TestCase):
    def test_generate_kernel_instance_rejects_on_every_arch(self):
        for op in _OPS:
            for arch in _ARCHES:
                with tempfile.TemporaryDirectory() as tmp:
                    b = _make_builder(op, tmp, _load_config(op), gpu_target=arch)
                    for p in _BAD_PIPELINES:
                        for e in ("cshuffle", "tdm"):
                            with self.assertRaises(ValueError, msg=(op, arch, p, e)):
                                b._generate_kernel_instance(_TILE, _combo(p, e))
                    with self.assertRaisesRegex(ValueError, "tdm epilogue"):
                        b._generate_kernel_instance(_TILE, _combo("compv3", "tdm"))
                    # Nothing may have been written for a rejected request.
                    self.assertEqual(os.listdir(tmp), ["config.json"])

    def test_config_listing_rejects(self):
        for op in _OPS:
            for key, value in (
                ("pipeline", "comp_async"),
                ("pipeline", "comp_tdm"),
                ("pipeline", "comp_tdm_v2"),
                ("epilogue", "tdm"),
            ):
                cfg = copy.deepcopy(_load_config(op))
                cfg["trait_config"][key]["values"].append(value)
                with tempfile.TemporaryDirectory() as tmp:
                    b = _make_builder(op, tmp, cfg, gpu_target="gfx1250")
                    with self.assertRaises(ValueError, msg=(op, value)):
                        b._get_sampled_kernel_list()

    def test_legacy_config_unchanged(self):
        for op in _OPS:
            with tempfile.TemporaryDirectory() as tmp:
                b = _make_builder(op, tmp, _load_config(op))
                # Listing a legacy config must not raise (its size is governed
                # by the op's existing validation, which is unchanged here).
                self.assertIsInstance(b._get_sampled_kernel_list(), list)
                _, code = b._generate_kernel_instance(
                    _TILE, _combo("compv3", "cshuffle")
                )
                self.assertIn("ck_tile::GemmPipelineAgBgCrCompV3", code)
                self.assertIn("ck_tile::CShuffleEpilogue", code)


class TestGroupedQuantGenSingleCli(unittest.TestCase):
    def _run(self, op, tmp, trait):
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(_load_config(op), f)
        return subprocess.run(
            [
                sys.executable,
                _builder_path(op),
                "--working_path",
                tmp,
                "--gpu_target",
                "gfx1250",
                "--datatype",
                "fp8",
                "--layout",
                "rcr",
                "--config_json",
                cfg_path,
                "--gen_single",
                "--kernel_name",
                "k",
                "--tile_config",
                _TILE_STR,
                "--trait_combo",
                trait,
            ],
            capture_output=True,
            text=True,
        )

    def test_gen_single_rejects(self):
        for op in _OPS:
            for p in _BAD_PIPELINES:
                with tempfile.TemporaryDirectory() as tmp:
                    r = self._run(
                        op, tmp, f"{p}_cshuffle_intrawave_False_False_False_False"
                    )
                    self.assertNotEqual(r.returncode, 0)
                    self.assertIn(f"does not support the {p} pipeline", r.stderr)
            with tempfile.TemporaryDirectory() as tmp:
                r = self._run(op, tmp, "compv3_tdm_intrawave_False_False_False_False")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("tdm epilogue", r.stderr)

    def test_gen_single_legacy_ok(self):
        for op in _OPS:
            with tempfile.TemporaryDirectory() as tmp:
                r = self._run(
                    op, tmp, "compv3_cshuffle_intrawave_False_False_False_False"
                )
                self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
