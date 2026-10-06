# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only negative tests: gemm_multi_abd rejects async/TDM pipelines and the
tdm epilogue on every arch instead of silently emitting a different kernel."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gemm_multi_abd_instance_builder as mabd  # noqa: E402

_BUILDER = os.path.join(_HERE, "gemm_multi_abd_instance_builder.py")
_CI_CONFIG = os.path.join(_HERE, "configs", "default_ci_config.json")

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


def _load_config():
    with open(_CI_CONFIG) as f:
        return json.load(f)


def _make_builder(tmpdir, config, gpu_target="gfx942"):
    cfg_path = os.path.join(tmpdir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    return mabd.GemmMultiABDKernelBuilder(
        "gemm_multi_abd",
        tmpdir,
        gpu_target,
        "fp16",
        "rcrr",
        "PassThrough",
        "PassThrough",
        "PassThrough",
        2,
        2,
        2,
        cfg_path,
    )


def _combo(pipeline, epilogue):
    return (pipeline, epilogue, "intrawave", False, False, False, False)


class TestHelpers(unittest.TestCase):
    def test_reject_pipelines(self):
        for p in _BAD_PIPELINES:
            with self.assertRaisesRegex(ValueError, f"does not support the {p} "):
                mabd.reject_async_tdm_traits(p, "cshuffle")

    def test_reject_tdm_epilogue(self):
        with self.assertRaisesRegex(ValueError, "tdm epilogue"):
            mabd.reject_async_tdm_traits("compv3", "tdm")

    def test_legacy_traits_accepted(self):
        for p in ("compv3", "compv4", "mem"):
            for e in ("cshuffle", "default"):
                mabd.reject_async_tdm_traits(p, e)
                mabd.reject_async_tdm_trait_string(
                    f"{p}_{e}_intrawave_False_False_False_False"
                )

    def test_trait_string_multi_token(self):
        for p in _BAD_PIPELINES:
            with self.assertRaises(ValueError):
                mabd.reject_async_tdm_trait_string(
                    f"{p}_cshuffle_intrawave_False_False_False_False"
                )
        with self.assertRaises(ValueError):
            mabd.reject_async_tdm_trait_string(
                "compv3_tdm_intrawave_False_False_False_False"
            )


class TestBuilderRejects(unittest.TestCase):
    def test_generate_kernel_instance_rejects_on_every_arch(self):
        for arch in _ARCHES:
            with tempfile.TemporaryDirectory() as tmp:
                b = _make_builder(tmp, _load_config(), gpu_target=arch)
                for p in _BAD_PIPELINES:
                    for e in ("cshuffle", "tdm"):
                        with self.assertRaises(ValueError, msg=(arch, p, e)):
                            b._generate_kernel_instance(_TILE, _combo(p, e))
                with self.assertRaises(ValueError, msg=arch):
                    b._generate_kernel_instance(_TILE, _combo("compv3", "tdm"))
                self.assertEqual(os.listdir(tmp), ["config.json"])

    def test_populate_epilogue_rejects_tdm(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = _make_builder(tmp, _load_config())
            with self.assertRaises(ValueError):
                b.populate_epilogue("tdm")
            self.assertIn("CShuffleEpilogue", b.populate_epilogue("cshuffle"))

    def test_config_listing_rejects(self):
        for key, value in (
            ("pipeline", "comp_async"),
            ("pipeline", "comp_tdm"),
            ("pipeline", "comp_tdm_v2"),
            ("epilogue", "tdm"),
        ):
            cfg = copy.deepcopy(_load_config())
            cfg["trait_config"][key]["values"].append(value)
            with tempfile.TemporaryDirectory() as tmp:
                b = _make_builder(tmp, cfg, gpu_target="gfx1250")
                with self.assertRaises(ValueError, msg=value):
                    b._get_sampled_kernel_list()

    def test_legacy_config_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = _make_builder(tmp, _load_config())
            kernels = b._get_sampled_kernel_list()
            self.assertTrue(kernels)
            name, code = b._generate_kernel_instance(
                _TILE, _combo("compv3", "cshuffle")
            )
            self.assertIn("ck_tile::GemmPipelineAgBgCrCompV3", code)
            self.assertIn("CShuffleEpilogue", code)


class TestGenSingleCli(unittest.TestCase):
    def _run(self, tmp, trait):
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(_load_config(), f)
        return subprocess.run(
            [
                sys.executable,
                _BUILDER,
                "--working_path",
                tmp,
                "--gpu_target",
                "gfx1250",
                "--datatype",
                "fp16",
                "--layout",
                "rcrr",
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

    def test_gen_single_rejects_multi_token_pipeline(self):
        for p in _BAD_PIPELINES:
            with tempfile.TemporaryDirectory() as tmp:
                r = self._run(tmp, f"{p}_tdm_intrawave_False_False_False_False")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(f"does not support the {p} pipeline", r.stderr)

    def test_gen_single_rejects_tdm_epilogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(tmp, "compv3_tdm_intrawave_False_False_False_False")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("tdm epilogue", r.stderr)

    def test_gen_single_legacy_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(tmp, "compv3_cshuffle_intrawave_False_False_False_False")
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
