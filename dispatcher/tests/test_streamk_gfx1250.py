#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU regressions for gfx1250 Stream-K config selection and input verification.

GPU correctness is exercised by the native Stream-K tests and the bridge worker.
These tests cover the host-side boundaries that choose a valid kernel
and compare its output against the same quantized inputs the device receives.
"""

import contextlib
import io
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CK_ROOT = DISPATCHER_DIR.parent
for path in (DISPATCHER_DIR / "python", CK_ROOT / "tile_engine" / "ops" / "gemm"):
    sys.path.insert(0, str(path))

import gemm_utils  # noqa: E402
import run_one_streamk_gemm_kernel  # noqa: E402
import streamk_gemm_full_benchmark  # noqa: E402
from gemm_utils import _SUPPORTED_ARCHES, expand_sweep  # noqa: E402

GFX1250_CONFIG = (
    CK_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "gemm_streamk"
    / "configs"
    / "default_config_gfx1250.json"
)


def _write_reduction_config(tmp, reductions, name="config.json", **pinned):
    """Write the gfx1250 default config with a Stream-K reduction sweep into
    tmp/name; ``pinned`` replaces tile/trait value lists (e.g. tile_k=[32])."""
    with open(GFX1250_CONFIG) as f:
        cfg = json.load(f)
    cfg["streamk_config"] = {"reduction_strategy": {"values": reductions}}
    for key, values in pinned.items():
        section = "tile_config" if key in cfg["tile_config"] else "trait_config"
        cfg[section][key] = {"values": values}
    config_path = Path(tmp) / name
    config_path.write_text(json.dumps(cfg))
    return str(config_path)


class TestGfx1250Supported(unittest.TestCase):
    """gfx1250 must be a first-class supported arch."""

    def test_gfx1250_in_supported_arches(self):
        self.assertIn("gfx1250", _SUPPORTED_ARCHES)


class TestGfx1250StreamKConfig(unittest.TestCase):
    """The gfx1250 sweep uses WMMA 16x16x32 or 16x16x64 for its input type."""

    def setUp(self):
        self.assertTrue(
            GFX1250_CONFIG.exists(),
            f"missing gfx1250 stream-K config: {GFX1250_CONFIG}",
        )
        with open(GFX1250_CONFIG) as f:
            self.cfg = json.load(f)

    def test_config_uses_wmma_warp_tile(self):
        tc = self.cfg["tile_config"]
        # gfx1250 fp16/bf16 WMMA warp tile is 16x16x32; the MFMA 32x32x16 tile of
        # the default stream-K config is not a gfx1250 WMMA tile.
        self.assertEqual(tc["warp_tile_m"]["values"], [16])
        self.assertEqual(tc["warp_tile_n"]["values"], [16])
        self.assertEqual(tc["warp_tile_k"]["values"], [32, 64])

    def test_config_warp_combo_is_gfx1250_supported(self):
        # 2x2x1 (4-warp) is in the gfx1250 wave-combo list.
        tc = self.cfg["tile_config"]
        self.assertEqual(tc["warp_m"]["values"], [2])
        self.assertEqual(tc["warp_n"]["values"], [2])
        self.assertEqual(tc["warp_k"]["values"], [1])


class TestGfx1250StreamKExpansion(unittest.TestCase):
    """expand_sweep drives the whole gfx1250 stream-K host path (no GPU)."""

    def _expand(self, dtype="fp16"):
        return expand_sweep(
            str(GFX1250_CONFIG),
            "gfx1250",
            dtype=dtype,
            layout="rcr",
            variant="stream_k",
        )

    def test_expands_to_streamk_kernels_for_gfx1250(self):
        configs = self._expand("fp16")
        self.assertGreater(len(configs), 0)
        for c in configs:
            # arch is stamped concretely (never a silent gfx942 default).
            self.assertEqual(c.gfx_arch, "gfx1250")
            # stream-K identity preserved end-to-end.
            self.assertEqual(c.variant, "stream_k")
            self.assertTrue(c.name.endswith("_streamk"))
            # WMMA warp tile carried through into the kernel name.
            self.assertIn("16x16x32", c.name)

    def test_bf16_also_expands_for_gfx1250(self):
        configs = self._expand("bf16")
        self.assertGreater(len(configs), 0)
        for c in configs:
            self.assertEqual(c.gfx_arch, "gfx1250")
            self.assertTrue(c.name.endswith("_streamk"))

    def test_all_input_types_expand_to_supported_wmma_tiles(self):
        for dtype in ("fp16", "bf16", "fp8", "bf8"):
            with self.subTest(dtype=dtype):
                configs = self._expand(dtype)
                expected_k = 32 if dtype in ("fp16", "bf16") else 64
                self.assertEqual(len(configs), 4 if expected_k == 32 else 2)
                self.assertEqual({c.pipeline for c in configs}, {"compv3", "compv4"})
                for config in configs:
                    self.assertEqual(config.warp_tile_k, expected_k)
                    self.assertGreaterEqual(config.tile_k, expected_k)
                    self.assertEqual(
                        config.dtype_c, "bf16" if dtype == "bf16" else "fp16"
                    )

    def test_unsupported_arch_still_rejected(self):
        # The gfx1250 addition must not weaken the arch guard for bogus archs.
        with self.assertRaises(ValueError):
            expand_sweep(
                str(GFX1250_CONFIG),
                "gfx999",
                dtype="fp16",
                layout="rcr",
                variant="stream_k",
            )

    def test_k_block_divides_across_all_k_waves(self):
        with open(GFX1250_CONFIG) as f:
            cfg = json.load(f)
        # 1x2x2 is a gfx1250 warp layout with two K waves, so only the K rule
        # separates the tile_k values.
        cfg["tile_config"]["warp_m"]["values"] = [1]
        cfg["tile_config"]["warp_k"]["values"] = [2]
        cfg["tile_config"]["warp_tile_k"]["values"] = [32]
        cfg["tile_config"]["tile_k"]["values"] = [32, 64, 96, 128]
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(cfg))
            configs = expand_sweep(
                str(config_path),
                "gfx1250",
                dtype="fp16",
                layout="rcr",
                variant="stream_k",
            )
        self.assertEqual({c.tile_k for c in configs}, {64, 128})
        self.assertEqual(len(configs), 4)

    def test_reduction_override_replaces_config_strategies(self):
        # Each expanded candidate counts a reject for the illegal 0_0_3 widths, so
        # an override applied after the product (its copies collapsed by name)
        # still shows up in the rejects. Only that strategy-independent reason
        # (vector_size_c=3 is not a power of two) is compared: a width rule for
        # one strategy may rightly reject more of the full sweep.
        rejects = ({}, {})
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_reduction_config(tmp, ["atomic", "linear", "tree"])
            kwargs = dict(dtype="fp16", layout="rcr", variant="stream_k")
            kwargs["vector_sizes"] = [(0, 0, 0), (0, 0, 3)]
            swept = expand_sweep(config_path, "gfx1250", rejects=rejects[0], **kwargs)
            tree = expand_sweep(
                config_path,
                "gfx1250",
                reduction_strategies=["tree"],
                rejects=rejects[1],
                **kwargs,
            )
            single = expand_sweep(
                config_path, "gfx1250", reduction_strategies="tree", **kwargs
            )
        self.assertEqual([c.name for c in single], [c.name for c in tree])
        self.assertEqual(len(swept), 3 * len(tree))
        pow2 = [{k: n for k, n in r.items() if "power of two" in k} for r in rejects]
        self.assertTrue(pow2[1])
        self.assertEqual(pow2[0], {k: 3 * n for k, n in pow2[1].items()})
        self.assertEqual({c.reduction_strategy for c in tree}, {"tree"})
        self.assertEqual(len({c.name for c in tree}), len(tree))
        self.assertTrue(all(c.name.endswith("_streamk_tree") for c in tree))

    def test_repeated_sweep_values_expand_once(self):
        # expand_sweep collapses configs that share a kernel name.
        kwargs = dict(dtype="fp16", layout="rcr", variant="stream_k")
        with tempfile.TemporaryDirectory() as tmp:
            once, twice = [
                expand_sweep(
                    _write_reduction_config(tmp, ["atomic"], name, tile_k=tile_k),
                    "gfx1250",
                    **kwargs,
                )
                for name, tile_k in (("once.json", [32]), ("twice.json", [32, 32]))
            ]
        self.assertTrue(once)
        self.assertEqual([c.name for c in twice], [c.name for c in once])

    def test_unknown_reduction_strategy_is_rejected(self):
        kwargs = dict(dtype="fp16", layout="rcr", variant="stream_k")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_reduction_config(tmp, ["bogus"])
            for path, override in ((GFX1250_CONFIG, ["Linear"]), (config_path, None)):
                with self.subTest(override=override), self.assertRaisesRegex(
                    ValueError, "reduction strategy"
                ):
                    expand_sweep(
                        str(path), "gfx1250", reduction_strategies=override, **kwargs
                    )

    def test_gfx1250_streamk_compile_command_has_architecture_features(self):
        config = self._expand()[0]
        from dispatcher_common import arch_feature_defines, unified_framework_flags

        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            with patch.object(
                gemm_utils._cu, "get_build_dir", return_value=build
            ), patch.object(
                gemm_utils._cu,
                "get_generated_kernels_dir",
                return_value=build / "generated",
            ), patch.object(
                gemm_utils, "_resolve_hipcc", return_value="hipcc"
            ):
                job, _ = gemm_utils._build_compile_jobs(config, build / "kernel.hpp")
        command = job["compile_cmd"]
        self.assertIn("--offload-arch=gfx1250", command)
        self.assertIn("-DCK_TILE_USE_WMMA=1", command)
        self.assertIn("-DCK_USE_GFX1250", command)
        self.assertIn("-DCK_GFX1250_SUPPORT", command)
        for define in arch_feature_defines("gfx1250") + unified_framework_flags(
            "gfx1250"
        ):
            self.assertIn(define, command)
        self.assertTrue(
            any(arg.endswith("streamk_gemm_ctypes_lib.cpp") for arg in command)
        )
        self.assertFalse(
            any(arg.endswith("libck_tile_dispatcher.a") for arg in job["link_cmd"])
        )


class TestGfx1250StreamKDriver(unittest.TestCase):
    driver = streamk_gemm_full_benchmark
    worker = run_one_streamk_gemm_kernel

    def test_implicit_gfx1250_config_selects_wmma(self):
        args = SimpleNamespace(configs=[], arch="gfx1250")
        self.assertEqual(self.driver.resolve_configs(args), [str(GFX1250_CONFIG)])

    def test_explicit_config_is_preserved(self):
        args = SimpleNamespace(configs=["custom.json"], arch="gfx1250")
        self.assertEqual(self.driver.resolve_configs(args), ["custom.json"])

    def test_other_arch_default_is_preserved(self):
        args = SimpleNamespace(configs=[], arch="gfx942")
        self.assertEqual(
            self.driver.resolve_configs(args), [str(self.driver.DEFAULT_CONFIG)]
        )

    def _main_build_configs(self, *args):
        """Run main() with a mocked build; return the configs it would build and
        its stdout."""
        argv = ["streamk_gemm_full_benchmark.py", "--max-kernels", "1", *args]
        out = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(
            self.driver, "resolve_devices", return_value=["0"]
        ), patch.object(
            self.driver,
            "load_problems",
            return_value=[{"M": 128, "N": 128, "K": 33}],
        ), patch.object(
            self.driver,
            "setup_multiple_gemm_dispatchers",
            side_effect=lambda configs, **kwargs: [None] * len(configs),
        ) as build, contextlib.redirect_stdout(
            out
        ):
            self.assertEqual(self.driver.main(), 1)
        return build.call_args.args[0], out.getvalue()

    def test_detected_arch_selects_its_default_config(self):
        with patch.object(gemm_utils, "_get_arch", return_value="gfx1250") as detect:
            configs, out = self._main_build_configs()
        detect.assert_called_once_with()
        self.assertTrue(configs)
        self.assertEqual({c.gfx_arch for c in configs}, {"gfx1250"})
        self.assertIn(f"Configs: {GFX1250_CONFIG}", out)

    def test_reduction_override_preserves_limited_vector_fallback(self):
        for reduction in (None, "atomic", "linear", "tree"):
            with self.subTest(reduction=reduction):
                argv = ["--arch", "gfx1250"]
                if reduction is not None:
                    argv.extend(["--reduction-strategy", reduction])
                configs, _ = self._main_build_configs(*argv)
                self.assertEqual(sum(not any(c.vector_sizes) for c in configs), 1)
                self.assertTrue(any(any(c.vector_sizes) for c in configs))
                self.assertEqual(
                    {c.reduction_strategy for c in configs}, {reduction or "atomic"}
                )

    def _assert_tree_override_builds_distinct(self, *config_paths):
        """--reduction-strategy tree --max-kernels 3 must build 3 distinct kernels."""
        argv = ["--arch", "gfx1250", "--reduction-strategy", "tree"]
        argv += ["--max-kernels", "3"]
        configs, _ = self._main_build_configs(*config_paths, *argv)
        base = [c.name for c in configs if not any(c.vector_sizes)]
        self.assertEqual(len(set(base)), 3)
        self.assertEqual(len({c.name for c in configs}), len(configs))

    def test_reduction_override_keeps_kernel_budget_distinct(self):
        # Overriding a multi-strategy sweep must not spend --max-kernels on copies
        # of one kernel.
        with tempfile.TemporaryDirectory() as tmp:
            self._assert_tree_override_builds_distinct(
                _write_reduction_config(tmp, ["atomic", "linear", "tree"])
            )

    def test_reduction_override_keeps_kernel_budget_distinct_across_files(self):
        # Nor on copies across positional configs: overridden to tree, file A's
        # only (atomic) kernel is the first kernel of file B (linear).
        with tempfile.TemporaryDirectory() as tmp:
            self._assert_tree_override_builds_distinct(
                _write_reduction_config(
                    tmp, ["atomic"], "a.json", tile_k=[32], pipeline=["compv3"]
                ),
                _write_reduction_config(tmp, ["linear"], "b.json"),
            )

    def test_reduction_override_reaches_every_expansion(self):
        # The driver hands --reduction-strategy to each file's expand_sweep, so
        # every expansion already carries it. The driver's dedupe of the combined
        # list hides a strategy rewritten after expansion, in place or by
        # replace(), from the built configs; the spy records each file's
        # strategies when expand_sweep returns, so such a rewrite or a skipped
        # file shows here. Without the option each file keeps its own strategies.
        sweeps = (["atomic", "linear"], ["linear", "tree"])
        expansions = []

        def expand(*args, **kwargs):
            configs = expand_sweep(*args, **kwargs)
            expansions.append({c.reduction_strategy for c in configs})
            return configs

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.driver, "expand_sweep", side_effect=expand
        ):
            paths = [_write_reduction_config(tmp, r, f"{r[0]}.json") for r in sweeps]
            for n, override in ((1, None), (1, "tree"), (2, None), (2, "tree")):
                with self.subTest(files=n, override=override):
                    expansions.clear()
                    argv = ["--arch", "gfx1250", "--max-kernels", "0"]
                    argv += ["--reduction-strategy", override] if override else []
                    configs, _ = self._main_build_configs(*paths[:n], *argv)
                    want = [{override} if override else set(r) for r in sweeps[:n]]
                    self.assertEqual(
                        len(expansions), n, "one expand_sweep call per config file"
                    )
                    self.assertEqual(expansions, want)
                    self.assertEqual(
                        {c.reduction_strategy for c in configs}, set.union(*want)
                    )

    def test_arch_config_is_a_file_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            default = Path(tmp) / "default_config.json"
            arch_config = Path(tmp) / "default_config_gfx950.json"
            arch_config.write_text("{}")
            args = SimpleNamespace(configs=[], arch="gfx950")
            with patch.object(self.driver, "DEFAULT_CONFIG", default):
                self.assertEqual(self.driver.resolve_configs(args), [str(arch_config)])

    def test_ocp_reference_matches_device_input_quantization(self):
        # tiny is half the smallest OCP subnormal (E4M3 2^-9, E5M2 2^-16): OCP
        # rounds it to zero while FNUZ keeps it, so a reference that silently
        # encodes FNUZ must fail this comparison.
        for dtype, tiny in (("fp8", 2**-10), ("bf8", 2**-17)):
            for corrupt in (False, True):
                with self.subTest(dtype=dtype, corrupt=corrupt):
                    a = np.array([[tiny, 2 * tiny], [1, 0.5]], dtype=np.float32)
                    a_ocp = np.array([[0, 2 * tiny], [1, 0.5]], dtype=np.float32)
                    b = np.ones((2, 2), dtype=np.float32)
                    expected = a_ocp @ b
                    if corrupt:
                        expected = expected + 1
                    runner = object.__new__(gemm_utils.GpuGemmRunner)
                    runner._kernel_name = f"gemm_{dtype}_rcr_streamk"
                    runner._use_ocp = True
                    runner.run = lambda *args: SimpleNamespace(
                        success=True, output=expected, time_ms=1, tflops=1
                    )
                    output = io.StringIO()
                    # The worker draws A/B through _sweep_inputs; no cached
                    # operands or reference may enter or leave the subtest.
                    with patch.object(
                        self.worker, "GpuGemmRunner", return_value=runner
                    ), patch.object(
                        self.worker, "_sweep_inputs", return_value=(a, b)
                    ) as inputs, patch.dict(
                        gemm_utils._SWEEP_CACHE, clear=True
                    ), contextlib.redirect_stdout(
                        output
                    ):
                        self.worker._run_one(
                            0,
                            "unused.so",
                            {"M": 2, "N": 2, "K": 2},
                            runner.kernel_name,
                            True,
                            1e-8,
                        )
                    inputs.assert_called_once_with((2, 2), (2, 2))
                    self.assertEqual(
                        json.loads(output.getvalue())["verified"], not corrupt
                    )


if __name__ == "__main__":
    unittest.main()
