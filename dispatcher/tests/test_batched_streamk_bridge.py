#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for batched Stream-K and the batched comp_tdm_v3 / comp_tdm_pc
kernels in the TE to dispatcher bridge.

A batched sweep config with a ``streamk_config`` block expands to Stream-K
kernels (StreamKKernel folds the batch into its tile space), built into the
batched ctypes library under ``<Stream-K kernel name>_batched``.
batched_gemm_full_benchmark.py runs each of them once per workgroup budget.
No GPU, hipcc or dispatcher build is required.

Run: python3 -m pytest tests/test_batched_streamk_bridge.py -v
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
REPO_ROOT = DISPATCHER_DIR.parent
GEMM_TE_DIR = REPO_ROOT / "tile_engine" / "ops" / "gemm"
for p in (
    DISPATCHER_DIR / "python",
    DISPATCHER_DIR / "codegen",
    REPO_ROOT / "tile_engine" / "ops" / "common",
    GEMM_TE_DIR,
):
    sys.path.insert(0, str(p))

import batched_gemm_utils as bu  # noqa: E402
import gemm_utils as gu  # noqa: E402

CONFIG_DIR = GEMM_TE_DIR / "batched_gemm" / "configs"
STREAMK_CONFIGS = (
    "streamk_config.json",
    "streamk_config_gfx1250.json",
    "streamk_tdm_config_gfx1250.json",
    "streamk_narrow_m_config_gfx950.json",
    "streamk_narrow_n_config_gfx950.json",
)
TDM_PIPELINES = ("comp_tdm", "comp_tdm_v2", "comp_tdm_v3")


def _cfg(name):
    return str(CONFIG_DIR / name)


def _kernel(**overrides):
    kw = dict(
        dtype_a="bf16",
        dtype_b="bf16",
        dtype_c="bf16",
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
        gfx_arch="gfx950",
    )
    kw.update(overrides)
    return bu.BatchedGemmKernelConfig(**kw)


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "batched_gemm_full_benchmark", GEMM_TE_DIR / "batched_gemm_full_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBatchedStreamKName(unittest.TestCase):
    def test_streamk_so_name_is_the_kernel_name_plus_batched(self):
        for red in ("atomic", "linear", "tree"):
            sk = _kernel(variant="stream_k", reduction_strategy=red)
            single = gu.GemmKernelConfig(
                **{f: getattr(sk, f) for f in gu.GemmKernelConfig.__dataclass_fields__}
            )
            # The header is the codegen's Stream-K kernel; only the .so is batched.
            self.assertEqual(sk.header_stem, single.name)
            self.assertEqual(sk.name, single.name + "_batched")
            self.assertIn("_streamk", sk.name)
            self.assertEqual(sk.name.count("_batched"), 1)

    def test_batched_header_stem_is_its_name(self):
        k = _kernel()
        self.assertEqual(k.variant, "batched")
        self.assertEqual(k.header_stem, k.name)
        self.assertTrue(k.name.endswith("_batched"))


class TestConfigVariant(unittest.TestCase):
    def test_streamk_configs(self):
        for name in STREAMK_CONFIGS:
            self.assertEqual(bu.config_variant(_cfg(name)), "stream_k", name)

    def test_batched_configs(self):
        for name in ("default_config.json", "default_ci_config_gfx1250.json",
                     "tdm_config_gfx1250.json"):
            self.assertEqual(bu.config_variant(_cfg(name)), "batched", name)


class TestExpandBatchedStreamK(unittest.TestCase):
    def test_gfx950_universal_pipelines(self):
        for layout in bu.BATCHED_SUPPORTED_LAYOUTS:
            cfgs = bu.expand_sweep(_cfg("streamk_config.json"), "gfx950", "bf16", layout)
            self.assertTrue(cfgs, layout)
            self.assertEqual({c.variant for c in cfgs}, {"stream_k"})
            self.assertEqual({c.pipeline for c in cfgs}, {"compv3", "compv4", "mem"})
            self.assertEqual(
                {c.reduction_strategy for c in cfgs}, {"atomic", "linear", "tree"}
            )
            self.assertEqual({c.persistent for c in cfgs}, {False, True})
            self.assertTrue(all(c.pad_m and c.pad_n and c.pad_k for c in cfgs))
            self.assertTrue(all(c.name.endswith("_batched") for c in cfgs))

    def test_gfx950_narrow_blocks(self):
        # A 32-wide M or N block with the 16x16x32 warp tile (streamk_config.json: 64+).
        for half, narrow in (
            ("m", lambda c: c.tile_m == 32),
            ("n", lambda c: c.tile_n == 32 and c.tile_m >= 64),
        ):
            with self.subTest(half=half):
                cfgs = bu.expand_sweep(
                    _cfg(f"streamk_narrow_{half}_config_gfx950.json"), "gfx950", "bf16", "rcr"
                )
                self.assertTrue(cfgs)
                self.assertEqual([c.name for c in cfgs if not narrow(c)], [])
                self.assertEqual({c.variant for c in cfgs}, {"stream_k"})
                self.assertEqual(
                    {(c.warp_tile_m, c.warp_tile_n, c.warp_tile_k) for c in cfgs}, {(16, 16, 32)}
                )
                self.assertEqual({c.reduction_strategy for c in cfgs}, {"linear", "tree"})
                self.assertEqual({c.persistent for c in cfgs}, {False, True})
                self.assertTrue(all(c.pad_m and c.pad_n and c.pad_k for c in cfgs))

    def test_reduction_override(self):
        full = bu.expand_sweep(_cfg("streamk_config.json"), "gfx950", "bf16", "rcr")
        tree = bu.expand_sweep(
            _cfg("streamk_config.json"), "gfx950", "bf16", "rcr",
            reduction_strategies=["tree"],
        )
        self.assertEqual({c.reduction_strategy for c in tree}, {"tree"})
        self.assertEqual(3 * len(tree), len(full))

    def test_reduction_override_needs_a_streamk_config(self):
        with self.assertRaises(ValueError):
            bu.expand_sweep(
                _cfg("default_config.json"), "gfx950", "bf16", "rcr",
                reduction_strategies=["tree"],
            )

    def test_gfx1250_tdm_pipelines(self):
        cfgs = bu.expand_sweep(
            _cfg("streamk_tdm_config_gfx1250.json"), "gfx1250", "bf16", "rcr"
        )
        self.assertEqual({c.pipeline for c in cfgs}, set(TDM_PIPELINES))
        self.assertEqual({c.variant for c in cfgs}, {"stream_k"})
        self.assertFalse(any(c.pad_m or c.pad_n or c.pad_k for c in cfgs))
        # The TDM epilogue overwrites C, so atomic reduction keeps CShuffle.
        self.assertEqual(
            {c.epilogue for c in cfgs if c.reduction_strategy == "atomic"}, {"cshuffle"}
        )
        self.assertEqual(
            {c.epilogue for c in cfgs if c.reduction_strategy != "atomic"},
            {"cshuffle", "tdm"},
        )
        self.assertEqual({c.persistent for c in cfgs}, {False, True})

    def test_tdm_streamk_needs_gfx1250(self):
        self.assertEqual(
            bu.expand_sweep(_cfg("streamk_tdm_config_gfx1250.json"), "gfx950", "bf16", "rcr"),
            [],
        )


class TestExpandBatchedTdm(unittest.TestCase):
    def test_every_tdm_pipeline_is_batched(self):
        for layout in bu.BATCHED_SUPPORTED_LAYOUTS:
            cfgs = bu.expand_sweep(_cfg("tdm_config_gfx1250.json"), "gfx1250", "bf16", layout)
            self.assertEqual({c.variant for c in cfgs}, {"batched"})
            self.assertEqual(
                {c.pipeline for c in cfgs}, {*TDM_PIPELINES, "comp_tdm_pc"}, layout
            )
            for c in cfgs:
                want = "cshuffle" if c.pipeline == "comp_tdm_v3" else "tdm"
                self.assertEqual(c.epilogue, want, c.name)
                if c.pipeline == "comp_tdm_v3":
                    self.assertIn((c.wave_m, c.wave_n, c.wave_k), ((2, 2, 1), (4, 4, 1)))


class TestBatchedCompileFlags(unittest.TestCase):
    def _cmd(self, config):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            bu._cu, "get_build_dir", return_value=Path(tmp)
        ):
            job, lib = bu._build_batched_compile_jobs(config, Path(tmp) / "k.hpp")
        self.assertEqual(lib.name, f"lib{config.name}.so")
        self.assertTrue(job["compile_cmd"][-3].endswith("batched_gemm_ctypes_lib.cpp"))
        return job["compile_cmd"]

    def test_streamk_kernel_gets_the_streamk_flags(self):
        cmd = self._cmd(_kernel(variant="stream_k", reduction_strategy="linear"))
        self.assertIn("-enable-post-misched=0", cmd)
        self.assertIn("--lsr-drop-solution=1", cmd)

    def test_batched_kernel_keeps_the_tile_engine_flags(self):
        cmd = self._cmd(_kernel())
        flags = list(gu._tile_engine_codegen_flags())
        start = cmd.index(flags[0])
        self.assertEqual(cmd[start : start + len(flags)], flags)
        self.assertNotIn("-enable-post-misched=1", cmd)

    def test_comp_tdm_v3_turns_the_post_ra_scheduler_on(self):
        cmd = self._cmd(_kernel(pipeline="comp_tdm_v3", gfx_arch="gfx1250"))
        last = max(i for i, a in enumerate(cmd) if a.startswith("-enable-post-misched="))
        self.assertEqual(cmd[last], "-enable-post-misched=1")


class TestBatchedCtypesSource(unittest.TestCase):
    SRC = DISPATCHER_DIR / "bindings" / "ctypes" / "batched_gemm_ctypes_lib.cpp"

    def test_streamk_launch_path(self):
        text = self.SRC.read_text()
        self.assertIn("std::void_t<typename Kernel::StreamKGemmKernel>", text)
        self.assertIn("ck_tile::StreamKHostArgs args{", text)
        self.assertIn("k_batch must be 1", text)
        # The budget is parsed as strictly as in the single-problem Stream-K lib.
        self.assertIn('#include "streamk_budget_env.hpp"', text)
        self.assertIn("parse_streamk_budget_pct(budget_pct)", text)
        streamk_src = (self.SRC.parent / "streamk_gemm_ctypes_lib.cpp").read_text()
        self.assertIn('#include "streamk_budget_env.hpp"', streamk_src)
        budget_hdr = (self.SRC.parent / "streamk_budget_env.hpp").read_text()
        self.assertIn('std::getenv("CK_TILE_STREAMK_BUDGET_PCT")', budget_hdr)
        self.assertEqual(gu.STREAMK_BUDGET_ENV, "CK_TILE_STREAMK_BUDGET_PCT")

    def test_padded_leading_stride_validation_and_final_batch_extent(self):
        text = self.SRC.read_text()
        self.assertIn("const std::int64_t min_bs = (outer - 1) * s + inner;", text)
        self.assertIn("if(batch_count > 1 && bs < min_bs)", text)
        self.assertIn("static_cast<std::size_t>(batch_count - 1)", text)
        self.assertIn("__builtin_add_overflow(total_elems, static_cast<std::size_t>(min_bs)", text)


class TestStreamKCodegenReset(unittest.TestCase):
    def test_atomic_reset_covers_every_batch(self):
        cfg = _kernel(variant="stream_k", pad_m=True, pad_n=True, pad_k=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg.to_codegen_json()))
            subprocess.run(
                [
                    sys.executable,
                    str(DISPATCHER_DIR / "codegen" / "unified_gemm_codegen.py"),
                    "--output-dir", tmp, "--datatype", "bf16", "--layout", "rcr",
                    "--gpu-target", "gfx950", "--config", str(path),
                    "--variants", "stream_k", "--no-parallel",
                ],
                check=True,
                capture_output=True,
                cwd=str(DISPATCHER_DIR / "codegen"),
            )
            text = (Path(tmp) / f"{cfg.header_stem}.hpp").read_text()
        self.assertIn("args.batch_stride_E == static_cast<ck_tile::long_index_t>(args.M)", text)
        self.assertIn("packed_batches ? 1 : args.batch_count", text)
        self.assertIn("+ b * args.batch_stride_E", text)


DIMS = ("batch_count", "M", "N", "K")


class TestBatchedDriverUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drv = _load_driver()

    def test_budget_pct_parsing(self):
        import argparse

        parser = argparse.ArgumentParser()
        gu.add_streamk_args(parser)
        args = parser.parse_args(["--budget-pct", "100, 50,", "--reduction-strategy", "tree"])
        self.assertEqual(gu.streamk_budgets_arg(parser, args), [100, 50])
        self.assertEqual(args.reduction_strategy, "tree")
        self.assertEqual(gu.streamk_budgets_arg(parser, parser.parse_args([])), [100])
        for bad in ("0", "101", "", "50,x"):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                gu.streamk_budgets_arg(parser, parser.parse_args(["--budget-pct", bad]))

    def test_streamk_driver_shares_the_units(self):
        """The single-problem Stream-K driver runs each kernel batch once per budget."""
        kernels = [(_kernel(variant="stream_k"), Path(f"{i}.so")) for i in range(3)]
        prob = {"M": 64, "N": 64, "K": 64, "extra": 1}
        units = gu.streamk_work_units([prob], [[0, 1, 2]], kernels, 2, [100, 50], ("M", "N", "K"))
        self.assertEqual(
            [([i for i, _, _ in u[2]], u[3]) for u in units],
            [([0, 1], 100), ([0, 1], 50), ([2], 100), ([2], 50)],
        )
        self.assertEqual(units[0][1], {"M": 64, "N": 64, "K": 64})

    def test_budgets_apply_to_streamk_kernels_only(self):
        kernels = [
            (_kernel(variant="stream_k"), Path("a.so")),
            (_kernel(), Path("b.so")),
            (_kernel(variant="stream_k", reduction_strategy="tree"), Path("c.so")),
        ]
        prob = {"batch_count": 4, "M": 64, "N": 64, "K": 64}
        units = gu.streamk_work_units([prob], [[0, 1, 2]], kernels, 1, [100, 50], DIMS)
        got = sorted((u[2][0][0], u[3] or 0) for u in units)
        self.assertEqual(got, [(0, 50), (0, 100), (1, 0), (2, 50), (2, 100)])
        self.assertEqual({u[0] for u in units}, {0})

    def test_batches_keep_variants_apart(self):
        kernels = [(_kernel(variant="stream_k"), Path("a.so")), (_kernel(), Path("b.so"))]
        prob = {"batch_count": 2, "M": 8, "N": 8, "K": 8}
        units = gu.streamk_work_units([prob, prob], [[0, 1], [1]], kernels, 20, [100], DIMS)
        self.assertEqual(
            [(u[0], [i for i, _, _ in u[2]], u[3]) for u in units],
            [(0, [0], 100), (0, [1], None), (1, [1], None)],
        )

    def test_main_lists_mixed_configs(self):
        drv = self.drv
        seen = []

        def fake_run(device_id, unit, args, worker, env):
            seen.append(unit)
            return [], [], 0

        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / "libfake.so"
            lib.write_bytes(b"")
            argv = [
                "batched_gemm_full_benchmark.py",
                _cfg("default_ci_config.json"),
                _cfg("streamk_config.json"),
                "--arch", "gfx950", "--dtype", "bf16", "--budget-pct", "100,50",
                "--reduction-strategy", "linear", "--csv", str(Path(tmp) / "out.csv"),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                drv, "resolve_devices", lambda spec: ["0"]
            ), mock.patch.object(
                drv,
                "setup_multiple_batched_gemm_dispatchers",
                lambda cfgs, **kw: [Path(tmp) / f"{c.name}.so" for c in cfgs],
            ), mock.patch.object(drv, "_run_batch_on_device", fake_run):
                self.assertEqual(drv.main(), 0)
            header = (Path(tmp) / "out.csv").read_text().splitlines()[0]
        self.assertIn("budget_pct", header.split(","))
        sk = [u for u in seen if u[3] is not None]
        plain = [u for u in seen if u[3] is None]
        self.assertTrue(sk and plain)
        self.assertEqual({u[3] for u in sk}, {100, 50})
        for u in sk:
            self.assertTrue(all(c.variant == "stream_k" for _, c, _ in u[2]))
            self.assertTrue(all(c.reduction_strategy == "linear" for _, c, _ in u[2]))
        for u in plain:
            self.assertTrue(all(c.variant == "batched" for _, c, _ in u[2]))


if __name__ == "__main__":
    unittest.main()
