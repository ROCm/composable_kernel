#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for codegen/codegen_common.py -- shared infrastructure for GEMM and grouped conv codegen.

Phase 1a TDD: these tests are written BEFORE the implementation exists.
Run: python3 -m pytest tests/test_codegen_common.py -v
"""

import logging
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from codegen_common import (  # noqa: E402
    TileConfig,
    TraitConfigBase,
    CommonTypeMappings,
    generate_cpp_compilation_unit,
    parallel_generate,
    valid_wave_configs,
    valid_warp_configs,
    valid_trait_configs,
    needs_wave_expansion,
    needs_warp_expansion,
    needs_pipeline_expansion,
    # Block-scale quant kernel-name / epilogue contract (see TestQuant* below).
    bquant_effective_epilogue,
    aquant_effective_epilogue,
    abquant_effective_epilogue,
    gemm_aquant_effective_epilogue,
    make_bquant_kernel_name,
    make_gemm_rowcolquant_kernel_name,
    make_aquant_kernel_name,
    # Shared quant spec-sweep plumbing (see TestQuantSpecSweepHelpers).
    fp8_warp_tile_k_for_arch,
    iter_quant_axes,
    quant_decode_default_config,
    rcr_only_layout_guard,
    tile_config_from_dict,
    make_abquant_kernel_name,
    make_gemm_aquant_kernel_name,
    make_gemm_abquant_kernel_name,
)
from unified_gemm_tensor_quant_codegen import (  # noqa: E402
    make_tensor_quant_kernel_name,
    tensor_quant_effective_epilogue,
)


class TestTileConfig(unittest.TestCase):
    """TileConfig dataclass tests."""

    def test_valid_config(self):
        tc = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertTrue(tc.is_valid())

    def test_zero_tile_invalid(self):
        tc = TileConfig(0, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertFalse(tc.is_valid())

    def test_non_divisible_invalid(self):
        tc = TileConfig(127, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertFalse(tc.is_valid())

    def test_all_fields_accessible(self):
        tc = TileConfig(256, 128, 64, 4, 1, 1, 32, 32, 16)
        self.assertEqual(tc.tile_m, 256)
        self.assertEqual(tc.tile_n, 128)
        self.assertEqual(tc.tile_k, 64)
        self.assertEqual(tc.warp_m, 4)
        self.assertEqual(tc.warp_n, 1)
        self.assertEqual(tc.warp_k, 1)
        self.assertEqual(tc.warp_tile_m, 32)
        self.assertEqual(tc.warp_tile_n, 32)
        self.assertEqual(tc.warp_tile_k, 16)

    def test_small_valid_config(self):
        tc = TileConfig(16, 16, 16, 1, 1, 1, 16, 16, 16)
        self.assertTrue(tc.is_valid())


class TestTraitConfigBase(unittest.TestCase):
    """TraitConfigBase dataclass tests."""

    def test_valid_intrawave(self):
        tc = TraitConfigBase("compv3", "cshuffle", "intrawave", False, False, False)
        self.assertTrue(tc.is_valid())

    def test_invalid_interwave_compv3(self):
        tc = TraitConfigBase("compv3", "cshuffle", "interwave", False, False, False)
        self.assertFalse(tc.is_valid())

    def test_invalid_interwave_compv4(self):
        tc = TraitConfigBase("compv4", "cshuffle", "interwave", False, False, False)
        self.assertFalse(tc.is_valid())

    def test_valid_mem_interwave(self):
        tc = TraitConfigBase("mem", "cshuffle", "interwave", False, False, False)
        self.assertTrue(tc.is_valid())

    def test_valid_mem_intrawave(self):
        tc = TraitConfigBase("mem", "cshuffle", "intrawave", False, False, False)
        self.assertTrue(tc.is_valid())

    def test_padding_fields(self):
        tc = TraitConfigBase("compv3", "cshuffle", "intrawave", True, True, True)
        self.assertTrue(tc.pad_m)
        self.assertTrue(tc.pad_n)
        self.assertTrue(tc.pad_k)


class TestCommonTypeMappings(unittest.TestCase):
    """CommonTypeMappings tests."""

    def test_dtype_to_ck(self):
        self.assertEqual(CommonTypeMappings.DTYPE_TO_CK["fp16"], "fp16_t")
        self.assertEqual(CommonTypeMappings.DTYPE_TO_CK["bf16"], "bf16_t")
        self.assertEqual(CommonTypeMappings.DTYPE_TO_CK["fp32"], "float")
        self.assertEqual(CommonTypeMappings.DTYPE_TO_CK["fp8"], "fp8_t")

    def test_pipeline_to_ck(self):
        self.assertEqual(
            CommonTypeMappings.PIPELINE_TO_CK["mem"], "GemmPipelineAgBgCrMem"
        )
        self.assertIn("compv3", CommonTypeMappings.PIPELINE_TO_CK)
        self.assertIn("compv4", CommonTypeMappings.PIPELINE_TO_CK)

    def test_pipeline_to_base(self):
        self.assertIn("mem", CommonTypeMappings.PIPELINE_TO_BASE)
        self.assertIn("compv3", CommonTypeMappings.PIPELINE_TO_BASE)
        self.assertIn("compv4", CommonTypeMappings.PIPELINE_TO_BASE)

    def test_scheduler_to_ck(self):
        self.assertIn("intrawave", CommonTypeMappings.SCHEDULER_TO_CK)
        self.assertIn("interwave", CommonTypeMappings.SCHEDULER_TO_CK)

    def test_epilogue_to_dispatcher(self):
        self.assertIn("cshuffle", CommonTypeMappings.EPILOGUE_TO_DISPATCHER)
        self.assertIn("default", CommonTypeMappings.EPILOGUE_TO_DISPATCHER)

    def test_layout_to_ck(self):
        self.assertIn("r", CommonTypeMappings.LAYOUT_TO_CK)
        self.assertIn("c", CommonTypeMappings.LAYOUT_TO_CK)

    def test_get_output_dtype(self):
        self.assertEqual(CommonTypeMappings.get_output_dtype("fp8"), "fp16")
        self.assertEqual(CommonTypeMappings.get_output_dtype("bf8"), "fp16")
        self.assertEqual(CommonTypeMappings.get_output_dtype("fp16"), "fp16")
        self.assertEqual(CommonTypeMappings.get_output_dtype("fp32"), "fp32")


class TestGenerateCppCompilationUnit(unittest.TestCase):
    """Tests for generate_cpp_compilation_unit."""

    def test_includes_kernel_header(self):
        result = generate_cpp_compilation_unit("my_kernel")
        self.assertIn('#include "my_kernel.hpp"', result)

    def test_contains_pragma_once_or_guard(self):
        result = generate_cpp_compilation_unit("test_kernel")
        self.assertIn("test_kernel", result)

    def test_different_names_different_output(self):
        a = generate_cpp_compilation_unit("kernel_a")
        b = generate_cpp_compilation_unit("kernel_b")
        self.assertNotEqual(a, b)


class TestParallelGenerate(unittest.TestCase):
    """Tests for parallel_generate helper."""

    def _dummy_generate(self, item):
        return f"generated_{item}"

    def test_parallel_returns_all(self):
        items = ["a", "b", "c", "d"]
        results = parallel_generate(self._dummy_generate, items, parallel=True)
        self.assertEqual(len(results), 4)
        for item in items:
            self.assertIn(f"generated_{item}", results)

    def test_sequential_returns_all(self):
        items = ["x", "y", "z"]
        results = parallel_generate(self._dummy_generate, items, parallel=False)
        self.assertEqual(len(results), 3)
        for item in items:
            self.assertIn(f"generated_{item}", results)

    def test_empty_items(self):
        results = parallel_generate(self._dummy_generate, [], parallel=True)
        self.assertEqual(len(results), 0)

    def test_logs_per_kernel_progress(self):
        items = ["k1", "k2"]
        with self.assertLogs(level="INFO") as cm:
            parallel_generate(self._dummy_generate, items, parallel=False)
        log_output = "\n".join(cm.output)
        self.assertIn("k1", log_output)
        self.assertIn("k2", log_output)


class TestArchAwareExpansion(unittest.TestCase):
    """Tests for arch-aware expansion helpers (best-of-conv)."""

    def test_valid_wave_configs_gfx942(self):
        configs = valid_wave_configs("gfx942")
        self.assertIsInstance(configs, list)
        self.assertIn([2, 2, 1], configs)
        self.assertIn([1, 4, 1], configs)

    def test_valid_wave_configs_unknown_arch(self):
        configs = valid_wave_configs("gfx_unknown")
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)

    def test_valid_warp_configs_gfx942_fp16(self):
        configs = valid_warp_configs("gfx942", "fp16")
        self.assertIsInstance(configs, list)
        self.assertIn([32, 32, 16], configs)

    def test_valid_warp_configs_unknown_arch(self):
        configs = valid_warp_configs("gfx_unknown", "fp16")
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)

    def test_valid_trait_configs_excludes_interwave_compute(self):
        configs = valid_trait_configs()
        self.assertIsInstance(configs, list)
        self.assertNotIn(("compv3", "cshuffle", "interwave"), configs)
        self.assertNotIn(("compv4", "cshuffle", "interwave"), configs)

    def test_valid_trait_configs_includes_mem_interwave(self):
        configs = valid_trait_configs()
        has_mem_interwave = any(p == "mem" and s == "interwave" for p, s in configs)
        self.assertTrue(has_mem_interwave)

    def test_needs_wave_expansion_wildcard(self):
        self.assertTrue(needs_wave_expansion({"wave_m": -1, "wave_n": 2}))
        self.assertTrue(needs_wave_expansion({"wave_m": 2, "wave_n": -1}))

    def test_needs_wave_expansion_explicit(self):
        self.assertFalse(needs_wave_expansion({"wave_m": 2, "wave_n": 2}))

    def test_needs_warp_expansion_wildcard(self):
        self.assertTrue(needs_warp_expansion({"warp_m": -1, "warp_n": 32}))

    def test_needs_warp_expansion_explicit(self):
        self.assertFalse(needs_warp_expansion({"warp_m": 32, "warp_n": 32}))

    def test_needs_pipeline_expansion_wildcard(self):
        self.assertTrue(needs_pipeline_expansion({"pipeline": "*"}))

    def test_needs_pipeline_expansion_explicit(self):
        self.assertFalse(needs_pipeline_expansion({"pipeline": "compv4"}))


# =============================================================================
# Block-scale quant: kernel-name and epilogue-selection contract
# =============================================================================
#
# CHARACTERIZATION TESTS. These pin the CURRENT output of the quant kernel-name
# builders and epilogue selectors, byte-exact. They exist so the codegen dedup
# refactor can be proven behaviour-preserving.
#
# Why byte-exact matters: the name a generator emits as KERNEL_NAME is the same
# string the runtime utils rebuild to locate the compiled .so. The two sides
# share these helpers precisely so they cannot drift -- and the gap these tests
# fill has already let one real bug through (see the shadowing NOTE at
# codegen_common.py:662).
#
# A failure here is NOT automatically a bug in the test: it means a name or an
# epilogue choice changed. That is only correct if it was changed deliberately,
# and it must then be matched on the runtime side in dispatcher/python/*_utils.py.

# One representative tile shared by every name test: N_Repeat = 128/(4*16) = 2,
# i.e. EVEN, so the permute_n parity condition is satisfiable and the tests can
# distinguish "cshuffle because parity failed" from "cshuffle unconditionally".
_TILE = dict(
    tile_m=128, tile_n=128, tile_k=128,
    warp_m=1, warp_n=4, warp_k=1,
    warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
)
_ARGS = ("fp8", "rcr", "compv3", "cshuffle", "intrawave")


class TestQuantEffectiveEpilogue(unittest.TestCase):
    """The permute_n vs cshuffle selection rule, per operator family."""

    # -- bquant: gated on preshuffle_b AND even N_Repeat AND quant_group_n == 1 --

    def test_bquant_permute_n_when_all_conditions_hold(self):
        self.assertEqual(bquant_effective_epilogue(128, 4, 16, 1, True), "permute_n")

    def test_bquant_cshuffle_without_preshuffle_b(self):
        # The preshuffle_b gate is load-bearing: omitting it once made MX kernels
        # emit a PermuteN epilogue ~16-17% slower than the CShuffle they wanted.
        self.assertEqual(bquant_effective_epilogue(128, 4, 16, 1, False), "cshuffle")

    def test_bquant_cshuffle_when_quant_group_n_gt_1(self):
        self.assertEqual(bquant_effective_epilogue(128, 4, 16, 8, True), "cshuffle")

    def test_bquant_cshuffle_when_n_repeat_odd(self):
        self.assertEqual(bquant_effective_epilogue(64, 4, 16, 1, True), "cshuffle")

    # -- aquant (grouped): same rule but with NO preshuffle_b gate --

    def test_aquant_permute_n_on_even_n_repeat(self):
        self.assertEqual(aquant_effective_epilogue(128, 4, 16, 1), "permute_n")

    def test_aquant_cshuffle_when_n_repeat_odd(self):
        self.assertEqual(aquant_effective_epilogue(64, 4, 16, 1), "cshuffle")

    def test_aquant_cshuffle_when_quant_group_n_gt_1(self):
        self.assertEqual(aquant_effective_epilogue(128, 4, 16, 8), "cshuffle")

    # -- abquant (grouped): as aquant, plus two pipelines forced to cshuffle --

    def test_abquant_permute_n_on_compv3(self):
        self.assertEqual(abquant_effective_epilogue(128, 4, 16, 1, "compv3"), "permute_n")

    def test_abquant_eightwaves_forces_cshuffle(self):
        # TransposeC=true is incompatible with PermuteNEpilogue.
        self.assertEqual(abquant_effective_epilogue(128, 4, 16, 1, "eightwaves"), "cshuffle")

    def test_abquant_preshuffleb_forces_cshuffle(self):
        self.assertEqual(abquant_effective_epilogue(128, 4, 16, 1, "preshuffleb"), "cshuffle")

    # -- the two unconditional-cshuffle selectors --

    def test_gemm_aquant_epilogue_is_always_cshuffle(self):
        # Pins current behaviour: this helper ignores all four arguments. Even
        # with parity satisfied and quant_group_n == 1 it returns cshuffle.
        self.assertEqual(gemm_aquant_effective_epilogue(128, 4, 16, 1), "cshuffle")

    def test_tensor_quant_epilogue_is_always_cshuffle(self):
        self.assertEqual(tensor_quant_effective_epilogue(128, 4, 16), "cshuffle")


class TestQuantKernelNames(unittest.TestCase):
    """Byte-exact kernel names. These strings are a cross-layer contract."""

    def test_tensor_quant(self):
        self.assertEqual(
            make_tensor_quant_kernel_name(*_ARGS, **_TILE),
            "gemm_tensor_quant_fp8_rcr_compv3_cshuffle_intrawave"
            "_128x128x128_1x4x1_16x16x32",
        )

    def test_rowcolquant(self):
        # RowColQuant has no quant-group segment: scales are per-row / per-col
        # vectors, not blocks.
        self.assertEqual(
            make_gemm_rowcolquant_kernel_name(*_ARGS, **_TILE),
            "gemm_rowcolquant_fp8_rcr_compv3_cshuffle_intrawave"
            "_128x128x128_1x4x1_16x16x32",
        )

    def test_gemm_aquant_with_preshuffle(self):
        # Note the flag spells "preshufflequant", not "preshuffleaq" as the
        # grouped variant does.
        self.assertEqual(
            make_gemm_aquant_kernel_name(
                *_ARGS, **_TILE,
                quant_group_m=1, quant_group_n=1, quant_group_k=128,
                preshuffle_aquant=True,
            ),
            "gemm_aquant_fp8_rcr_compv3_cshuffle_intrawave"
            "_128x128x128_1x4x1_16x16x32_qg1x1x128_preshufflequant",
        )

    def test_gemm_abquant_preshuffle_b(self):
        self.assertEqual(
            make_gemm_abquant_kernel_name(
                *_ARGS, **_TILE,
                aquant_group_k=128, bquant_group_n=1, bquant_group_k=128,
                preshuffle_b=True, preshuffle_bquant=True, eight_waves=False,
            ),
            "gemm_abquant_fp8_rcr_compv3_cshuffle_intrawave"
            "_128x128x128_1x4x1_16x16x32_aqg1x1x128_bqg1x1x128"
            "_preshuffleb_preshufflebq",
        )

    def test_gemm_abquant_never_emits_permute_n(self):
        # Pins a surprise. make_gemm_abquant_kernel_name has an
        # `if preshuffle_b and not eight_waves` branch that calls
        # bquant_effective_epilogue -- but WITHOUT forwarding preshuffle_b, so
        # that helper sees its default False and returns cshuffle. Both arms of
        # the branch therefore yield cshuffle and permute_n is unreachable,
        # even here where parity holds and bquant_group_n == 1.
        for preshuffle_b, eight_waves in ((True, False), (False, False), (True, True)):
            name = make_gemm_abquant_kernel_name(
                *_ARGS, **_TILE,
                aquant_group_k=128, bquant_group_n=1, bquant_group_k=128,
                preshuffle_b=preshuffle_b, eight_waves=eight_waves,
            )
            self.assertIn("_cshuffle_", name)
            self.assertNotIn("permute_n", name)

    def test_gemm_abquant_eight_waves(self):
        self.assertEqual(
            make_gemm_abquant_kernel_name(
                "fp8", "rcr", "eightwaves", "cshuffle", "intrawave", **_TILE,
                aquant_group_k=128, bquant_group_n=1, bquant_group_k=128,
                preshuffle_b=True, eight_waves=True,
            ),
            "gemm_abquant_fp8_rcr_eightwaves_cshuffle_intrawave"
            "_128x128x128_1x4x1_16x16x32_aqg1x1x128_bqg1x1x128"
            "_preshuffleb_eightwaves",
        )

    def test_bquant_plain_prefix_emits_permute_n(self):
        # name_prefix="gemm_bquant" selects the non-grouped family. preshuffle_b
        # is forwarded here (unlike gemm_abquant above), so permute_n is live.
        self.assertEqual(
            make_bquant_kernel_name(
                *_ARGS, **_TILE,
                quant_group_m=1, quant_group_n=1, quant_group_k=128,
                preshuffle_b=True, preshuffle_bquant=True,
                name_prefix="gemm_bquant",
            ),
            "gemm_bquant_fp8_rcr_compv3_permute_n_intrawave"
            "_128x128x128_1x4x1_16x16x32_qg1x1x128_preshuffleb_preshufflebq",
        )

    def test_bquant_default_prefix_is_grouped(self):
        # The default name_prefix is the GROUPED family, which is easy to get
        # wrong: callers that forget it silently emit a grouped name.
        self.assertEqual(
            make_bquant_kernel_name(
                *_ARGS, **_TILE,
                quant_group_m=1, quant_group_n=1, quant_group_k=128,
            ),
            "grouped_gemm_bquant_fp8_rcr_compv3_cshuffle_intrawave"
            "_128x128x128_1x4x1_16x16x32_qg1x1x128",
        )

    def test_grouped_aquant(self):
        self.assertEqual(
            make_aquant_kernel_name(
                *_ARGS, **_TILE,
                quant_group_m=1, quant_group_n=1, quant_group_k=128,
                preshuffle_aq=True,
            ),
            "grouped_gemm_aquant_fp8_rcr_compv3_permute_n_intrawave"
            "_128x128x128_1x4x1_16x16x32_aqg1x1x128_preshuffleaq",
        )

    def test_grouped_abquant_all_flags(self):
        # Flag order is part of the contract: b, aq, bq, transposec.
        self.assertEqual(
            make_abquant_kernel_name(
                *_ARGS, **_TILE,
                aquant_group_m=1, aquant_group_n=1, aquant_group_k=128,
                bquant_group_m=1, bquant_group_n=1, bquant_group_k=128,
                preshuffle_b=True, preshuffle_aq=True, preshuffle_bq=True,
                transpose_c=True,
            ),
            "grouped_gemm_abquant_fp8_rcr_compv3_permute_n_intrawave"
            "_128x128x128_1x4x1_16x16x32_aqg1x1x128_bqg1x1x128"
            "_preshuffleb_preshuffleaq_preshufflebq_transposec",
        )


class TestQuantKernelNameInvariants(unittest.TestCase):
    """Properties that must hold for every quant name builder."""

    def _all_builders(self):
        return [
            ("tensor_quant", lambda **t: make_tensor_quant_kernel_name(*_ARGS, **t)),
            ("rowcolquant", lambda **t: make_gemm_rowcolquant_kernel_name(*_ARGS, **t)),
            ("gemm_aquant", lambda **t: make_gemm_aquant_kernel_name(
                *_ARGS, **t, quant_group_m=1, quant_group_n=1, quant_group_k=128)),
            ("gemm_abquant", lambda **t: make_gemm_abquant_kernel_name(
                *_ARGS, **t, aquant_group_k=128, bquant_group_n=1, bquant_group_k=128)),
            ("bquant", lambda **t: make_bquant_kernel_name(
                *_ARGS, **t, quant_group_m=1, quant_group_n=1, quant_group_k=128)),
            ("grouped_aquant", lambda **t: make_aquant_kernel_name(
                *_ARGS, **t, quant_group_m=1, quant_group_n=1, quant_group_k=128)),
            ("grouped_abquant", lambda **t: make_abquant_kernel_name(
                *_ARGS, **t,
                aquant_group_m=1, aquant_group_n=1, aquant_group_k=128,
                bquant_group_m=1, bquant_group_n=1, bquant_group_k=128)),
        ]

    def test_names_are_filename_safe(self):
        # Names become .hpp filenames and .so basenames.
        for label, fn in self._all_builders():
            name = fn(**_TILE)
            self.assertRegex(name, r"^[a-z0-9_]+$", f"{label}: {name!r} not filename-safe")

    def test_tile_shape_is_encoded(self):
        # The tile triple must survive into the name, or two differently-shaped
        # kernels collide on one filename and one silently overwrites the other.
        for label, fn in self._all_builders():
            self.assertIn("128x128x128", fn(**_TILE), label)

    def test_warp_tile_k_distinguishes_names(self):
        # WarpTileK is the gfx942(32) / gfx950(128) arch trap: the wrong value
        # silently all-zeros the output, so it MUST be part of the identity.
        alt = dict(_TILE, warp_tile_k=128)
        for label, fn in self._all_builders():
            self.assertNotEqual(fn(**_TILE), fn(**alt), f"{label}: warp_tile_k not in name")

    def test_builders_are_deterministic(self):
        for label, fn in self._all_builders():
            self.assertEqual(fn(**_TILE), fn(**_TILE), label)


class TestQuantSpecSweepHelpers(unittest.TestCase):
    """Contracts for the shared _build_specs()/_default_config() plumbing.

    The generated-header A/B gate catches a sweep point that starts or stops
    being emitted, but it cannot see *why* one was skipped. These pin the guard
    order and the skip diagnostics, which are the operator's only signal that a
    config axis was silently dropped.
    """

    VARIANTS = {"fp8": {}, "bf8": {}}
    PIPELINES = {"compv3": "X"}

    def _cfg(self, **over):
        cfg = {
            "variant_keys": ["fp8"],
            "layouts": ["rcr"],
            "tile_configs": [dict(_TILE)],
        }
        cfg.update(over)
        return cfg

    def _axes(self, cfg, **kw):
        kw.setdefault("variants", self.VARIANTS)
        kw.setdefault("logger", logging.getLogger("test_quant_axes"))
        return list(iter_quant_axes(cfg, **kw))

    def test_yields_tile_config_and_empty_extra_without_extra_axis(self):
        (variant, layout, tile, extra), = self._axes(self._cfg())
        self.assertEqual((variant, layout, extra), ("fp8", "rcr", {}))
        self.assertIsInstance(tile, TileConfig)
        self.assertEqual(tile.warp_tile_k, _TILE["warp_tile_k"])

    def test_extra_axis_default_is_used_when_key_absent(self):
        default = [{"quant_group_k": 128}]
        (_, _, _, extra), = self._axes(
            self._cfg(), extra_axis=("quant_groups", default)
        )
        self.assertEqual(extra, default[0])

    def test_unknown_variant_is_skipped_with_a_warning(self):
        with self.assertLogs("test_quant_axes", level="WARNING") as cm:
            self.assertEqual(self._axes(self._cfg(variant_keys=["int4"])), [])
        self.assertIn("Unknown variant_key int4", "\n".join(cm.output))

    def test_unsupported_pipeline_is_skipped_only_when_a_map_is_given(self):
        cfg = self._cfg()
        # No pipeline_map (AQuant) -> the pipeline axis is not policed at all.
        self.assertEqual(len(self._axes(cfg, pipeline="nonsense")), 1)
        with self.assertLogs("test_quant_axes", level="WARNING"):
            self.assertEqual(
                self._axes(cfg, pipeline="nonsense", pipeline_map=self.PIPELINES), []
            )

    def test_rcr_only_layout_guard(self):
        self.assertIsNone(rcr_only_layout_guard("rcr"))
        self.assertIn("only rcr", rcr_only_layout_guard("ccr"))
        with self.assertLogs("test_quant_axes", level="WARNING"):
            self.assertEqual(
                self._axes(self._cfg(layouts=["ccr"]), layout_guard=rcr_only_layout_guard),
                [],
            )

    def test_invalid_tile_is_dropped_without_a_warning(self):
        # tile_n=64 is not divisible by warp_n*warp_tile_n=4*32.
        bad = dict(_TILE, tile_n=64, warp_n=4, warp_tile_n=32)
        self.assertEqual(self._axes(self._cfg(tile_configs=[bad])), [])

    def test_guard_order_is_variant_then_pipeline_then_layout(self):
        # A config that trips all three must report the *first* failure, so the
        # operator fixes the outermost problem rather than chasing a symptom.
        cfg = self._cfg(variant_keys=["int4"], layouts=["ccr"])
        with self.assertLogs("test_quant_axes", level="WARNING") as cm:
            self._axes(cfg, pipeline="nonsense", pipeline_map=self.PIPELINES,
                       layout_guard=rcr_only_layout_guard)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("Unknown variant_key", cm.output[0])

    def test_tile_config_from_dict_requires_every_key(self):
        with self.assertRaises(KeyError):
            tile_config_from_dict({k: v for k, v in _TILE.items()
                                   if k != "warp_tile_k"})

    def test_decode_default_config_shape(self):
        cfg = quant_decode_default_config(warp_tile_k=128)
        self.assertEqual(cfg["variant_keys"], ["fp8", "bf8"])
        self.assertEqual(cfg["layouts"], ["rcr"])
        self.assertEqual(cfg["pad_k"], True)
        tile, = cfg["tile_configs"]
        self.assertEqual((tile["tile_m"], tile["tile_n"], tile["tile_k"]), (16, 64, 256))
        self.assertEqual(tile["warp_tile_k"], 128)

    def test_decode_default_config_overrides_and_isolation(self):
        cfg = quant_decode_default_config(warp_tile_k=32, preshuffle_b=True, pad_k=False)
        self.assertTrue(cfg["preshuffle_b"])
        self.assertFalse(cfg["pad_k"])
        # Each call must own its nested containers, or one operator's sweep
        # mutates another's defaults.
        other = quant_decode_default_config(warp_tile_k=32)
        cfg["tile_configs"][0]["tile_m"] = 999
        self.assertEqual(other["tile_configs"][0]["tile_m"], 16)


class TestArchWarpTileK(unittest.TestCase):
    """The gfx942/gfx950 WarpTileK rule -- a silent-wrong-answer trap."""

    def test_gfx950_is_128_regardless_of_preshuffle(self):
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx950"), 128)
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx950", preshuffle_quant=True), 128)

    def test_gfx942_is_32_plain_and_64_preshuffle(self):
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx942"), 32)
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx942", preshuffle_quant=True), 64)

    def test_unknown_arch_takes_the_safe_non_gfx950_branch(self):
        # 128 on a non-gfx950 target compiles and then emits all zeros, so an
        # unrecognised arch must never fall through to it.
        self.assertEqual(fp8_warp_tile_k_for_arch("gfx90a"), 32)


if __name__ == "__main__":
    unittest.main()
