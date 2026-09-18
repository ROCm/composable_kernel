#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for python/dispatcher_common.py -- shared Python dispatcher utilities.

Phase 1b TDD: tests written BEFORE implementation exists.
Run: python3 -m pytest tests/test_dispatcher_common.py -v
"""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from dispatcher_common import (  # noqa: E402
    get_dispatcher_root,
    get_ck_root,
    get_build_dir,
    get_generated_kernels_dir,
    get_arch_filter_data,
    ValidationResultBase,
    validate_wave_config,
    validate_warp_tile_config,
    validate_trait_combo,
    auto_correct_wave,
    auto_correct_trait,
    Colors,
    print_phase,
    print_success,
    print_error,
    print_info,
    cleanup_generated_kernels,
    normalize_arch,
    fp8_uses_ocp,
    ocp_arch_defines,
    arch_feature_defines,
)


class TestPathHelpers(unittest.TestCase):
    """Tests for path helper functions."""

    def test_dispatcher_root_contains_codegen(self):
        root = get_dispatcher_root()
        self.assertTrue((root / "codegen").exists())

    def test_ck_root_contains_include_or_is_parent(self):
        root = get_ck_root()
        self.assertTrue(root.exists())
        self.assertEqual(root, get_dispatcher_root().parent)

    def test_build_dir_is_under_dispatcher(self):
        build = get_build_dir()
        self.assertEqual(build.parent, get_dispatcher_root())

    def test_generated_kernels_dir_under_build(self):
        gen_dir = get_generated_kernels_dir()
        self.assertEqual(gen_dir.parent, get_build_dir())


class TestGetArchFilterData(unittest.TestCase):
    """Tests for get_arch_filter_data."""

    def test_returns_dict(self):
        data = get_arch_filter_data()
        self.assertIsInstance(data, dict)

    def test_has_warp_combos(self):
        data = get_arch_filter_data()
        self.assertIn("warp_combos", data)

    def test_has_warp_tile_combos(self):
        data = get_arch_filter_data()
        self.assertIn("warp_tile_combos", data)

    def test_has_trait_unsupported(self):
        data = get_arch_filter_data()
        self.assertIn("trait_unsupported", data)

    def test_has_supported_archs(self):
        data = get_arch_filter_data()
        self.assertIn("supported_archs", data)
        self.assertIn("gfx942", data["supported_archs"])

    def test_gfx942_wave_configs(self):
        data = get_arch_filter_data()
        gfx942 = data["warp_combos"].get("gfx942", [])
        self.assertIn([2, 2, 1], gfx942)


class TestValidationResultBase(unittest.TestCase):
    """Tests for ValidationResultBase dataclass."""

    def test_valid_result(self):
        vr = ValidationResultBase(is_valid=True)
        self.assertTrue(vr.is_valid)
        self.assertEqual(vr.errors, [])
        self.assertEqual(vr.warnings, [])
        self.assertEqual(vr.suggested_fixes, {})

    def test_invalid_result(self):
        vr = ValidationResultBase(
            is_valid=False,
            errors=["bad wave"],
            suggested_fixes={"wave_m": 2},
        )
        self.assertFalse(vr.is_valid)
        self.assertEqual(len(vr.errors), 1)
        self.assertIn("wave_m", vr.suggested_fixes)


class TestValidateWaveConfig(unittest.TestCase):
    """Tests for validate_wave_config."""

    def test_valid_wave(self):
        is_valid, msg = validate_wave_config([2, 2, 1], "gfx942")
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_invalid_wave(self):
        is_valid, msg = validate_wave_config([3, 3, 1], "gfx942")
        self.assertFalse(is_valid)
        self.assertIn("wave", msg.lower())


class TestValidateWarpTileConfig(unittest.TestCase):
    """Tests for validate_warp_tile_config."""

    def test_valid_warp_tile(self):
        is_valid, msg = validate_warp_tile_config([32, 32, 16], "gfx942", "fp16")
        self.assertTrue(is_valid)

    def test_invalid_warp_tile(self):
        is_valid, msg = validate_warp_tile_config([99, 99, 99], "gfx942", "fp16")
        self.assertFalse(is_valid)
        self.assertIn("warp", msg.lower())


class TestValidateTraitCombo(unittest.TestCase):
    """Tests for validate_trait_combo."""

    def test_valid_trait(self):
        is_valid, msg = validate_trait_combo("compv3", "cshuffle", "intrawave")
        self.assertTrue(is_valid)

    def test_invalid_trait_interwave_compute(self):
        is_valid, msg = validate_trait_combo("compv4", "cshuffle", "interwave")
        self.assertFalse(is_valid)

    def test_valid_mem_interwave(self):
        is_valid, msg = validate_trait_combo("mem", "cshuffle", "interwave")
        self.assertTrue(is_valid)


class TestAutoCorrectWave(unittest.TestCase):
    """Tests for auto_correct_wave."""

    def test_corrects_invalid_wave(self):
        corrected = auto_correct_wave([1, 1, 1], "gfx942")
        self.assertIsInstance(corrected, list)
        self.assertEqual(len(corrected), 3)
        data = get_arch_filter_data()
        valid_waves = data["warp_combos"].get("gfx942", [[2, 2, 1]])
        self.assertIn(corrected, valid_waves)


class TestAutoCorrectTrait(unittest.TestCase):
    """Tests for auto_correct_trait."""

    def test_corrects_invalid_scheduler(self):
        corrected_pipeline, corrected_scheduler = auto_correct_trait(
            "compv4", "interwave"
        )
        self.assertEqual(corrected_scheduler, "intrawave")


class TestColors(unittest.TestCase):
    """Tests for Colors class (cross-platform ANSI support from conv)."""

    def test_green_returns_string(self):
        result = Colors.green("ok")
        self.assertIn("ok", result)

    def test_red_returns_string(self):
        result = Colors.red("error")
        self.assertIn("error", result)

    def test_yellow_returns_string(self):
        result = Colors.yellow("warn")
        self.assertIn("warn", result)

    def test_bold_returns_string(self):
        result = Colors.bold("title")
        self.assertIn("title", result)

    def test_plain_mode_no_ansi(self):
        with patch.object(Colors, "_use_color", return_value=False):
            result = Colors.green("plain")
            self.assertEqual(result, "plain")


class TestPhasedOutput(unittest.TestCase):
    """Tests for phased output helpers."""

    def test_print_phase(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_phase(1, "Setup")
        self.assertIn("Setup", buf.getvalue())

    def test_print_success(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_success("Done")
        self.assertIn("Done", buf.getvalue())

    def test_print_error(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_error("Oops")
        self.assertIn("Oops", buf.getvalue())

    def test_print_info(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_info("FYI")
        self.assertIn("FYI", buf.getvalue())


class TestCleanup(unittest.TestCase):
    """Tests for cleanup_generated_kernels."""

    def test_cleanup_nonexistent_dir_no_error(self):
        cleanup_generated_kernels(Path("/tmp/nonexistent_ck_test_dir_12345"))


class TestNormalizeArch(unittest.TestCase):
    """Tests for normalize_arch -- the front door for every arch predicate."""

    def test_bare_names_pass_through(self):
        for arch in ("gfx90a", "gfx942", "gfx950", "gfx1200", "gfx1250"):
            with self.subTest(arch=arch):
                self.assertEqual(normalize_arch(arch), arch)

    def test_feature_suffix_is_stripped(self):
        # What rocminfo and the HSA runtime actually hand back.
        self.assertEqual(normalize_arch("gfx950:sramecc+:xnack-"), "gfx950")
        self.assertEqual(normalize_arch("gfx942:sramecc+:xnack-"), "gfx942")
        self.assertEqual(normalize_arch("gfx90a:xnack+"), "gfx90a")

    def test_offload_target_prefix_is_stripped(self):
        # What --offload-arch round-trips through in some toolchain paths. This
        # is the form the old startswith() test missed outright.
        self.assertEqual(normalize_arch("amdgcn-amd-amdhsa--gfx950"), "gfx950")
        self.assertEqual(
            normalize_arch("amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-"), "gfx950"
        )

    def test_case_and_whitespace_are_normalized(self):
        self.assertEqual(normalize_arch("  GFX950  "), "gfx950")

    def test_none_and_empty_are_empty(self):
        self.assertEqual(normalize_arch(None), "")
        self.assertEqual(normalize_arch(""), "")


class TestFp8ArchPredicates(unittest.TestCase):
    """fp8_uses_ocp and the define lists derived from it.

    A miss here is silent: the -DCK_USE_OCP_FP8 pair simply does not get passed,
    the host pass of config.hpp falls back to FNUZ, and the kernel builds and
    runs while disagreeing with its own reference on what the bytes mean.
    """

    _OCP = ("gfx950", "gfx1200", "gfx1201", "gfx1250")
    _FNUZ = ("gfx90a", "gfx908", "gfx942", "gfx1100", "gfx1030")

    def test_ocp_arches(self):
        for arch in self._OCP:
            with self.subTest(arch=arch):
                self.assertTrue(fp8_uses_ocp(arch))

    def test_fnuz_arches(self):
        for arch in self._FNUZ:
            with self.subTest(arch=arch):
                self.assertFalse(fp8_uses_ocp(arch))

    def test_full_target_triples_agree_with_bare_names(self):
        for arch in self._OCP + self._FNUZ:
            triple = f"{arch}:sramecc+:xnack-"
            with self.subTest(arch=triple):
                self.assertEqual(fp8_uses_ocp(triple), fp8_uses_ocp(arch))

    def test_offload_target_form_agrees_with_bare_name(self):
        for arch in self._OCP + self._FNUZ:
            target = f"amdgcn-amd-amdhsa--{arch}"
            with self.subTest(arch=target):
                self.assertEqual(fp8_uses_ocp(target), fp8_uses_ocp(arch))

    def test_unknown_and_none_default_to_fnuz(self):
        # FNUZ is the compiler default for both passes, so it is also the safe
        # answer for an arch we do not recognize.
        for arch in (None, "", "not-a-gpu"):
            with self.subTest(arch=arch):
                self.assertFalse(fp8_uses_ocp(arch))

    def test_ocp_defines_are_emitted_as_a_pair(self):
        # Passing one without the other splits ck/ from ck_tile/.
        self.assertEqual(
            ocp_arch_defines("gfx950"),
            ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8"],
        )
        self.assertEqual(ocp_arch_defines("gfx942"), [])


class TestArchFeatureDefines(unittest.TestCase):
    """arch_feature_defines mirrors the top-level CMakeLists.txt arch block."""

    def test_gfx942_gets_no_feature_defines(self):
        self.assertEqual(arch_feature_defines("gfx942"), [])

    def test_gfx950_gets_ocp_plus_mx(self):
        self.assertEqual(
            arch_feature_defines("gfx950"),
            [
                "-DCK_USE_OCP_FP8",
                "-DCK_TILE_USE_OCP_FP8",
                "-DCK_USE_NATIVE_MX_SUPPORT",
                "-DCK_GFX950_SUPPORT",
            ],
        )

    def test_gfx12_enables_wmma(self):
        # CK_TILE_USE_WMMA must be passed explicitly: it selects the outermost
        # branch of get_k_warp_tile(), and leaving it undefined reads as 0 --
        # the MFMA branch, which no gfx12 part has.
        defines = arch_feature_defines("gfx1200")
        self.assertIn("-DCK_TILE_USE_WMMA=1", defines)
        self.assertIn("-DCK_GFX12_SUPPORT", defines)
        self.assertIn("-DCK_USE_OCP_FP8", defines)
        self.assertNotIn("-DCK_GFX950_SUPPORT", defines)

    def test_gfx1250_gets_its_own_defines_on_top_of_gfx12(self):
        defines = arch_feature_defines("gfx1250")
        for expected in (
            "-DCK_TILE_USE_WMMA=1",
            "-DCK_GFX12_SUPPORT",
            "-DCK_USE_GFX1250",
            "-DCK_USE_NATIVE_MX_SUPPORT",
            "-DCK_GFX1250_SUPPORT",
        ):
            with self.subTest(define=expected):
                self.assertIn(expected, defines)

    def test_gfx950_and_gfx12_defines_are_mutually_exclusive(self):
        # get_k_warp_tile() branches on these; both at once is not a state the
        # header is written for.
        for arch in ("gfx950", "gfx1250"):
            with self.subTest(arch=arch):
                defines = arch_feature_defines(arch)
                self.assertNotEqual(
                    "-DCK_GFX950_SUPPORT" in defines,
                    "-DCK_TILE_USE_WMMA=1" in defines,
                )

    def test_target_triple_gets_the_same_defines_as_the_bare_name(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            with self.subTest(arch=arch):
                self.assertEqual(
                    arch_feature_defines(f"{arch}:sramecc+:xnack-"),
                    arch_feature_defines(arch),
                )


if __name__ == "__main__":
    unittest.main()
