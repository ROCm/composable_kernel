#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for python/dispatcher_common.py -- shared Python dispatcher utilities.

Phase 1b TDD: tests written BEFORE implementation exists.
Run: python3 -m pytest tests/test_dispatcher_common.py -v
"""

import ctypes
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

import dispatcher_common  # noqa: E402
from dispatcher_common import (  # noqa: E402
    _hip_error_text,
    _hip_runtime_once,
    hip_device_problem,
    hip_last_error,
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
    unified_framework_flags,
)


class TestUnifiedFrameworkFlags(unittest.TestCase):
    """Only gfx1250 kernel builds should override the framework default."""

    def test_gfx1250_disables_unified_framework(self):
        for arch in (
            "gfx1250",
            "gfx1250:xnack-",
            "gfx1250:sramecc+:xnack-",
            " GFX1250 ",
            "amdgcn-amd-amdhsa--gfx1250",
        ):
            with self.subTest(arch=arch):
                self.assertEqual(
                    unified_framework_flags(arch), ["-DUSE_NEW_UNIFIED_FRAMEWORK=0"]
                )

    def test_other_architectures_keep_framework_default(self):
        for arch in ("gfx942", "gfx950", "gfx1200", "gfx1201", "gfx12500"):
            for target in (arch, f"{arch}:xnack-"):
                with self.subTest(arch=target):
                    self.assertEqual(unified_framework_flags(target), [])

    def test_unspecified_architecture_keeps_framework_default(self):
        for arch in (None, ""):
            with self.subTest(arch=arch):
                self.assertEqual(unified_framework_flags(arch), [])


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


# ============================================================================
# HIP device / error helpers
#
# These drive hip_device_problem() and hip_last_error() against a fake HIP
# handle rather than mocking the two helpers themselves. The sweep-level test
# (test_gemm_result_status.TestNoGpu) does mock them, which is right for what it
# asserts, but it leaves the ctypes glue -- the byref write, the two restypes,
# the pop-on-read contract -- with no coverage at all. That glue is exactly
# where a silent regression hurts: a dropped restype makes hipGetErrorName
# return an int and every reason string raise, and a byref that is never read
# makes a healthy node look device-less.
#
# load_hip_runtime is the seam, not _hip_runtime_once: patching the latter would
# skip the restype assignment that is itself under test.
# ============================================================================


class _FakeHip:
    """A libamdhip64 stand-in with the four entry points the helpers call.

    Only hipGetDeviceCount and hipGetLastError need real behaviour; the two
    error-string lookups are Mocks so tests can assert how they were called.
    """

    _MESSAGES = {
        100: (b"hipErrorNoDevice", b"no ROCm-capable device is detected"),
        700: (b"hipErrorIllegalAddress", b"an illegal memory access was encountered"),
    }

    def __init__(self, *, device_count=1, count_err=0, last_errs=()):
        self.device_count = device_count
        self.count_err = count_err
        # A queue, not a scalar: hipGetLastError POPS the pending error, and the
        # sweep relies on that to stop one kernel's failure leaking into the
        # next one's report.
        self._pending = list(last_errs)
        self.last_error_calls = 0
        self.hipGetErrorName = Mock(side_effect=self._name)
        self.hipGetErrorString = Mock(side_effect=self._string)

    def _name(self, err):
        return self._MESSAGES.get(err, (None, None))[0]

    def _string(self, err):
        return self._MESSAGES.get(err, (None, None))[1]

    def hipGetDeviceCount(self, ptr):
        # The real symbol writes through the int* its caller passed. ctypes.cast
        # rather than ptr._obj: _obj is a CPython implementation detail of
        # CArgObject, while cast is public API and raises if byref() is ever
        # dropped from the call site -- which is the regression worth catching.
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int)).contents.value = self.device_count
        return self.count_err

    def hipGetLastError(self):
        self.last_error_calls += 1
        return self._pending.pop(0) if self._pending else 0


class _HipHelperTestCase(unittest.TestCase):
    """Clears the _hip_runtime_once cache around every test.

    _hip_runtime_once is lru_cache'd because each load re-runs find_library.
    Without clearing it on BOTH sides, the first test to run fixes the handle
    for the whole process: a fake leaks into later tests, or worse, a real
    handle loaded by an earlier test file makes these assertions depend on
    whether the box has ROCm installed.
    """

    def setUp(self):
        _hip_runtime_once.cache_clear()

    def tearDown(self):
        _hip_runtime_once.cache_clear()

    def _patch_hip(self, fake):
        return patch.object(dispatcher_common, "load_hip_runtime", return_value=fake)


class TestHipRuntimeHandle(_HipHelperTestCase):
    def test_load_failure_is_none_not_an_exception(self):
        # load_hip_runtime raises OSError naming every candidate it tried. The
        # helpers run on nodes that legitimately have no ROCm, so that must
        # degrade to None rather than take the sweep down.
        with patch.object(
            dispatcher_common, "load_hip_runtime", side_effect=OSError("no libamdhip64")
        ):
            self.assertIsNone(_hip_runtime_once())

    def test_restypes_are_set_so_error_names_are_bytes(self):
        # Without these two assignments ctypes reports a c_char_p return as a
        # plain int, and both .decode() calls in _hip_error_text raise
        # AttributeError on a real handle. A fake that happens to return bytes
        # would hide that, so assert the restypes directly.
        fake = _FakeHip()
        with self._patch_hip(fake):
            hip = _hip_runtime_once()
        self.assertIs(hip.hipGetErrorName.restype, ctypes.c_char_p)
        self.assertIs(hip.hipGetErrorString.restype, ctypes.c_char_p)

    def test_handle_is_loaded_once_per_process(self):
        fake = _FakeHip()
        with patch.object(
            dispatcher_common, "load_hip_runtime", return_value=fake
        ) as load:
            _hip_runtime_once()
            _hip_runtime_once()
        self.assertEqual(load.call_count, 1)


class TestHipErrorText(_HipHelperTestCase):
    def test_formats_name_and_description(self):
        fake = _FakeHip()
        self.assertEqual(
            _hip_error_text(fake, 100),
            "hipErrorNoDevice: no ROCm-capable device is detected",
        )

    def test_unknown_code_degrades_to_question_marks(self):
        # A real handle returns NULL for a code it does not know, which ctypes
        # surfaces as None. The `or b"?"` fallbacks keep that from raising.
        fake = _FakeHip()
        self.assertEqual(_hip_error_text(fake, 9999), "?: ?")


class TestHipDeviceProblem(_HipHelperTestCase):
    def test_healthy_device_reports_no_problem(self):
        with self._patch_hip(_FakeHip(device_count=1, count_err=0)):
            self.assertIsNone(hip_device_problem())

    def test_zero_devices_without_an_error_is_still_a_problem(self):
        # hipGetDeviceCount can succeed and report zero. Returning None here
        # would put the sweep straight back to 64 opaque status=-1 lines, so
        # this is the single likeliest regression in the helper.
        with self._patch_hip(_FakeHip(device_count=0, count_err=0)):
            problem = hip_device_problem()
        self.assertIsNotNone(problem)
        self.assertIn("0 device(s)", problem)

    def test_error_code_is_named_in_the_reason(self):
        with self._patch_hip(_FakeHip(device_count=0, count_err=100)):
            problem = hip_device_problem()
        self.assertIn("hipErrorNoDevice", problem)
        self.assertIn("no ROCm-capable device is detected", problem)

    def test_missing_kfd_is_called_out(self):
        # This is develop #2687/#2698 verbatim: BUILD_INSTANCES_ONLY starts the
        # container without --device=/dev/kfd, so the fix an operator needs is
        # the docker flag, not the device count.
        with self._patch_hip(_FakeHip(device_count=0, count_err=100)), patch(
            "os.path.exists", return_value=False
        ), patch.dict(os.environ, {}, clear=True):
            problem = hip_device_problem()
        self.assertIn("/dev/kfd does not exist", problem)
        self.assertIn("--device=/dev/kfd", problem)

    def test_unreadable_kfd_is_called_out(self):
        # /dev/kfd present but not group-accessible is a different fix
        # (--group-add), so it must not be reported as a missing device node.
        with self._patch_hip(_FakeHip(device_count=0, count_err=100)), patch(
            "os.path.exists", return_value=True
        ), patch("os.access", return_value=False), patch.dict(
            os.environ, {}, clear=True
        ):
            problem = hip_device_problem()
        self.assertNotIn("does not exist", problem)
        self.assertIn("render/video group", problem)

    def test_visible_device_filters_are_appended(self):
        # The commonest "the node has GPUs but the container does not" cause. A
        # typo in one of these names would silently drop it from every report.
        for var in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            with self.subTest(var=var):
                _hip_runtime_once.cache_clear()
                with self._patch_hip(
                    _FakeHip(device_count=0, count_err=100)
                ), patch.dict(os.environ, {var: ""}, clear=True):
                    problem = hip_device_problem()
                self.assertIn(f"{var}=''", problem)

    def test_runtime_that_cannot_load_is_not_reported_as_a_problem(self):
        # Documented but otherwise untested: the kernel .so files carry their own
        # RUNPATH and may well find HIP even when this process cannot, so an
        # unloadable runtime here is not evidence of a missing GPU.
        with patch.object(
            dispatcher_common, "load_hip_runtime", side_effect=OSError("no libamdhip64")
        ):
            self.assertIsNone(hip_device_problem())


class TestHipLastError(_HipHelperTestCase):
    def test_no_pending_error_is_empty_string(self):
        # Must be "" and not "hipSuccess: no error": the sweep appends this to a
        # run_fail reason only when it is truthy, and a success string there
        # would read as the cause of the failure.
        with self._patch_hip(_FakeHip()):
            self.assertEqual(hip_last_error(), "")

    def test_pending_error_is_named(self):
        with self._patch_hip(_FakeHip(last_errs=(700,))):
            self.assertEqual(
                hip_last_error(),
                "hipErrorIllegalAddress: an illegal memory access was encountered",
            )

    def test_no_runtime_is_empty_string_not_a_crash(self):
        with patch.object(
            dispatcher_common, "load_hip_runtime", side_effect=OSError("no libamdhip64")
        ):
            self.assertEqual(hip_last_error(), "")

    def test_each_call_drains_the_error(self):
        # The sweep calls this once before every kernel precisely to drop what
        # the previous one left. Memoize the result and kernel N+1 inherits
        # kernel N's error, which is worse than no error at all.
        fake = _FakeHip(last_errs=(700,))
        with self._patch_hip(fake):
            first = hip_last_error()
            second = hip_last_error()
        self.assertIn("hipErrorIllegalAddress", first)
        self.assertEqual(second, "")
        self.assertEqual(fake.last_error_calls, 2)


if __name__ == "__main__":
    unittest.main()
