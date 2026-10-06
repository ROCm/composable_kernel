#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Regression tests for the architecture-aware LDS staging budget.

The budget for the A+B staging tiles used to be a single table keyed on the
pipeline alone. That held every architecture to gfx942's 64 KB, so gfx950
(160 KB of LDS) and gfx1250 (320 KB) silently lost the largest and deepest
tiles at codegen time -- they were never generated, never benchmarked and
never selectable.

These tests pin the property that makes the defect impossible to reintroduce:
the effective budget must differ between architectures with different LDS
capacities, and must never exceed what the silicon actually has.

Can be run as:
    python3 tests/test_lds_capacity_arch_aware.py
    ctest -R test_lds_capacity_arch_aware
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from arch_filter import ArchFilter, KernelConfig  # noqa: E402
from arch_specs_generated import (  # noqa: E402
    LDS_CAPACITY_LIMITS_BY_ARCH,
    LDS_TOTAL_CAPACITY_BY_ARCH,
    get_lds_limit,
)

# Source of truth: get_lds_size() in include/ck_tile/core/arch/arch.hpp
HARDWARE_LDS_KB = {
    "gfx908": 64,
    "gfx90a": 64,
    "gfx942": 64,
    "gfx950": 160,
    "gfx1100": 64,
    "gfx1200": 64,
    "gfx1201": 64,
    "gfx1250": 320,
}

# gfx942's budget is already correct, so this change must leave it untouched.
# Any diff here means the change stopped being a pure widening.
GFX942_FROZEN = {
    "mem": 65536,
    "compv1": 65536,
    "compv2": 65536,
    "compv3": 65536,
    "compv4": 32768,
    "compv5": 65536,
    "compv6": 32768,
    "preshufflev1": 32768,
    "preshufflev2": 32768,
    # comp_async allocates two LDS buffers unconditionally. It previously had no
    # entry and inherited the 64 KB default, which is twice what it can use.
    "comp_async": 32768,
    "wavelet": 65536,
    "default": 65536,
}

# Pipelines that get half the capacity. compv4, preshufflev2 and comp_async are
# genuinely double-buffered, so half is the exact model rather than a margin.
# comp_async doubles unconditionally (num_lds_buffers = 2), which makes a
# missing entry silently grant it twice the LDS it allocates.
HALF_CAPACITY_PIPELINES = (
    "compv4",
    "compv6",
    "preshufflev1",
    "preshufflev2",
    "comp_async",
)

# Every pipeline the validators can encounter must have a deliberate entry.
# Falling through to "default" is how comp_async got a full-capacity budget.
EXPECTED_PIPELINES = set(HALF_CAPACITY_PIPELINES) | {
    "mem",
    "compv1",
    "compv2",
    "compv3",
    "compv5",
    "wavelet",
    "default",
}


class TestLdsBudgetIsArchAware(unittest.TestCase):
    """The core guard: the budget must depend on the architecture."""

    def test_budget_differs_across_architectures(self):
        for pipeline in ("mem", "compv3", "compv4", "default"):
            with self.subTest(pipeline=pipeline):
                gfx942 = get_lds_limit("gfx942", pipeline)
                gfx950 = get_lds_limit("gfx950", pipeline)
                gfx1250 = get_lds_limit("gfx1250", pipeline)

                self.assertLess(
                    gfx942,
                    gfx950,
                    f"{pipeline}: gfx950 has 160 KB of LDS but is budgeted like "
                    f"gfx942 ({gfx942} B). The budget is architecture-blind again.",
                )
                self.assertLess(
                    gfx950,
                    gfx1250,
                    f"{pipeline}: gfx1250 has 320 KB of LDS but is budgeted at or "
                    f"below gfx950 ({gfx950} B).",
                )

    def test_budget_tracks_hardware_capacity(self):
        """Architectures with equal LDS get equal budgets, and vice versa."""
        for arch, capacity_kb in HARDWARE_LDS_KB.items():
            with self.subTest(arch=arch):
                expected = HARDWARE_LDS_KB["gfx942"] == capacity_kb
                actual = get_lds_limit(arch, "default") == get_lds_limit(
                    "gfx942", "default"
                )
                self.assertEqual(
                    expected,
                    actual,
                    f"{arch} has {capacity_kb} KB of LDS; its default budget "
                    f"should match gfx942's only when its capacity does.",
                )


class TestLdsBudgetIsSafe(unittest.TestCase):
    """Guards that keep a mis-edited budget from reaching a GPU."""

    def test_budget_never_exceeds_hardware(self):
        for arch, per_pipeline in LDS_CAPACITY_LIMITS_BY_ARCH.items():
            capacity = HARDWARE_LDS_KB[arch] * 1024
            for pipeline, budget in per_pipeline.items():
                with self.subTest(arch=arch, pipeline=pipeline):
                    self.assertLessEqual(
                        budget,
                        capacity,
                        f"{arch}/{pipeline}: budget {budget} B exceeds the "
                        f"{capacity} B the hardware has.",
                    )

    def test_gfx942_budget_is_unchanged(self):
        self.assertEqual(LDS_CAPACITY_LIMITS_BY_ARCH["gfx942"], GFX942_FROZEN)

    def test_every_architecture_is_present(self):
        """A new GPU must not silently inherit another architecture's budget."""
        self.assertEqual(set(LDS_CAPACITY_LIMITS_BY_ARCH), set(HARDWARE_LDS_KB))

    def test_every_pipeline_has_a_deliberate_entry(self):
        """Falling through to 'default' is how a double-buffered pipeline
        silently gets twice the LDS it allocates."""
        for arch, per_pipeline in LDS_CAPACITY_LIMITS_BY_ARCH.items():
            with self.subTest(arch=arch):
                self.assertEqual(set(per_pipeline), EXPECTED_PIPELINES)

    def test_double_buffered_pipelines_get_half(self):
        """Pipelines that stage two LDS buffers get half the capacity.

        Only the ordering and the ratio are pinned, not a byte count, so the
        test keeps holding when an architecture's capacity changes.
        """
        for arch, per_pipeline in LDS_CAPACITY_LIMITS_BY_ARCH.items():
            capacity = HARDWARE_LDS_KB[arch] * 1024
            for pipeline in HALF_CAPACITY_PIPELINES:
                with self.subTest(arch=arch, pipeline=pipeline):
                    self.assertEqual(per_pipeline[pipeline], capacity // 2)

    def test_total_capacity_matches_hardware(self):
        for arch, capacity_kb in HARDWARE_LDS_KB.items():
            with self.subTest(arch=arch):
                self.assertEqual(
                    LDS_TOTAL_CAPACITY_BY_ARCH[arch], capacity_kb * 1024
                )

    def test_unknown_arch_gets_the_smallest_budget(self):
        """An unrecognised target must not be handed more LDS than it may have."""
        smallest = min(p["default"] for p in LDS_CAPACITY_LIMITS_BY_ARCH.values())
        self.assertEqual(get_lds_limit("gfx9999", "default"), smallest)


class TestDoubleBufferedStaging(unittest.TestCase):
    """Ping-pong staging allocates 2 * (A + B), so the budget must halve.

    The pipeline name alone does not imply it: mem, compv3, compv5 and compv6
    make double buffering a configuration choice, so the flag has to reach the
    capacity check or those kernels get twice the LDS they are budgeted for.
    """

    def test_configurable_pipelines_halve_when_double_buffered(self):
        for arch in LDS_CAPACITY_LIMITS_BY_ARCH:
            capacity = LDS_TOTAL_CAPACITY_BY_ARCH[arch]
            for pipeline in ("mem", "compv3", "compv5"):
                with self.subTest(arch=arch, pipeline=pipeline):
                    self.assertEqual(
                        get_lds_limit(arch, pipeline, double_smem_buffer=True),
                        capacity // 2,
                    )

    def test_double_buffered_budget_exactly_fits_capacity(self):
        """Two buffers of the budgeted size must not exceed the silicon."""
        for arch in LDS_CAPACITY_LIMITS_BY_ARCH:
            capacity = LDS_TOTAL_CAPACITY_BY_ARCH[arch]
            for pipeline in LDS_CAPACITY_LIMITS_BY_ARCH[arch]:
                with self.subTest(arch=arch, pipeline=pipeline):
                    budget = get_lds_limit(arch, pipeline, double_smem_buffer=True)
                    self.assertLessEqual(2 * budget, capacity)

    def test_always_double_pipelines_are_not_halved_twice(self):
        """compv4/preshufflev2 already carry the halving in their budget."""
        for arch in LDS_CAPACITY_LIMITS_BY_ARCH:
            for pipeline in ("compv4", "preshufflev2"):
                with self.subTest(arch=arch, pipeline=pipeline):
                    self.assertEqual(
                        get_lds_limit(arch, pipeline, double_smem_buffer=True),
                        get_lds_limit(arch, pipeline),
                    )

    def test_single_buffered_is_the_default(self):
        """Existing callers pass no flag and must see no change."""
        for arch in LDS_CAPACITY_LIMITS_BY_ARCH:
            for pipeline in LDS_CAPACITY_LIMITS_BY_ARCH[arch]:
                with self.subTest(arch=arch, pipeline=pipeline):
                    self.assertEqual(
                        get_lds_limit(arch, pipeline),
                        LDS_CAPACITY_LIMITS_BY_ARCH[arch][pipeline],
                    )


class TestLdsValidationEndToEnd(unittest.TestCase):
    """The budget has to actually reach the validator, not just the table."""

    @staticmethod
    def _config(pipeline, tile_m=128, tile_n=256, tile_k=128, double_smem_buffer=False):
        # fp16 A and B: 128x128x2 + 256x128x2 = 96 KB of staging.
        return KernelConfig(
            double_smem_buffer=double_smem_buffer,
            datatype_a="fp16",
            datatype_b="fp16",
            datatype_c="fp16",
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=8,
            pipeline=pipeline,
        )

    def _lds_errors(self, arch, config):
        result = ArchFilter(arch, strict_mode=False).validate_kernel(config)
        return [e for e in result.errors if "LDS capacity exceeded" in e]

    def test_96kb_tile_rejected_on_gfx942_accepted_on_gfx950(self):
        """The tile at the heart of the defect: fits gfx950, not gfx942."""
        config = self._config("compv3")
        self.assertTrue(
            self._lds_errors("gfx942", config),
            "96 KB of staging must not fit gfx942's 64 KB budget.",
        )
        self.assertFalse(
            self._lds_errors("gfx950", config),
            "96 KB of staging fits comfortably in gfx950's 160 KB.",
        )

    def test_double_buffered_96kb_tile_rejected_on_gfx950(self):
        """96 KB fits gfx950 once, but not twice: 2 x 96 KB > 160 KB."""
        self.assertFalse(
            self._lds_errors("gfx950", self._config("compv3")),
            "single-buffered 96 KB fits gfx950",
        )
        self.assertTrue(
            self._lds_errors(
                "gfx950", self._config("compv3", double_smem_buffer=True)
            ),
            "double-buffered 96 KB needs 192 KB and must be rejected on gfx950.",
        )

    def test_error_message_names_the_architecture(self):
        """The old message was architecture-neutral, which hid the defect."""
        errors = self._lds_errors("gfx942", self._config("compv3"))
        self.assertTrue(errors)
        self.assertIn("gfx942", errors[0])


class TestCppPythonParity(unittest.TestCase):
    """The two validators must agree; a Python-only fix emits kernels that the
    C++ filter then rejects at dispatch.

    Compiles the generated header and diffs every value against the Python
    table. Needs only a host C++ compiler -- no GPU, no hipcc -- and skips
    cleanly where none is available.
    """

    # comp_async is deliberately absent: it has no Pipeline enumerator on the
    # C++ side, so it exists only in the Python table.
    CPP_PIPELINE_ENUM = {
        "mem": "Mem",
        "compv1": "CompV1",
        "compv2": "CompV2",
        "compv3": "CompV3",
        "compv4": "CompV4",
        "compv5": "CompV5",
        "compv6": "CompV6",
        "preshufflev1": "PreShuffleV1",
        "preshufflev2": "PreShuffleV2",
        "wavelet": "Wavelet",
    }

    @staticmethod
    def _arch_enum(arch):
        return arch.upper().replace("GFX", "GFX_")

    def _run_cpp_probe(self):
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("no host C++ compiler available")

        pairs = [
            (arch, pipeline)
            for arch in sorted(LDS_CAPACITY_LIMITS_BY_ARCH)
            for pipeline in sorted(self.CPP_PIPELINE_ENUM)
        ]
        lines = "\n".join(
            f'    std::cout << "{a} {p} "'
            f" << get_lds_capacity(GpuArch::{self._arch_enum(a)},"
            f" Pipeline::{self.CPP_PIPELINE_ENUM[p]})"
            f' << " " << get_lds_total_capacity(GpuArch::{self._arch_enum(a)})'
            f' << "\\n";'
            for a, p in pairs
        )
        source = (
            '#include "ck_tile/dispatcher/arch_specs_generated.hpp"\n'
            "#include <iostream>\n"
            "using namespace ck_tile::dispatcher;\n"
            "using namespace ck_tile::dispatcher::arch_specs;\n"
            "int main() {\n" + lines + "\n    return 0;\n}\n"
        )

        includes = [
            DISPATCHER_DIR / "include",
            DISPATCHER_DIR.parent / "include",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "probe.cpp"
            exe = Path(tmp) / "probe"
            src.write_text(source)
            cmd = [compiler, "-std=c++17"]
            for inc in includes:
                cmd += ["-I", str(inc)]
            cmd += [str(src), "-o", str(exe)]

            build = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                build.returncode,
                0,
                f"generated header failed to compile:\n{build.stderr}",
            )
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)

        parsed = {}
        for line in run.stdout.strip().splitlines():
            arch, pipeline, budget, total = line.split()
            parsed[(arch, pipeline)] = (int(budget), int(total))
        return parsed

    def test_cpp_budgets_match_python(self):
        for (arch, pipeline), (budget, _) in self._run_cpp_probe().items():
            with self.subTest(arch=arch, pipeline=pipeline):
                self.assertEqual(budget, get_lds_limit(arch, pipeline))

    def test_cpp_total_capacity_matches_python(self):
        for (arch, _), (_, total) in self._run_cpp_probe().items():
            with self.subTest(arch=arch):
                self.assertEqual(total, LDS_TOTAL_CAPACITY_BY_ARCH[arch])

    def test_cpp_budget_brackets_the_96kb_tile(self):
        """The budget accessor puts 96 KB of staging on opposite sides of the
        line for gfx942 and gfx950.

        This compares accessor values only. TestCppValidateEndToEnd exercises
        the validator itself.
        """
        staging = 128 * 128 * 2 + 256 * 128 * 2  # 98304 bytes
        cpp = self._run_cpp_probe()
        self.assertGreater(staging, cpp[("gfx942", "compv3")][0])
        self.assertLessEqual(staging, cpp[("gfx950", "compv3")][0])


class TestCppValidateEndToEnd(unittest.TestCase):
    """Drive the real C++ entry point, not just the budget accessor.

    Builds a KernelKey and calls ArchFilter::validate(), so the LDS check is
    reached the same way a caller reaches it. Scoped to gfx942 and gfx950:
    they agree on warp tiles, so a rejection here can only come from the LDS
    budget and not from an unrelated validator.
    """

    SOURCE = r"""
#include "ck_tile/dispatcher/arch_filter.hpp"
#include <iostream>
using namespace ck_tile::dispatcher;

static KernelKey make_key(int m, int n, int k)
{
    KernelKey key{};
    key.signature.dtype_a        = DataType::FP16;
    key.signature.dtype_b        = DataType::FP16;
    key.signature.dtype_c        = DataType::FP16;
    key.signature.dtype_acc      = DataType::FP32;
    key.signature.layout_a       = LayoutTag::RowMajor;
    key.signature.layout_b       = LayoutTag::ColMajor;
    key.signature.layout_c       = LayoutTag::RowMajor;
    key.signature.split_k        = 1;
    key.signature.elementwise_op = "PassThrough";
    key.signature.num_d_tensors  = 0;

    key.algorithm.tile_shape      = {(std::uint16_t)m, (std::uint16_t)n, (std::uint16_t)k};
    key.algorithm.wave_shape      = {2, 2, 1};
    key.algorithm.warp_tile_shape = {32, 32, 16};
    key.algorithm.pipeline        = Pipeline::CompV3;
    key.algorithm.scheduler       = Scheduler::Intrawave;
    key.algorithm.epilogue        = Epilogue::CShuffle;
    key.algorithm.block_size      = 256;
    key.algorithm.double_buffer   = false;
    key.algorithm.persistent      = false;
    key.algorithm.preshuffle      = false;
    key.algorithm.transpose_c     = false;
    key.algorithm.num_wave_groups = 1;
    return key;
}

int main()
{
    // 128x256x128 fp16 -> 128*128*2 + 256*128*2 = 98304 bytes of staging.
    for(const char* arch : {"gfx942", "gfx950"})
    {
        auto result = ArchFilter(arch, false).validate(make_key(128, 256, 128));
        bool lds    = false;
        for(const auto& e : result.errors)
            if(e.find("LDS capacity exceeded") != std::string::npos)
                lds = true;
        std::cout << arch << " " << (result.valid ? 1 : 0) << " " << (lds ? 1 : 0) << "\n";
    }
    return 0;
}
"""

    def _run(self):
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("no host C++ compiler available")

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "validate_probe.cpp"
            exe = Path(tmp) / "validate_probe"
            src.write_text(self.SOURCE)
            cmd = [compiler, "-std=c++17"]
            for inc in (DISPATCHER_DIR / "include", DISPATCHER_DIR.parent / "include"):
                cmd += ["-I", str(inc)]
            cmd += [str(src), "-o", str(exe)]

            build = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                build.returncode, 0, f"probe failed to compile:\n{build.stderr}"
            )
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)

        out = {}
        for line in run.stdout.strip().splitlines():
            arch, valid, lds = line.split()
            out[arch] = (valid == "1", lds == "1")
        return out

    def test_96kb_tile_rejected_on_gfx942_accepted_on_gfx950(self):
        out = self._run()

        valid_942, lds_942 = out["gfx942"]
        self.assertFalse(valid_942, "98304 B of staging must not fit gfx942's 64 KB")
        self.assertTrue(
            lds_942, "rejection must come from the LDS check, not another validator"
        )

        valid_950, lds_950 = out["gfx950"]
        self.assertTrue(
            valid_950, "98304 B fits gfx950's 160 KB and must validate cleanly"
        )
        self.assertFalse(lds_950, "gfx950 must not raise an LDS error for this tile")

    def test_agrees_with_python_validator(self):
        """The same tile, through both validators, must reach the same verdict."""
        config = TestLdsValidationEndToEnd._config("compv3")
        out = self._run()
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                py_lds = bool(
                    [
                        e
                        for e in ArchFilter(arch, strict_mode=False)
                        .validate_kernel(config)
                        .errors
                        if "LDS capacity exceeded" in e
                    ]
                )
                self.assertEqual(py_lds, out[arch][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
