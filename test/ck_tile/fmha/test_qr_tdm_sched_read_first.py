# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU compilation checks for the ReadFirst placement's address dependency guard.

Run with the configured CK HIP compiler and build directory. The positive case
must compile before either expected failure is checked. No device code or GPU
work is launched, and no pipeline tuning definitions are required.
"""

import argparse
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_GUARD_DIAGNOSTIC = "read-first placement requires the D64/V64 N64 default read mapping"
_ERROR_LINE = re.compile(r"^.*: (?:fatal )?error: (.*)$", re.MULTILINE)
_GUARD_ERROR = re.compile(
    r"static assertion failed[^\n]*: " + re.escape(_GUARD_DIAGNOSTIC) + r"$"
)

_COMMON_SOURCE = """\
#include "ck/config.h"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"

// The placement guard only needs the ordinary-read/rescale mode. Use the real
// geometry, schedule, and executor without instantiating an unrelated pipeline.
template <typename Geometry_>
struct OrdinaryMode
{
    using Geometry = Geometry_;
    struct SelectedPolicy
    {
        static constexpr bool kLeadFirstLoad = false;
    };
    static constexpr bool kStreamOutputRescale = false;
};

struct PvTrace
{
    int wmmas                 = 0;
    int reads                 = 0;
    int wmmas_before_observed = -1;
    bool valid_reads          = true;
};

template <typename Geometry,
          typename ReadMapping,
          ck_tile::index_t Stage,
          ck_tile::index_t ObservedAccess>
constexpr PvTrace ObservePvStage()
{
    using Placement            = ck_tile::FmhaTdmSchedReadFirstPlacement<ReadMapping>;
    using Schedule             = ck_tile::FmhaTdmSchedSequentialTileSchedule<Placement>;
    using Mode                 = OrdinaryMode<Geometry>;
    constexpr bool final_stage = Stage == Geometry::kPvStages - 1;
    constexpr int num_reads    = final_stage ? Geometry::kKSuLoadCount : Geometry::kVStageLoadCount;
    constexpr auto expected_kind =
        final_stage ? ck_tile::FmhaTdmSchedLoadKind::KRead : ck_tile::FmhaTdmSchedLoadKind::VRead;
    int accesses[num_reads]{};
    PvTrace trace;
    auto wmma = [&](auto, auto) { ++trace.wmmas; };
    auto read = [&](auto kind, auto access) {
        constexpr int index = decltype(access)::value;
        ++trace.reads;
        if constexpr(index < 0 || index >= num_reads || decltype(kind)::value != expected_kind)
            trace.valid_reads = false;
        else
            ++accesses[index];
        if constexpr(index == ObservedAccess)
            trace.wmmas_before_observed = trace.wmmas;
    };
    auto fence   = [](auto, auto, auto) {};
    auto rescale = [](auto) {};
    ck_tile::FmhaTdmSchedPlacementExecutor<Schedule, Mode>::template ExecutePvStage<Stage>(
        wmma, read, fence, rescale);
    for(int count : accesses)
        trace.valid_reads = trace.valid_reads && count == 1;
    trace.valid_reads =
        trace.valid_reads && trace.reads == num_reads && trace.wmmas == Geometry::kPvWmmasPerStage;
    return trace;
}
"""

_POSITIVE_SOURCE = _COMMON_SOURCE + """\
using Geometry        = ck_tile::FmhaTdmSchedNativeGeometry<ck_tile::bf16_t, 64, 64, 64>;
using Mapping         = ck_tile::FmhaTdmSchedScheduleFor<Geometry>;
constexpr auto first  = ObservePvStage<Geometry, Mapping, 0, 0>();
constexpr auto second = ObservePvStage<Geometry, Mapping, 0, 1>();
static_assert(first.valid_reads && second.valid_reads);
static_assert(first.wmmas_before_observed == 0);
// WMMA 0 prepares PV access 1's address before the next row issues the read.
static_assert(second.wmmas_before_observed == 1);
"""

_D192_SOURCE = _COMMON_SOURCE + """\
using Geometry       = ck_tile::FmhaTdmSchedNativeGeometry<ck_tile::half_t, 192, 128, 128>;
using Mapping        = ck_tile::FmhaTdmSchedScheduleFor<Geometry>;
constexpr auto trace = ObservePvStage<Geometry, Mapping, 3, 0>();
static_assert(trace.valid_reads);
// Without the guard, the final PV stage can issue the next K read before WMMA 0.
static_assert(trace.wmmas_before_observed == 0);
"""

_CUSTOM_MAPPING_SOURCE = _COMMON_SOURCE + """\
using Geometry = ck_tile::FmhaTdmSchedNativeGeometry<ck_tile::bf16_t, 64, 64, 64>;
struct PackedPvMapping : ck_tile::FmhaTdmSchedScheduleFor<Geometry>
{
    template <ck_tile::index_t Stage, ck_tile::index_t Wmma, bool SecondHalf, typename Visitor>
    static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        if constexpr(SecondHalf)
        {
            constexpr auto kind = Stage == Geometry::kPvStages - 1
                                      ? ck_tile::FmhaTdmSchedLoadKind::KRead
                                      : ck_tile::FmhaTdmSchedLoadKind::VRead;
            auto access         = [&](auto index) {
                visitor(std::integral_constant<ck_tile::FmhaTdmSchedLoadKind, kind>{}, index);
            };
            // Move row 1's read into row 0 while keeping every access exactly
            // once and preserving the geometry's stage/read/WMMA totals.
            if constexpr(Wmma == 0)
            {
                access(ck_tile::number<0>{});
                access(ck_tile::number<1>{});
            }
            else if constexpr(Wmma > 1)
                access(ck_tile::number<Wmma>{});
        }
    }
};
constexpr auto trace = ObservePvStage<Geometry, PackedPvMapping, 0, 1>();
static_assert(trace.valid_reads);
// Counts alone cannot detect this read-before-address dependency.
static_assert(trace.wmmas_before_observed == 0);
"""


class TestReadFirstAddressGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (cls.ck_root / "include" / "ck_tile").is_dir():
            raise RuntimeError(f"CK headers not found under {cls.ck_root}")
        if not (cls.ck_build / "include" / "ck" / "config.h").is_file():
            raise RuntimeError(
                f"Configured ck/config.h not found under {cls.ck_build / 'include'}"
            )
        cls.directory = tempfile.TemporaryDirectory(prefix="ck_read_first_")
        cls.addClassCleanup(cls.directory.cleanup)
        # A failed compiler/header/config setup must fail the whole suite, even
        # when its diagnostics happen to contain the expected guard text.
        cls.positive = cls.compile_source("d64_default", _POSITIVE_SOURCE)
        if cls.positive.returncode != 0:
            raise RuntimeError(
                "The legal D64 default mapping did not compile:\n" + cls.positive.stdout
            )

    @classmethod
    def compile_source(cls, name, source):
        source_path = Path(cls.directory.name) / f"{name}.cpp"
        source_path.write_text(source, encoding="utf-8")
        argv = [
            cls.compiler,
            "--offload-host-only",
            "-x",
            "hip",
            "-std=c++20",
            "-fsyntax-only",
            "-fno-color-diagnostics",
            "-I" + str(cls.ck_build / "include"),
            "-I" + str(cls.ck_root / "include"),
            "-I" + str(cls.ck_root / "library" / "include"),
            str(source_path),
        ]
        try:
            result = subprocess.run(
                argv,
                cwd=cls.ck_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Could not run {shlex.join(argv)}: {error}") from error
        result.stdout = f"Command: {shlex.join(argv)}\n" + result.stdout
        return result

    def expect_mapping_rejection(self, name, source):
        result = self.compile_source(name, source)
        self.assertGreater(result.returncode, 0, result.stdout)
        errors = _ERROR_LINE.findall(result.stdout)
        self.assertTrue(errors, result.stdout)
        # Match the exact assertion diagnostic on error lines, not echoed source
        # text or a generic compilation failure. Reject any unrelated error too.
        for error in errors:
            self.assertRegex(error, _GUARD_ERROR, result.stdout)

    def test_legal_d64_default_mapping_compiles(self):
        self.assertEqual(self.positive.returncode, 0, self.positive.stdout)

    def test_d192_n128_read_first_is_rejected(self):
        self.expect_mapping_rejection("d192_n128", _D192_SOURCE)

    def test_d64_packed_pv_read_first_is_rejected(self):
        self.expect_mapping_rejection("d64_packed_pv", _CUSTOM_MAPPING_SOURCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler", required=True, help="Configured CMAKE_HIP_COMPILER"
    )
    parser.add_argument(
        "--ck-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--ck-build", required=True, type=Path, help="Configured CK build directory"
    )
    arguments, unittest_arguments = parser.parse_known_args()
    TestReadFirstAddressGuard.compiler = arguments.compiler
    TestReadFirstAddressGuard.ck_root = arguments.ck_root.resolve()
    TestReadFirstAddressGuard.ck_build = arguments.ck_build.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_arguments])
