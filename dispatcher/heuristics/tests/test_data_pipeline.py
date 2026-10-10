#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for data_pipeline.py.

Covers: kernel name parsing, layout derivation, streaming log parsing,
schema validation, and corner cases (empty logs, malformed JSON, single-shape).
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import (
    parse_kernel_name,
    _layout_from_problem,
    parse_streaming_log,
    save_parquet,
    load_parquet,
    CANONICAL_COLUMNS,
)


# ---------------------------------------------------------------------------
# parse_kernel_name
# ---------------------------------------------------------------------------


class TestParseKernelName:
    def test_standard_name(self):
        name = "gemm_universal_fp8_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x128_1x4x1_16x16x128"
        result = parse_kernel_name(name)
        assert result["dtype"] == "fp8"
        assert result["layout"] == "rcr"
        assert result["pipeline"] == "compv3"
        assert result["epilogue"] == "cshuffle"
        assert result["scheduler"] == "intrawave"
        assert result["pad_m"] is False
        assert result["pad_n"] is False
        assert result["pad_k"] is False
        assert result["persistent"] is False
        assert result["tile_m"] == 128
        assert result["tile_n"] == 128
        assert result["tile_k"] == 128
        assert result["warp_m"] == 1
        assert result["warp_n"] == 4
        assert result["warp_k"] == 1
        assert result["warp_tile_m"] == 16
        assert result["warp_tile_n"] == 16
        assert result["warp_tile_k"] == 128

    def test_with_padding_and_persistent(self):
        name = "gemm_universal_fp16_rrr_compv4_default_interwave_True_True_True_True_256x256x64_2x2x1_32x32x16"
        result = parse_kernel_name(name)
        assert result["dtype"] == "fp16"
        assert result["layout"] == "rrr"
        assert result["pad_m"] is True
        assert result["pad_n"] is True
        assert result["pad_k"] is True
        assert result["persistent"] is True
        assert result["tile_m"] == 256

    def test_empty_name(self):
        assert parse_kernel_name("") == {}

    def test_malformed_name(self):
        assert parse_kernel_name("not_a_kernel_name") == {}

    def test_partial_name(self):
        result = parse_kernel_name("gemm_universal_fp8_rcr_compv3")
        assert result.get("dtype") == "fp8"
        assert result.get("layout") == "rcr"
        assert "tile_m" not in result  # not enough parts

    def test_all_layouts(self):
        for layout in ["rcr", "rrr", "crr", "ccr"]:
            name = f"gemm_universal_fp8_{layout}_compv3_cshuffle_intrawave_False_False_False_False_128x128x128_1x4x1_16x16x128"
            result = parse_kernel_name(name)
            assert result["layout"] == layout


# ---------------------------------------------------------------------------
# _layout_from_problem
# ---------------------------------------------------------------------------


class TestLayoutFromProblem:
    def test_rcr(self):
        assert (
            _layout_from_problem(
                {
                    "layout_a": "RowMajor",
                    "layout_b": "ColumnMajor",
                    "layout_c": "RowMajor",
                }
            )
            == "rcr"
        )

    def test_rrr(self):
        assert (
            _layout_from_problem(
                {"layout_a": "RowMajor", "layout_b": "RowMajor", "layout_c": "RowMajor"}
            )
            == "rrr"
        )

    def test_empty(self):
        assert _layout_from_problem({}) == "???"

    def test_case_insensitive(self):
        assert (
            _layout_from_problem(
                {
                    "layout_a": "rowmajor",
                    "layout_b": "COLUMNMAJOR",
                    "layout_c": "RowMajor",
                }
            )
            == "rcr"
        )


# ---------------------------------------------------------------------------
# parse_streaming_log
# ---------------------------------------------------------------------------

SAMPLE_LOG = """\
================================================================================
LOG FILE: test.log
================================================================================
CK Tile Profiling Run
GPU ID: 0

--- Running CK Tile benchmarks on GPU 0 ---

========================================
Shape 1: M=16 N=1536 K=7168 dtype=fp8 layout=rcr
========================================
Found 2 kernels
{
 "name": "gemm_universal_fp8_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x128_1x4x1_16x16x128",
 "problem": {
   "split_k":1,
   "m":16,
   "n":1536,
   "k":7168,
   "stride_a":7168,
   "stride_b":7168,
   "stride_c":1536,
   "dtype_a":"fp8",
   "dtype_b":"fp8",
   "dtype_acc":"fp32",
   "dtype_c":"fp16",
   "layout_a":"RowMajor",
   "layout_b":"ColumnMajor",
   "layout_c":"RowMajor",
   "structured_sparsity":false
},
 "perf_result": {
   "latency(ms)": 0.04,
   "tflops(TFlops)": 8.81,
   "bandwidth(GB/s)": 279.51
}
}
{
 "name": "gemm_universal_fp8_rcr_compv4_default_intrawave_False_False_False_False_128x128x64_2x2x1_32x32x16",
 "problem": {
   "split_k":1,
   "m":16,
   "n":1536,
   "k":7168,
   "stride_a":7168,
   "stride_b":7168,
   "stride_c":1536,
   "dtype_a":"fp8",
   "dtype_b":"fp8",
   "dtype_acc":"fp32",
   "dtype_c":"fp16",
   "layout_a":"RowMajor",
   "layout_b":"ColumnMajor",
   "layout_c":"RowMajor",
   "structured_sparsity":false
},
 "perf_result": {
   "latency(ms)": 0.05,
   "tflops(TFlops)": 7.22,
   "bandwidth(GB/s)": 228.85
}
}

========================================
Shape 2: M=20480 N=7168 K=256 dtype=fp8 layout=rcr
========================================
Found 1 kernels
{
 "name": "gemm_universal_fp8_rcr_mem_cshuffle_intrawave_False_False_False_True_64x64x128_1x4x1_16x16x32",
 "problem": {
   "split_k":1,
   "m":20480,
   "n":7168,
   "k":256,
   "stride_a":256,
   "stride_b":256,
   "stride_c":7168,
   "dtype_a":"fp8",
   "dtype_b":"fp8",
   "dtype_acc":"fp32",
   "dtype_c":"fp16",
   "layout_a":"RowMajor",
   "layout_b":"ColumnMajor",
   "layout_c":"RowMajor",
   "structured_sparsity":false
},
 "perf_result": {
   "latency(ms)": 0.15,
   "tflops(TFlops)": 505.00,
   "bandwidth(GB/s)": 1200.50
}
}
"""


class TestParseStreamingLog:
    def _write_log(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        f.write(content)
        f.close()
        return Path(f.name)

    def test_basic_parse(self):
        path = self._write_log(SAMPLE_LOG)
        df = parse_streaming_log(path, arch="gfx950")
        assert len(df) == 3
        assert df["arch"].iloc[0] == "gfx950"
        assert df["m"].tolist() == [16, 16, 20480]
        assert df["n"].tolist() == [1536, 1536, 7168]
        assert df["k"].tolist() == [7168, 7168, 256]

    def test_tflops_values(self):
        path = self._write_log(SAMPLE_LOG)
        df = parse_streaming_log(path)
        assert df["measured_tflops"].tolist() == pytest.approx([8.81, 7.22, 505.0])

    def test_kernel_config_parsed(self):
        path = self._write_log(SAMPLE_LOG)
        df = parse_streaming_log(path)
        assert df["tile_m"].iloc[0] == 128
        assert df["pipeline"].iloc[0] == "compv3"
        assert df["pipeline"].iloc[1] == "compv4"

    def test_layout_derived_from_json(self):
        path = self._write_log(SAMPLE_LOG)
        df = parse_streaming_log(path)
        assert all(df["layout"] == "rcr")

    def test_empty_log(self):
        path = self._write_log("No shapes here\nJust noise\n")
        df = parse_streaming_log(path)
        assert len(df) == 0
        for col in CANONICAL_COLUMNS:
            assert col in df.columns

    def test_single_kernel(self):
        log = """\
Shape 1: M=1 N=1 K=1 dtype=fp8 layout=rcr
{
 "name": "gemm_universal_fp8_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x128_1x4x1_16x16x128",
 "problem": {"split_k":1, "m":1, "n":1, "k":1, "dtype_a":"fp8", "dtype_b":"fp8", "layout_a":"RowMajor", "layout_b":"ColumnMajor", "layout_c":"RowMajor"},
 "perf_result": {"latency(ms)": 0.001, "tflops(TFlops)": 0.002, "bandwidth(GB/s)": 0.01}
}
"""
        path = self._write_log(log)
        df = parse_streaming_log(path)
        assert len(df) == 1
        assert df["m"].iloc[0] == 1
        assert bool(df["is_valid"].iloc[0]) is True

    def test_zero_tflops_marked_invalid(self):
        log = """\
Shape 1: M=16 N=16 K=16 dtype=fp8 layout=rcr
{
 "name": "test_kernel",
 "problem": {"split_k":1, "m":16, "n":16, "k":16, "dtype_a":"fp8"},
 "perf_result": {"latency(ms)": 0.0, "tflops(TFlops)": 0.0, "bandwidth(GB/s)": 0.0}
}
"""
        path = self._write_log(log)
        df = parse_streaming_log(path)
        assert len(df) == 1
        assert bool(df["is_valid"].iloc[0]) is False

    def test_malformed_json_skipped(self):
        log = """\
Shape 1: M=16 N=16 K=16 dtype=fp8 layout=rcr
{
 "name": "good_kernel",
 "problem": {"split_k":1, "m":16, "n":16, "k":16, "dtype_a":"fp8"},
 "perf_result": {"latency(ms)": 0.01, "tflops(TFlops)": 1.0, "bandwidth(GB/s)": 10.0}
}
{ this is not valid json }
{
 "name": "another_good",
 "problem": {"split_k":1, "m":16, "n":16, "k":16, "dtype_a":"fp8"},
 "perf_result": {"latency(ms)": 0.02, "tflops(TFlops)": 2.0, "bandwidth(GB/s)": 20.0}
}
"""
        path = self._write_log(log)
        df = parse_streaming_log(path)
        assert len(df) == 2

    def test_extreme_shapes(self):
        """Tiny M=1 (single token) and very large M=20480."""
        path = self._write_log(SAMPLE_LOG)
        df = parse_streaming_log(path)
        assert 1 not in df["m"].values  # sample has M=16, M=20480
        assert 16 in df["m"].values
        assert 20480 in df["m"].values

    def test_run_id_assigned(self):
        path = self._write_log(SAMPLE_LOG)
        df = parse_streaming_log(path, run_id="test_run_123")
        assert all(df["run_id"] == "test_run_123")

    def test_op_type_assigned(self):
        path = self._write_log(SAMPLE_LOG)
        df = parse_streaming_log(path, op_type="gemm_streamk")
        assert all(df["op_type"] == "gemm_streamk")


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


class TestParquetIO:
    def test_round_trip(self, tmp_path):
        df = pd.DataFrame(
            {
                "m": [16, 32],
                "n": [1536, 1536],
                "k": [7168, 7168],
                "measured_tflops": [8.81, 15.0],
            }
        )
        path = tmp_path / "test.parquet"
        save_parquet(df, path)
        loaded = load_parquet(path)
        assert len(loaded) == 2
        assert loaded["m"].tolist() == [16, 32]

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.parquet"
        df = pd.DataFrame({"x": [1]})
        save_parquet(df, path)
        assert path.exists()


class TestVectorWidthSuffix:
    BASE = "gemm_universal_bf16_rcr_compv3_default_intrawave_False_False_False_True_128x128x128_2x2x1_32x32x16"

    def test_native_kernel_reports_zero(self):
        r = parse_kernel_name(self.BASE)
        assert (r["vec_a"], r["vec_b"], r["vec_c"]) == (0, 0, 0)

    def test_suffix_is_parsed(self):
        r = parse_kernel_name(self.BASE + "_vec4_4_8")
        assert (r["vec_a"], r["vec_b"], r["vec_c"]) == (4, 4, 8)

    def test_suffix_does_not_disturb_the_rest_of_the_name(self):
        plain = parse_kernel_name(self.BASE)
        vec = parse_kernel_name(self.BASE + "_vec8_1_1")
        for key in (
            "tile_m",
            "tile_n",
            "tile_k",
            "warp_tile_k",
            "pipeline",
            "persistent",
        ):
            assert plain[key] == vec[key], key

    def test_columns_are_canonical(self):
        for c in ("vec_a", "vec_b", "vec_c"):
            assert c in CANONICAL_COLUMNS

    def test_vec_is_parsed_when_further_suffixes_follow(self):
        r = parse_kernel_name(self.BASE + "_vec4_4_8_preshuffle")
        assert (r["vec_a"], r["vec_b"], r["vec_c"]) == (4, 4, 8)
        assert r["tile_m"] == 128, "stripping the suffix must not shift the split"

    def test_a_longer_width_is_not_truncated(self):
        r = parse_kernel_name(self.BASE + "_vec4_4_80")
        assert r["vec_c"] == 80

    def test_malformed_width_suffix_is_rejected_not_read_as_native(self):
        assert parse_kernel_name(self.BASE + "_vec4_4") == {}

    def test_a_non_width_vec_token_is_not_treated_as_a_suffix(self):
        r = parse_kernel_name(self.BASE + "_vecnative")
        assert (r["vec_a"], r["vec_b"], r["vec_c"]) == (0, 0, 0)
        assert r["tile_m"] == 128, "the rest of the name must still parse"

    def test_the_bridge_short_prefix_parses(self):
        short = self.BASE.replace("gemm_universal_", "gemm_", 1)
        r = parse_kernel_name(short + "_vec4_4_8")
        assert (r["vec_a"], r["vec_b"], r["vec_c"]) == (4, 4, 8)
        assert r["dtype"] == "bf16" and r["layout"] == "rcr"
        assert (r["tile_m"], r["tile_n"], r["tile_k"]) == (128, 128, 128)
        assert r["pipeline"] == "compv3" and r["persistent"] is True

    def test_both_prefixes_agree_field_for_field(self):
        long_r = parse_kernel_name(self.BASE)
        short_r = parse_kernel_name(self.BASE.replace("gemm_universal_", "gemm_", 1))
        assert long_r == short_r

    def test_a_trailing_non_separator_is_not_a_width_suffix(self):
        assert parse_kernel_name(self.BASE + "_vec4_4_8x") == {}

    def test_single_digit_and_multi_digit_widths_both_parse(self):
        assert parse_kernel_name(self.BASE + "_vec16_2_16")["vec_a"] == 16


class TestHardwareProfileLdsCapacity:
    @pytest.mark.parametrize(
        "reported,expected",
        [
            ("amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-", 163840),
            ("amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-", 65536),
            ("amdgcn-amd-amdhsa--gfx1250", 327680),
            ("gfx950", 163840),
            ("gfx950:xnack-", 163840),
        ],
    )
    def test_the_capacity_is_resolved_from_the_reported_name(
        self, reported, expected, monkeypatch
    ):
        import subprocess

        import data_pipeline as dp

        def fake_run(*a, **k):
            class R:
                stdout = (
                    "  Device Type:             GPU\n"
                    f"  Name:                    {reported}\n"
                    "  Compute Unit:            256\n"
                )

            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert dp.get_hardware_profile()["lds_capacity"] == expected

    def test_an_unrecognised_name_omits_the_key(self, monkeypatch):
        import subprocess

        import data_pipeline as dp

        def fake_run(*a, **k):
            class R:
                stdout = "  Device Type:   GPU\n  Name:   not-a-target\n"

            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert "lds_capacity" not in dp.get_hardware_profile()

    def test_every_profile_matches_the_arch_table(self):
        from convert_csv_to_parquet import HW_PROFILES

        from arch_specs_generated import LDS_TOTAL_CAPACITY_BY_ARCH

        assert HW_PROFILES, "no hardware profiles"
        for arch, profile in HW_PROFILES.items():
            assert arch in LDS_TOTAL_CAPACITY_BY_ARCH, arch
            assert profile["hw_lds_capacity"] == LDS_TOTAL_CAPACITY_BY_ARCH[arch], arch

    def test_the_profiles_are_sourced_not_restated(self, monkeypatch):
        import importlib

        import arch_specs_generated as specs
        import convert_csv_to_parquet as cc

        monkeypatch.setitem(specs.LDS_TOTAL_CAPACITY_BY_ARCH, "gfx950", 12345)
        try:
            reloaded = importlib.reload(cc)
            assert reloaded.HW_PROFILES["gfx950"]["hw_lds_capacity"] == 12345, (
                "the profile did not follow the arch table, so the value is "
                "restated rather than sourced"
            )
        finally:
            monkeypatch.undo()
            importlib.reload(cc)

    #: gfx1250 constants measured from rocminfo's GPU agent.
    MEASURED_GFX1250 = {
        "hw_num_cus": 256,
        "hw_simds_per_cu": 4,
        "hw_max_clock_mhz": 2400,
        "hw_wavefront_size": 32,
    }

    def test_the_measured_gfx1250_constants(self):
        from convert_csv_to_parquet import HW_PROFILES

        got = HW_PROFILES["gfx1250"]
        for key, want in self.MEASURED_GFX1250.items():
            assert got[key] == want, (
                f"gfx1250 {key} is {got[key]}, measured value is {want}"
            )

    def test_gfx1250_carries_no_unmeasured_constant(self):
        from convert_csv_to_parquet import HW_PROFILES

        assert set(HW_PROFILES["gfx1250"]) == set(self.MEASURED_GFX1250) | {
            "hw_lds_capacity"
        }

    def test_gfx950_num_cus(self):
        from convert_csv_to_parquet import HW_PROFILES

        assert HW_PROFILES["gfx950"]["hw_num_cus"] == 256
        assert HW_PROFILES["gfx950"]["hw_wavefront_size"] == 64

    def test_the_profiles_are_not_interchangeable(self):
        from convert_csv_to_parquet import HW_PROFILES

        caps = {a: p["hw_lds_capacity"] for a, p in HW_PROFILES.items()}
        assert len(set(caps.values())) > 1, caps
        assert caps["gfx1250"] != caps["gfx950"]

    def test_an_unknown_arch_emits_no_hardware_columns(self, tmp_path):
        import pandas as pd

        from convert_csv_to_parquet import HW_PROFILES, convert_csv_to_parquet

        assert "gfx1250" in HW_PROFILES, (
            "gfx1250 must have its own profile; it has 320 KB and would "
            "otherwise be stamped with gfx950's 160 KB"
        )
        csv = tmp_path / "in.csv"
        csv.write_text(
            "kernel,problem_idx,M,N,K,device,latency_ms,tflops,"
            "non_zero,max_rel,verified\n"
            "gemm_bf16_rcr_compv3_cshuffle_intrawave_0_0_0_0_"
            "128x128x64_2x2x1_32x32x16,0,512,1024,1024,0,0.5,4.3,1,1e-3,True\n"
        )
        out = tmp_path / "out.parquet"
        convert_csv_to_parquet(csv, out, arch="gfx_not_a_real_target")
        hw_cols = [c for c in pd.read_parquet(out).columns if c.startswith("hw_")]
        assert hw_cols == [], (
            f"an unknown arch was given another part's constants: {hw_cols}"
        )

    def test_a_known_arch_does_emit_them(self, tmp_path):
        import pandas as pd

        from convert_csv_to_parquet import convert_csv_to_parquet

        csv = tmp_path / "in.csv"
        csv.write_text(
            "kernel,problem_idx,M,N,K,device,latency_ms,tflops,"
            "non_zero,max_rel,verified\n"
            "gemm_bf16_rcr_compv3_cshuffle_intrawave_0_0_0_0_"
            "128x128x64_2x2x1_32x32x16,0,512,1024,1024,0,0.5,4.3,1,1e-3,True\n"
        )
        out = tmp_path / "out.parquet"
        convert_csv_to_parquet(csv, out, arch="gfx950")
        df = pd.read_parquet(out)
        assert df["hw_lds_capacity"].iloc[0] == 163840

    def test_capture_hw_refuses_a_foreign_arch(self):
        from data_pipeline import check_capture_hw_arch

        gfx950_node = {"gfx_name": "amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-"}
        with pytest.raises(SystemExit, match="gfx1250"):
            check_capture_hw_arch(gfx950_node, "gfx1250")

    def test_capture_hw_accepts_a_matching_arch(self):
        from data_pipeline import check_capture_hw_arch

        node = {"gfx_name": "amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-"}
        assert check_capture_hw_arch(node, "gfx950") is None
        assert check_capture_hw_arch(node, "gfx950:xnack-") is None

    def test_capture_hw_allows_an_unidentifiable_node(self):
        from data_pipeline import check_capture_hw_arch

        assert check_capture_hw_arch({"gfx_name": "not-a-target"}, "gfx1250") is None
        assert check_capture_hw_arch({}, "gfx1250") is None

    #: A streaming log that parses to one row.
    LOG = (
        "Shape 1: M=512 N=1024 K=1024 dtype=bf16 layout=rcr\n"
        "{\n"
        '  "name": "gemm_universal_bf16_rcr_compv3_cshuffle_intrawave_'
        '0_0_0_0_128x128x64_2x2x1_32x32x16",\n'
        '  "perf_result": { "latency(ms)": 0.5, "tflops(TFlops)": 4.3,\n'
        '                   "bandwidth(GB/s)": 100.0 }\n'
        "}\n"
    )

    def test_the_log_fixture_parses_to_a_real_row(self, tmp_path):
        from data_pipeline import parse_streaming_log

        log = tmp_path / "run.log"
        log.write_text(self.LOG)
        assert len(parse_streaming_log(log, arch="gfx950")) >= 1

    def test_main_actually_invokes_the_arch_check(self, tmp_path, monkeypatch):
        import data_pipeline as dp

        log = tmp_path / "run.log"
        log.write_text(self.LOG)
        out = tmp_path / "out.parquet"
        monkeypatch.setattr(
            dp,
            "get_hardware_profile",
            lambda: {
                "gfx_name": "amdgcn-amd-amdhsa--gfx950:xnack-",
                "num_cus": 256,
                "lds_capacity": 163840,
            },
        )
        with pytest.raises(SystemExit, match="gfx1250"):
            dp.main([str(log), "-o", str(out), "--arch", "gfx1250", "--capture_hw"])
        assert not out.exists(), "the parquet was written before the guard ran"

    def test_main_writes_the_profile_when_the_arch_matches(self, tmp_path, monkeypatch):
        import pandas as pd

        import data_pipeline as dp

        log = tmp_path / "run.log"
        log.write_text(self.LOG)
        out = tmp_path / "out.parquet"
        monkeypatch.setattr(
            dp,
            "get_hardware_profile",
            lambda: {
                "gfx_name": "amdgcn-amd-amdhsa--gfx950:xnack-",
                "num_cus": 256,
                "lds_capacity": 163840,
            },
        )
        dp.main([str(log), "-o", str(out), "--arch", "gfx950", "--capture_hw"])
        assert out.exists()
        df = pd.read_parquet(out)
        assert df["hw_lds_capacity"].iloc[0] == 163840


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
