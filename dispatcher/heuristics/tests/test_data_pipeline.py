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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
