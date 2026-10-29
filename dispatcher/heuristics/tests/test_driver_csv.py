#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for converting tile_engine/ops/gemm/gemm_full_benchmark.py CSVs into
training data (data_pipeline.parse_gemm_driver_csv, convert_csv_to_parquet).

Covers: canonical columns, kernel fields from the name (vector widths
included), the --verify filter, exact float values, bandwidth with the driver's
output dtype, hardware constants, the legacy CSV path, and that the
vector-width feature engine accepts the result.
"""

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_pipeline  # noqa: E402
from convert_csv_to_parquet import HW_PROFILES, convert_csv_to_parquet  # noqa: E402
from data_pipeline import (  # noqa: E402
    CANONICAL_COLUMNS,
    is_gemm_driver_csv,
    parse_gemm_driver_csv,
)
from feature_engine_vec import GemmUniversalVecFeatureEngine  # noqa: E402

NATIVE = "gemm_bf16_rcr_compv3_cshuffle_intrawave_False_False_False_True_256x128x64_2x2x1_32x32x16"
FIXED = "gemm_bf16_rcr_compv3_cshuffle_intrawave_True_True_True_False_128x128x64_2x2x1_32x32x16_vec2_2_8"
FIELDS = [
    "kernel",
    "problem_idx",
    "M",
    "N",
    "K",
    "device",
    "latency_ms",
    "tflops",
    "non_zero",
    "max_rel",
    "verified",
]
ROWS = [
    # kernel, idx, M, N, K, latency, tflops, verified
    (NATIVE, 0, 1024, 2048, 512, "0.0123456789012345678", "174.06529823081453", "True"),
    (FIXED, 0, 1024, 2048, 512, "0.0251", "85.5", "True"),
    (NATIVE, 1, 1024, 2048, 250, "0.02", "50.0", "False"),
    (FIXED, 1, 1024, 2048, 250, "0.03", "34.1", "True"),
]


def _write(path, rows=ROWS, verify=True):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for name, idx, m, n, k, lat, tf, ver in rows:
            w.writerow([name, idx, m, n, k, 0, lat, tf, 1, "", ver if verify else ""])
    return path


def test_canonical_rows_from_names(tmp_path):
    df = parse_gemm_driver_csv(_write(tmp_path / "s.csv"), arch="gfx950")
    assert list(df.columns) == CANONICAL_COLUMNS
    assert len(df) == 3 and df.attrs["dropped_unverified"] == 1
    native = df[df.kernel_name == NATIVE].iloc[0]
    assert (native.op_type, native["dtype"], native.layout, native.arch) == (
        "gemm_universal",
        "bf16",
        "rcr",
        "gfx950",
    )
    assert (native.tile_m, native.tile_n, native.tile_k) == (256, 128, 64)
    assert (native.warp_m, native.warp_n, native.warp_k) == (2, 2, 1)
    assert (native.warp_tile_m, native.warp_tile_n, native.warp_tile_k) == (32, 32, 16)
    assert (native.epilogue, native.pipeline, native.scheduler) == (
        "cshuffle",
        "compv3",
        "intrawave",
    )
    assert (native.pad_m, native.persistent) == (False, True)
    assert (native.vec_a, native.vec_b, native.vec_c) == (0, 0, 0)
    fixed = df[(df.kernel_name == FIXED) & (df.k == 250)].iloc[0]
    assert (fixed.vec_a, fixed.vec_b, fixed.vec_c) == (2, 2, 8)
    assert bool(df.is_valid.all()) and set(df.split_k) == {1}


def test_values_are_exact_and_bandwidth_counts_a_b_c(tmp_path):
    df = parse_gemm_driver_csv(_write(tmp_path / "s.csv"), arch="gfx950")
    row = df.iloc[0]
    assert row.measured_tflops == float("174.06529823081453")
    assert row.latency_ms == float("0.0123456789012345678")
    m, n, k = 1024, 2048, 512
    assert row.bandwidth_gb_s == pytest.approx(
        (m * k + k * n + m * n) * 2 / (row.latency_ms * 1e6)
    )


def test_fp8_output_is_counted_at_fp16(tmp_path):
    name = NATIVE.replace("gemm_bf16_", "gemm_fp8_")
    path = _write(tmp_path / "s.csv", [(name, 0, 64, 32, 16, "0.5", "1.0", "")])
    row = parse_gemm_driver_csv(path, arch="gfx950").iloc[0]
    assert row.bandwidth_gb_s == pytest.approx(
        (64 * 16 + 16 * 32 + 2 * 64 * 32) / 0.5e6
    )


def test_rows_without_verify_are_kept(tmp_path):
    df = parse_gemm_driver_csv(_write(tmp_path / "s.csv", verify=False), arch="gfx950")
    assert len(df) == 4 and df.attrs["dropped_unverified"] == 0


def test_undecodable_name_is_an_error(tmp_path):
    path = _write(
        tmp_path / "s.csv", [("gemm_bf16_rcr_broken", 0, 1, 1, 1, "1", "1", "")]
    )
    with pytest.raises(ValueError, match="do not decode"):
        parse_gemm_driver_csv(path, arch="gfx950")


def test_detection():
    assert is_gemm_driver_csv(FIELDS, NATIVE)
    assert not is_gemm_driver_csv(FIELDS, "grouped_conv_fwd_bf16_2d_256x128x128_compv3")
    assert not is_gemm_driver_csv(["kernel", "M", "N", "K"], NATIVE)


def test_converter_adds_the_arch_hardware_constants(tmp_path):
    out = tmp_path / "data" / "train.parquet"
    convert_csv_to_parquet(_write(tmp_path / "s.csv"), out, arch="gfx950")
    df = pd.read_parquet(out)
    assert set(CANONICAL_COLUMNS) <= set(df.columns) and len(df) == 3
    for col, value in HW_PROFILES["gfx950"].items():
        assert set(df[col]) == {value}


def test_other_csvs_keep_the_generic_path(tmp_path):
    path = tmp_path / "conv.csv"
    pd.DataFrame(
        {
            "kernel": ["grouped_conv_fwd_bf16_2d_256x128x128_compv3"],
            "N": [1],
            "latency_ms": [1.0],
            "tflops": [2.0],
        }
    ).to_csv(path, index=False)
    df = convert_csv_to_parquet(path, tmp_path / "conv.parquet", arch="gfx950")
    assert "tflops" in df.columns and "measured_tflops" not in df.columns


def test_data_pipeline_cli_accepts_a_driver_csv(tmp_path):
    out = tmp_path / "x.parquet"
    data_pipeline.main(
        [str(_write(tmp_path / "s.csv")), "-o", str(out), "--arch", "gfx950"]
    )
    assert len(pd.read_parquet(out)) == 3


def test_vector_width_engine_reads_the_converted_rows(tmp_path):
    out = tmp_path / "train.parquet"
    df = convert_csv_to_parquet(_write(tmp_path / "s.csv"), out, arch="gfx950")
    engine = GemmUniversalVecFeatureEngine(
        lds_capacity=HW_PROFILES["gfx950"]["hw_lds_capacity"]
    )
    X = engine.extract_batch(df)
    assert X.shape == (len(df), len(engine.get_feature_names()))
