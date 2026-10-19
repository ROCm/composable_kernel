#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Feature engine for batched GEMM.

``GemmUniversalFeatureEngine`` scores one GEMM: its ``cu_utilization`` is
``ceil(M/tile_m) * ceil(N/tile_n) / num_cus`` and does not see ``batch_count``.
Batched problems are many small GEMMs, so occupancy is dominated by the batch.
This engine keeps the universal features and appends batch, workgroup, and
vector-width features.

``vec_a/b/c`` in the data follow ``data_pipeline.parse_kernel_name``: 0 means
the kernel's native width. The features expose the width the kernel actually
uses (native when the stored width is 0). ``is_fixed_width`` is 1 when the
kernel name carried an explicit ``_vec`` suffix.
"""

import numpy as np
import pandas as pd

from feature_engine import GemmUniversalFeatureEngine
from feature_engine_vec import (
    _GEOMETRY_KEYS,
    _LAYOUTS,
    _dtype_max_width,
    _native_widths,
    _out_dtype,
    _width,
    gemm_contiguous_dims,
)


BATCHED_FEATURES = [
    "batch_count",
    "log2_batch",
    "log2_total_flops",
    "total_workgroups",
    "workgroups_per_cu",
    "last_wave_fill",
    "vec_a",
    "vec_b",
    "vec_c",
    "vec_min_ab",
    "is_fixed_width",
    "prob_vec_a",
    "prob_vec_b",
    "prob_vec_c",
    "contig_extent_a",
    "contig_extent_b",
    "contig_extent_c",
]

_INT_FIELDS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
)


def observed_search_space(df: pd.DataFrame) -> dict:
    """Kernel parameter values present in ``df``.

    Saved with a batched model so selection enumerates only configurations the
    model was trained on.
    """
    needed = (
        *_INT_FIELDS,
        "epilogue",
        "persistent",
        "pipeline",
        "scheduler",
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"cannot record a search space without columns {missing}. "
            "They come from data_pipeline.parse_kernel_name."
        )
    space = {f: sorted(int(v) for v in df[f].dropna().unique()) for f in _INT_FIELDS}
    space["epilogue"] = sorted(str(v) for v in df["epilogue"].dropna().unique())
    space["persistent"] = sorted(bool(v) for v in df["persistent"].dropna().unique())
    pipe_sched = df.groupby("pipeline")["scheduler"].unique()
    space["pipeline_scheduler"] = {
        str(p): sorted(str(x) for x in s) for p, s in pipe_sched.items()
    }
    return space


def _native_columns(df: pd.DataFrame) -> np.ndarray:
    """Per-row native A/B/C widths, resolved once per distinct geometry."""
    keys = [
        "layout",
        "dtype",
        "arch",
        "epilogue",
        *(k for group in _GEOMETRY_KEYS for k in group),
    ]
    combos = df[keys].astype({k: np.int64 for group in _GEOMETRY_KEYS for k in group})
    codes, uniques = pd.factorize(
        pd.MultiIndex.from_frame(combos), use_na_sentinel=False
    )
    resolved = np.empty((len(uniques), 3), dtype=np.int64)
    for j, row in enumerate(uniques):
        geom = tuple(
            tuple(int(row[4 + 3 * g + o]) for o in range(3)) for g in range(3)
        )
        resolved[j] = _native_widths(
            str(row[1]), str(row[0]), str(row[2]), str(row[3]), geom
        )
    return resolved[codes]


class BatchedGemmFeatureEngine(GemmUniversalFeatureEngine):
    """Universal GEMM features plus batch occupancy and vector width."""

    ENCODING_VERSION = 1

    #: ``super().extract_batch`` sizes its output from the overridden
    #: ``get_feature_names()`` and only fills the leading universal columns.
    _N_BASE = len(GemmUniversalFeatureEngine().get_feature_names())

    def get_feature_names(self) -> list[str]:
        return GemmUniversalFeatureEngine.get_feature_names(self) + BATCHED_FEATURES

    def extract(self, problem: dict, kernel: dict) -> np.ndarray:
        return self.extract_batch(pd.DataFrame([{**problem, **kernel}]))[0]

    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.zeros((0, len(self.get_feature_names())), dtype=np.float64)

        df = df.copy()
        for col in ("m", "n", "k"):
            if col not in df.columns and col.upper() in df.columns:
                df[col] = df[col.upper()]
        if "split_k" not in df.columns:
            df["split_k"] = 1

        required = (
            "m",
            "n",
            "k",
            "batch_count",
            "layout",
            "dtype",
            "arch",
            "epilogue",
            "vec_a",
            "vec_b",
            "vec_c",
            "tile_m",
            "tile_n",
            *(k for group in _GEOMETRY_KEYS for k in group),
        )
        for col in required:
            if col not in df.columns:
                raise KeyError(
                    f"{col!r} is required by {type(self).__name__}. "
                    "batch_count comes from the benchmark CSV; vec_a/b/c and the "
                    "tile columns come from data_pipeline.parse_kernel_name "
                    "(0 means native)."
                )
            if df[col].isna().any():
                raise ValueError(f"column {col!r} contains null")

        layout = df["layout"].astype(str)
        unknown = sorted(set(layout) - _LAYOUTS)
        if unknown:
            raise ValueError(f"unknown layout(s) {unknown}; expected {sorted(_LAYOUTS)}")

        dims = {d: df[d].to_numpy(dtype=np.int64) for d in ("m", "n", "k")}
        if (np.column_stack([dims["m"], dims["n"], dims["k"]]) < 1).any():
            raise ValueError("m, n and k must be >= 1")
        batch = df["batch_count"].to_numpy(dtype=np.float64)
        if (batch < 1).any():
            raise ValueError("batch_count must be >= 1")

        base = super().extract_batch(df)[:, : self._N_BASE]

        m = dims["m"].astype(np.float64)
        n = dims["n"].astype(np.float64)
        k = dims["k"].astype(np.float64)
        tile_m = df["tile_m"].to_numpy(dtype=np.float64)
        tile_n = df["tile_n"].to_numpy(dtype=np.float64)
        workgroups = (
            batch
            * np.ceil(m / np.maximum(tile_m, 1.0))
            * np.ceil(n / np.maximum(tile_n, 1.0))
        )
        waves = workgroups / float(self._hw["num_cus"])
        last_fill = waves / np.maximum(np.ceil(waves), 1.0)

        native = _native_columns(df)
        raw = np.column_stack(
            [
                np.array([_width(v) for v in df["vec_" + operand]], dtype=np.int64)
                for operand in ("a", "b", "c")
            ]
        )
        effective = np.where(raw == 0, native, raw).astype(np.float64)
        if "is_fixed_width" in df.columns:
            if df["is_fixed_width"].isna().any():
                raise ValueError("column 'is_fixed_width' contains null")
            fixed = df["is_fixed_width"].astype(float).to_numpy()
        else:
            fixed = (raw != 0).any(axis=1).astype(np.float64)

        dtypes = df["dtype"].astype(str)
        caps = (
            np.array([_dtype_max_width(d) for d in dtypes], dtype=np.int64),
            np.array([_dtype_max_width(d) for d in dtypes], dtype=np.int64),
            np.array(
                [_dtype_max_width(_out_dtype(d)) for d in dtypes], dtype=np.int64
            ),
        )
        extents = np.empty((len(df), 3), dtype=np.float64)
        prob_vec = np.empty((len(df), 3), dtype=np.float64)
        for i, cap in enumerate(caps):
            which = layout.map(lambda s, i=i: gemm_contiguous_dims(s)[i]).to_numpy()
            extent = np.select(
                [which == "m", which == "n", which == "k"],
                [dims["m"], dims["n"], dims["k"]],
                default=-1,
            )
            if (extent < 0).any():
                raise AssertionError("gemm_contiguous_dims returned a dim outside m/n/k")
            extents[:, i] = extent
            prob_vec[:, i] = np.gcd(extent.astype(np.int64), cap)

        extra = np.column_stack(
            [
                batch,
                np.log2(np.maximum(batch, 1.0)),
                np.log2(np.maximum(2.0 * batch * m * n * k, 1.0)),
                workgroups,
                waves,
                last_fill,
                effective,
                np.minimum(effective[:, 0], effective[:, 1]),
                fixed,
                prob_vec,
                extents,
            ]
        )
        return np.hstack([base, extra])
