"""Tests for BatchedGemmFeatureEngine."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HEURISTICS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HEURISTICS))

from feature_engine import (  # noqa: E402
    GemmUniversalFeatureEngine,
    OPERATION_ENGINES,
)
from feature_engine_batched import (  # noqa: E402
    BATCHED_FEATURES,
    BatchedGemmFeatureEngine,
)
from feature_engine_vec import _native_widths  # noqa: E402


def _row(batch_count=8, vec=(0, 0, 0), **over):
    row = {
        "op_type": "batched_gemm",
        "dtype": "bf16",
        "layout": "rcr",
        "arch": "gfx950",
        "m": 16,
        "n": 32,
        "k": 64,
        "split_k": 1,
        "batch_count": batch_count,
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_k": 1,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": "compv3",
        "epilogue": "default",
        "scheduler": "intrawave",
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "persistent": False,
        "vec_a": vec[0],
        "vec_b": vec[1],
        "vec_c": vec[2],
        "is_fixed_width": any(v != 0 for v in vec),
    }
    row.update(over)
    return pd.DataFrame([row])


class TestRegistration:
    def test_the_operation_selects_this_engine(self):
        assert OPERATION_ENGINES["batched_gemm"] == "BatchedGemmFeatureEngine"

    def test_feature_names_append_the_batched_block(self):
        base = GemmUniversalFeatureEngine().get_feature_names()
        names = BatchedGemmFeatureEngine().get_feature_names()
        assert names[: len(base)] == base
        assert names[len(base) :] == BATCHED_FEATURES
        assert len(BATCHED_FEATURES) == 17


class TestBatchOccupancy:
    def test_universal_features_ignore_batch_and_the_extras_do_not(self):
        engine = BatchedGemmFeatureEngine()
        small = engine.extract_batch(_row(batch_count=4))
        large = engine.extract_batch(_row(batch_count=16))
        n_base = len(GemmUniversalFeatureEngine().get_feature_names())
        np.testing.assert_allclose(small[0, :n_base], large[0, :n_base])
        names = engine.get_feature_names()
        workgroups = names.index("total_workgroups")
        assert large[0, workgroups] == pytest.approx(small[0, workgroups] * 4)

    def test_extract_matches_extract_batch(self):
        engine = BatchedGemmFeatureEngine()
        frame = _row()
        row = frame.iloc[0].to_dict()
        np.testing.assert_allclose(
            engine.extract(row, row), engine.extract_batch(frame)[0]
        )

    def test_missing_batch_count_raises(self):
        frame = _row().drop(columns=["batch_count"])
        with pytest.raises(KeyError, match="batch_count"):
            BatchedGemmFeatureEngine().extract_batch(frame)

    def test_batch_below_one_raises(self):
        with pytest.raises(ValueError, match="batch_count"):
            BatchedGemmFeatureEngine().extract_batch(_row(batch_count=0))


class TestVectorWidths:
    def test_a_native_width_is_resolved_rather_than_left_at_zero(self):
        engine = BatchedGemmFeatureEngine()
        frame = _row(vec=(0, 0, 0))
        values = engine.extract_batch(frame)[0]
        names = engine.get_feature_names()
        native = _native_widths(
            "bf16",
            "rcr",
            "gfx950",
            "default",
            ((128, 128, 64), (2, 2, 1), (32, 32, 16)),
        )
        assert native[0] > 0
        assert values[names.index("vec_a")] == pytest.approx(native[0])
        assert values[names.index("is_fixed_width")] == 0

    def test_an_explicit_width_is_kept(self):
        engine = BatchedGemmFeatureEngine()
        values = engine.extract_batch(_row(vec=(1, 2, 4)))[0]
        names = engine.get_feature_names()
        assert values[names.index("vec_a")] == pytest.approx(1)
        assert values[names.index("vec_b")] == pytest.approx(2)
        assert values[names.index("vec_c")] == pytest.approx(4)
        assert values[names.index("vec_min_ab")] == pytest.approx(1)
        assert values[names.index("is_fixed_width")] == 1
