#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for train.py.

Covers: group key computation, TFLOPS efficiency calculation, edge cases
(single group, all-invalid data, tied predictions), and warm-start
incremental training (feature compat, lineage, quality).
"""

import json
import sys
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engine import GemmUniversalFeatureEngine
from train import (
    compute_group_keys,
    compute_tflops_efficiency,
    check_feature_compatibility,
    load_warm_start_model,
    train_final_model,
    DEFAULT_PARAMS,
)


class TestComputeGroupKeys:
    def test_basic(self):
        df = pd.DataFrame(
            {"m": [16, 16, 32], "n": [1536, 1536, 1536], "k": [7168, 7168, 7168]}
        )
        keys = compute_group_keys(df, "gemm_universal")
        assert keys[0] == keys[1]
        assert keys[0] != keys[2]

    def test_unique_shapes(self):
        df = pd.DataFrame({"m": [1, 2, 3], "n": [4, 5, 6], "k": [7, 8, 9]})
        keys = compute_group_keys(df, "gemm_universal")
        assert len(set(keys)) == 3


class TestComputeTflopsEfficiency:
    def test_perfect_prediction(self):
        """Model predicts highest TFLOPS kernel => efficiency = 1.0."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 150],
                "pred_tflops": [50, 300, 100],  # correctly ranks kernel 1 highest
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] == pytest.approx(1.0)

    def test_worst_prediction(self):
        """Model picks the worst kernel."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 150],
                "pred_tflops": [999, 1, 1],  # incorrectly ranks kernel 0 highest
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert eff["efficiency"].iloc[0] == pytest.approx(100 / 200)

    def test_multiple_shapes(self):
        df = pd.DataFrame(
            {
                "m": [16, 16, 32, 32],
                "n": [1536, 1536, 1536, 1536],
                "k": [7168, 7168, 7168, 7168],
                "measured_tflops": [10, 20, 100, 200],
                "pred_tflops": [5, 25, 150, 190],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 2
        assert eff.iloc[0]["efficiency"] == pytest.approx(1.0)
        assert eff.iloc[1]["efficiency"] == pytest.approx(1.0)

    def test_zero_tflops_shape_skipped(self):
        df = pd.DataFrame(
            {
                "m": [16, 16],
                "n": [16, 16],
                "k": [16, 16],
                "measured_tflops": [0, 0],
                "pred_tflops": [1, 2],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 0

    def test_single_kernel_per_shape(self):
        df = pd.DataFrame(
            {
                "m": [1024],
                "n": [1024],
                "k": [1024],
                "measured_tflops": [150],
                "pred_tflops": [100],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] == pytest.approx(1.0)

    def test_tied_predictions(self):
        """When multiple kernels have the same predicted TFLOPS, pandas idxmax picks the first."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 200],
                "pred_tflops": [50, 50, 50],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] >= 0.5


# ---------------------------------------------------------------------------
# Helpers for warm-start tests
# ---------------------------------------------------------------------------


def _make_dummy_data(n_rows=200, n_shapes=5):
    """Create a small synthetic benchmark DataFrame for testing training."""
    rng = np.random.RandomState(42)
    rows = []
    for _ in range(n_rows):
        m = rng.choice([64, 128, 256, 512, 1024])
        n = rng.choice([64, 128, 256, 512, 1024])
        k = rng.choice([64, 128, 256, 512, 1024])
        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "split_k": 1,
                "dtype": "fp8",
                "layout": "rcr",
                "op_type": "gemm_universal",
                "tile_m": rng.choice([64, 128, 256]),
                "tile_n": rng.choice([64, 128, 256]),
                "tile_k": rng.choice([32, 64, 128]),
                "warp_m": rng.choice([1, 2, 4]),
                "warp_n": rng.choice([1, 2, 4]),
                "warp_k": 1,
                "warp_tile_m": 32,
                "warp_tile_n": 32,
                "warp_tile_k": 16,
                "pipeline": rng.choice(["compv3", "compv4", "mem"]),
                "scheduler": rng.choice(["intrawave", "interwave"]),
                "epilogue": "cshuffle",
                "pad_m": False,
                "pad_n": False,
                "pad_k": False,
                "persistent": False,
                "measured_tflops": float(rng.uniform(10, 500)),
                "latency_ms": float(rng.uniform(0.01, 1.0)),
                "bandwidth_gb_s": float(rng.uniform(50, 1500)),
                "is_valid": True,
                "kernel_name": f"test_kernel_{rng.randint(0, 100)}",
            }
        )
    return pd.DataFrame(rows)


def _save_feature_spec(model_dir, fe):
    """Save a feature_spec.json matching the given feature engine."""
    spec = {
        "feature_names": fe.get_feature_names(),
        "categorical_features": fe.get_categorical_features(),
    }
    with open(model_dir / "feature_spec.json", "w") as f:
        json.dump(spec, f)


def _train_and_save_base_model(model_dir, df, fe, target="tflops"):
    """Train a small base model and save it to model_dir."""
    params = dict(DEFAULT_PARAMS)
    params["n_estimators"] = 20
    params["n_jobs"] = 1
    model = train_final_model(df, fe, target, params, "gemm_universal")
    model.booster_.save_model(str(model_dir / f"model_{target}.lgbm"))
    _save_feature_spec(model_dir, fe)
    return model


# ---------------------------------------------------------------------------
# Warm-start tests
# ---------------------------------------------------------------------------


class TestCheckFeatureCompatibility:
    def test_compatible_passes(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        _save_feature_spec(tmp_path, fe)
        check_feature_compatibility(tmp_path, fe)

    def test_missing_spec_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        with pytest.raises(FileNotFoundError, match="feature_spec.json"):
            check_feature_compatibility(tmp_path, fe)

    def test_added_feature_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names()[:-1],
            "categorical_features": fe.get_categorical_features(),
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Feature schema mismatch"):
            check_feature_compatibility(tmp_path, fe)

    def test_removed_feature_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names() + ["extra_feature"],
            "categorical_features": fe.get_categorical_features(),
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Feature schema mismatch"):
            check_feature_compatibility(tmp_path, fe)

    def test_categorical_mismatch_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names(),
            "categorical_features": ["layout", "pipeline"],
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Categorical feature mismatch"):
            check_feature_compatibility(tmp_path, fe)


class TestLoadWarmStartModel:
    def test_loads_existing_model(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data()
        _train_and_save_base_model(tmp_path, df, fe)
        path = load_warm_start_model(tmp_path, "tflops")
        assert path is not None
        assert Path(path).exists()

    def test_returns_none_for_missing_target(self, tmp_path):
        assert load_warm_start_model(tmp_path, "tflops") is None

    def test_returns_none_for_wrong_target(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data()
        _train_and_save_base_model(tmp_path, df, fe, target="tflops")
        assert load_warm_start_model(tmp_path, "bandwidth") is None


class TestWarmStartTraining:
    def test_warm_start_produces_more_trees(self, tmp_path):
        """A warm-started model should have more trees than the base."""
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data(n_rows=300)

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_model = _train_and_save_base_model(base_dir, df, fe)
        base_n_trees = base_model.booster_.num_trees()

        init_model_path = load_warm_start_model(base_dir, "tflops")
        params = dict(DEFAULT_PARAMS)
        params["n_estimators"] = 15
        params["n_jobs"] = 1
        warm_model = train_final_model(
            df, fe, "tflops", params, "gemm_universal", init_model=init_model_path
        )
        warm_n_trees = warm_model.booster_.num_trees()

        assert warm_n_trees > base_n_trees

    def test_warm_start_does_not_degrade(self, tmp_path):
        """Warm-started model on the same data should not be significantly worse."""
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data(n_rows=300)

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_model = _train_and_save_base_model(base_dir, df, fe)

        X = fe.extract_batch(df[df["is_valid"]].reset_index(drop=True))
        y = df[df["is_valid"]]["measured_tflops"].values
        base_rmse = np.sqrt(np.mean((base_model.predict(X) - y) ** 2))

        init_model_path = load_warm_start_model(base_dir, "tflops")
        params = dict(DEFAULT_PARAMS)
        params["n_estimators"] = 15
        params["n_jobs"] = 1
        warm_model = train_final_model(
            df, fe, "tflops", params, "gemm_universal", init_model=init_model_path
        )
        warm_rmse = np.sqrt(np.mean((warm_model.predict(X) - y) ** 2))

        assert warm_rmse <= base_rmse * 1.1

    def test_warm_start_from_nonexistent_dir(self, tmp_path):
        missing = tmp_path / "model" / "dir"
        assert not missing.exists()
        with pytest.raises(FileNotFoundError):
            check_feature_compatibility(missing, GemmUniversalFeatureEngine())


class TestEngineVariants:
    def test_base_operation_maps_the_variant(self):
        from train import base_operation

        assert base_operation("gemm_universal_vec") == "gemm_universal"

    def test_base_operation_is_identity_for_real_operations(self):
        from train import base_operation

        for op in ("gemm_universal", "grouped_conv", "fmha"):
            assert base_operation(op) == op

    def test_factory_returns_the_wider_engine(self):
        from train import get_feature_engine

        base = get_feature_engine("gemm_universal")
        vec = get_feature_engine("gemm_universal_vec")
        assert type(vec).__name__ == "GemmUniversalVecFeatureEngine"
        assert len(vec.get_feature_names()) == len(base.get_feature_names()) + 6

    def test_target_columns_resolve_for_the_variant(self):
        from train import TARGET_COLUMNS, base_operation

        assert "tflops" in TARGET_COLUMNS[base_operation("gemm_universal_vec")]

    @staticmethod
    def _frame():
        return pd.DataFrame(
            {
                "m": [128, 128, 256],
                "n": [256, 256, 512],
                "k": [512, 512, 1024],
                "measured_tflops": [10.0, 5.0, 8.0],
                "pred_tflops": [9.0, 6.0, 7.0],
            }
        )

    def test_group_keys_normalise_the_variant(self):
        from train import compute_group_keys

        df = self._frame()
        np.testing.assert_array_equal(
            compute_group_keys(df, "gemm_universal_vec"),
            compute_group_keys(df, "gemm_universal"),
        )

    def test_efficiency_normalises_the_variant(self):
        from train import compute_tflops_efficiency

        df = self._frame()
        pd.testing.assert_frame_equal(
            compute_tflops_efficiency(df, "gemm_universal_vec", "pred_tflops"),
            compute_tflops_efficiency(df, "gemm_universal", "pred_tflops"),
        )

    def test_parser_accepts_the_variant_and_rejects_junk(self):
        import train

        argv = [
            "--data_dir",
            "d",
            "--out_dir",
            "o",
            "--operation",
            "gemm_universal_vec",
        ]
        with mock.patch.object(sys, "argv", ["train.py", *argv]):
            parser = train.build_arg_parser()
            assert parser.parse_args(argv).operation == "gemm_universal_vec"
            with pytest.raises(SystemExit):
                parser.parse_args(
                    [
                        "--data_dir",
                        "d",
                        "--out_dir",
                        "o",
                        "--operation",
                        "definitely_not_an_operation",
                    ]
                )


class TestFeatureSpecRecordsTheEngine:
    @staticmethod
    def _spec(operation):
        from train import build_feature_spec, get_feature_engine

        fe = get_feature_engine(operation)
        return fe, build_feature_spec(
            operation, fe, "bf16", "gfx950", ["tflops"], ["tflops"], {}
        )

    @pytest.mark.parametrize("operation", ["gemm_universal", "gemm_universal_vec"])
    def test_round_trip_through_predict(self, tmp_path, operation):
        import json

        from predict import _engine_for_spec

        fe, spec = self._spec(operation)
        path = tmp_path / "feature_spec.json"
        path.write_text(json.dumps(spec))
        assert type(_engine_for_spec(json.loads(path.read_text()))) is type(fe)

    def test_the_spec_records_the_two_engines_distinctly(self):
        assert (
            self._spec("gemm_universal")[1]["feature_engine"]
            != self._spec("gemm_universal_vec")[1]["feature_engine"]
        )

    def test_the_spec_feature_names_match_the_engine(self):
        for operation in ("gemm_universal", "gemm_universal_vec"):
            fe, spec = self._spec(operation)
            assert spec["feature_names"] == fe.get_feature_names(), operation

    def test_the_two_operations_give_different_engines(self):
        from train import get_feature_engine

        assert type(get_feature_engine("gemm_universal")) is not type(
            get_feature_engine("gemm_universal_vec")
        )


class TestVariantReachesTheRunners:
    @staticmethod
    def _frame(n=24):
        rs = np.random.RandomState(0)
        return pd.DataFrame(
            {
                "m": rs.choice([128, 256, 512], n),
                "n": rs.choice([256, 512], n),
                "k": rs.choice([512, 1024], n),
                "measured_tflops": rs.uniform(1.0, 100.0, n),
                "is_valid": True,
            }
        )

    class _Engine:
        def get_feature_names(self):
            return ["m", "n", "k"]

        def get_categorical_features(self):
            return []

        def extract_batch(self, df):
            return df[["m", "n", "k"]].to_numpy(dtype=float)

    def test_run_cv_accepts_the_variant(self):
        from train import run_cv

        out = run_cv(
            self._frame(),
            self._Engine(),
            target="tflops",
            params={"n_estimators": 2, "verbose": -1},
            operation="gemm_universal_vec",
            n_splits=2,
        )
        assert out is not None

    def test_train_final_model_accepts_the_variant(self):
        from train import train_final_model

        out = train_final_model(
            self._frame(),
            self._Engine(),
            target="tflops",
            params={"n_estimators": 2, "verbose": -1},
            operation="gemm_universal_vec",
        )
        assert out is not None

    def test_the_dataset_is_loaded_under_the_base_operation(self):
        import train

        seen = {}

        def fake(data_dir, op_type=None, dtype=None, **kw):
            seen["op_type"] = op_type
            raise SystemExit  # stop before training; we only need the kwarg

        with mock.patch.object(train, "build_training_dataset", fake):
            argv = [
                "train.py",
                "--data_dir",
                "d",
                "--out_dir",
                "o",
                "--operation",
                "gemm_universal_vec",
                "--targets",
                "tflops",
            ]
            with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit):
                train.main()
        assert seen["op_type"] == "gemm_universal"


class TestEncodingVersion:
    @staticmethod
    def _spec_dir(tmp_path, engine, **override):
        import json

        from train import build_feature_spec

        spec = build_feature_spec(
            "gemm_universal_vec", engine, "bf16", "gfx950", ["tflops"], ["tflops"], {}
        )
        spec.update(override)
        for k in [k for k, v in override.items() if v is None]:
            spec.pop(k)
        (tmp_path / "feature_spec.json").write_text(json.dumps(spec))
        return tmp_path

    def test_the_engines_declare_distinct_versions(self):
        from feature_engine import GemmUniversalFeatureEngine
        from feature_engine_vec import GemmUniversalVecFeatureEngine

        assert GemmUniversalFeatureEngine.ENCODING_VERSION == 0
        assert GemmUniversalVecFeatureEngine.ENCODING_VERSION == 1

    @pytest.mark.parametrize(
        "operation,expected",
        [("gemm_universal", 0), ("gemm_universal_vec", 1), ("grouped_conv", 0)],
    )
    def test_the_spec_records_the_engines_version(self, operation, expected):
        from train import build_feature_spec, get_feature_engine

        fe = get_feature_engine(operation)
        spec = build_feature_spec(
            operation, fe, "bf16", "gfx950", ["tflops"], ["tflops"], {}
        )
        assert spec["feature_encoding_version"] == expected
        assert spec["feature_encoding_version"] == fe.ENCODING_VERSION

    @pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx1250"])
    def test_the_spec_records_the_arch(self, arch):
        from train import build_feature_spec, get_feature_engine

        spec = build_feature_spec(
            "gemm_universal_vec",
            get_feature_engine("gemm_universal_vec"),
            "bf16",
            arch,
            ["tflops"],
            ["tflops"],
            {},
        )
        assert spec["arch"] == arch

    def test_a_matching_version_warm_starts(self, tmp_path):
        from feature_engine_vec import GemmUniversalVecFeatureEngine
        from train import check_feature_compatibility

        fe = GemmUniversalVecFeatureEngine()
        check_feature_compatibility(self._spec_dir(tmp_path, fe), fe)

    def test_a_spec_with_no_version_is_refused(self, tmp_path):
        from feature_engine_vec import GemmUniversalVecFeatureEngine
        from train import check_feature_compatibility

        fe = GemmUniversalVecFeatureEngine()
        d = self._spec_dir(tmp_path, fe, feature_encoding_version=None)
        with pytest.raises(ValueError, match="encoding version 0"):
            check_feature_compatibility(d, fe)

    def test_a_newer_version_is_refused(self, tmp_path):
        from feature_engine_vec import GemmUniversalVecFeatureEngine
        from train import check_feature_compatibility

        fe = GemmUniversalVecFeatureEngine()
        d = self._spec_dir(tmp_path, fe, feature_encoding_version=99)
        with pytest.raises(ValueError, match="encoding"):
            check_feature_compatibility(d, fe)

    def test_the_base_engine_still_warm_starts_from_a_version_less_spec(self, tmp_path):
        import json

        from feature_engine import GemmUniversalFeatureEngine
        from train import check_feature_compatibility

        fe = GemmUniversalFeatureEngine()
        (tmp_path / "feature_spec.json").write_text(
            json.dumps(
                {
                    "feature_names": fe.get_feature_names(),
                    "categorical_features": fe.get_categorical_features(),
                }
            )
        )
        check_feature_compatibility(tmp_path, fe)


class TestHardwareConstantsRoundTrip:
    #: gfx950 constants; only lds_capacity differs from DEFAULTS.
    HW = {"lds_capacity": 163840, "num_cus": 256, "num_xcd": 8}

    #: GemmUniversalFeatureEngine constructor defaults.
    DEFAULTS = {
        "num_cus": 256,
        "lds_capacity": 65536,
        "max_clock_mhz": 2400,
        "simds_per_cu": 4,
        "shader_engines": 32,
        "max_waves_per_cu": 32,
        "wavefront_size": 64,
        "l1_cache_kb": 32,
        "l2_cache_kb": 4096,
        "l3_cache_kb": 262144,
        "num_xcd": 8,
    }

    def _frame(self, **hw):
        from feature_engine import GemmUniversalFeatureEngine

        row = {
            "m": 1024,
            "n": 1024,
            "k": 1024,
            "split_k": 1,
            "layout": "rcr",
            "dtype": "bf16",
            "pipeline": "compv3",
            "epilogue": "cshuffle",
            "scheduler": "intrawave",
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 64,
            "warp_m": 2,
            "warp_n": 2,
            "warp_k": 1,
            "warp_tile_m": 32,
            "warp_tile_n": 32,
            "warp_tile_k": 16,
            "pad_m": False,
            "pad_n": False,
            "pad_k": False,
            "persistent": False,
            "is_valid": True,
            "measured_tflops": 1.0,
        }
        row.update({f"hw_{k}": v for k, v in hw.items()})
        return pd.DataFrame([row]), GemmUniversalFeatureEngine

    def test_training_refuses_an_arch_the_data_contradicts(self):
        from train import check_arch_matches_data

        df, _ = self._frame(**self.HW)
        df["arch"] = "gfx950"
        with pytest.raises(ValueError, match="but the data carries"):
            check_arch_matches_data(df, "gfx1250")

    def test_main_actually_invokes_the_arch_check(self):
        import train

        df, _ = self._frame(**self.HW)
        df["arch"] = "gfx950"

        with mock.patch.object(
            train, "build_training_dataset", lambda *a, **k: df.copy()
        ):
            argv = [
                "train.py",
                "--data_dir",
                "d",
                "--out_dir",
                "o",
                "--operation",
                "gemm_universal",
                "--arch",
                "gfx1250",
                "--targets",
                "tflops",
            ]
            with mock.patch.object(sys, "argv", argv):
                with pytest.raises(ValueError, match="but the data carries"):
                    train.main()

    def test_training_accepts_a_matching_arch(self):
        from train import check_arch_matches_data

        df, _ = self._frame(**self.HW)
        df["arch"] = "gfx950"
        assert check_arch_matches_data(df, "gfx950") is None
        assert check_arch_matches_data(df, "gfx950:xnack-") is None

    def test_training_refuses_a_mixed_arch_dataset(self):
        from train import check_arch_matches_data

        df, _ = self._frame(**self.HW)
        mixed = pd.concat([df, df], ignore_index=True)
        mixed["arch"] = ["gfx950", "gfx1250"]
        with pytest.raises(ValueError, match="but the data carries"):
            check_arch_matches_data(mixed, "gfx950")

    def test_training_tolerates_data_without_an_arch_column(self):
        from train import check_arch_matches_data

        df, _ = self._frame(**self.HW)
        assert "arch" not in df.columns
        assert check_arch_matches_data(df, "gfx1250") is None

    def test_warm_start_rejects_a_changed_constant(self, tmp_path):
        import json

        from feature_engine import GemmUniversalFeatureEngine
        from train import build_feature_spec, check_feature_compatibility

        spec = build_feature_spec(
            "gemm_universal",
            GemmUniversalFeatureEngine(),
            "bf16",
            "gfx950",
            ["tflops"],
            [],
            {},
        )
        spec.pop("hardware")  # a model written before the key existed
        (tmp_path / "feature_spec.json").write_text(json.dumps(spec))

        with pytest.raises(ValueError, match="Hardware constant mismatch"):
            check_feature_compatibility(
                tmp_path, GemmUniversalFeatureEngine(lds_capacity=163840)
            )

    def test_warm_start_reads_the_recorded_constants(self, tmp_path):
        import json

        from feature_engine import GemmUniversalFeatureEngine
        from train import build_feature_spec, check_feature_compatibility

        spec = build_feature_spec(
            "gemm_universal",
            GemmUniversalFeatureEngine(lds_capacity=163840),
            "bf16",
            "gfx950",
            ["tflops"],
            [],
            {},
        )
        assert spec["hardware"]["lds_capacity"] == 163840
        (tmp_path / "feature_spec.json").write_text(json.dumps(spec))

        with pytest.raises(ValueError, match="Hardware constant mismatch"):
            check_feature_compatibility(tmp_path, GemmUniversalFeatureEngine())

        check_feature_compatibility(
            tmp_path, GemmUniversalFeatureEngine(lds_capacity=163840)
        )

    def test_warm_start_accepts_matching_constants(self, tmp_path):
        import json

        from feature_engine import GemmUniversalFeatureEngine
        from train import build_feature_spec, check_feature_compatibility

        spec = build_feature_spec(
            "gemm_universal",
            GemmUniversalFeatureEngine(),
            "bf16",
            "gfx950",
            ["tflops"],
            [],
            {},
        )
        spec.pop("hardware")
        (tmp_path / "feature_spec.json").write_text(json.dumps(spec))
        check_feature_compatibility(tmp_path, GemmUniversalFeatureEngine())

    def test_build_hw_kwargs_keys_on_the_engine_actually_built(self, monkeypatch):
        import feature_engine as fe_mod
        from feature_engine import GemmUniversalFeatureEngine
        from train import build_hw_kwargs

        class VariantEngine(GemmUniversalFeatureEngine):
            def __init__(self, invented_constant: int = 1, lds_capacity: int = 65536):
                super().__init__(lds_capacity=lds_capacity)
                self._hw["invented_constant"] = invented_constant

        real = fe_mod.feature_engine_class
        monkeypatch.setattr(
            fe_mod,
            "feature_engine_class",
            lambda name: VariantEngine if name == "VariantEngine" else real(name),
        )
        monkeypatch.setitem(
            fe_mod.OPERATION_ENGINES, "gemm_universal_vec", "VariantEngine"
        )

        df, _ = self._frame(**self.HW)
        df["hw_invented_constant"] = 11
        assert build_hw_kwargs(df, "gemm_universal_vec")["invented_constant"] == 11

    @pytest.mark.parametrize("op", ["gemm_universal", "gemm_universal_vec"])
    def test_every_constructor_constant_is_read(self, op):
        import inspect

        from feature_engine import OPERATION_ENGINES, feature_engine_class
        from train import build_hw_kwargs

        engine_cls = feature_engine_class(OPERATION_ENGINES[op])
        expected = set(inspect.signature(engine_cls.__init__).parameters) - {"self"}

        df, _ = self._frame(**{k: self.DEFAULTS[k] for k in self.DEFAULTS})
        assert set(build_hw_kwargs(df, op)) == expected

    def test_every_constant_is_read_from_the_data_not_defaulted(self):
        from train import build_hw_kwargs

        off_default = {k: v * 2 for k, v in self.DEFAULTS.items()}
        df, _ = self._frame(**off_default)
        got = build_hw_kwargs(df, "gemm_universal")
        assert got == off_default

    def test_the_value_is_an_int(self):
        from train import build_hw_kwargs

        df, _ = self._frame(**self.HW)
        value = build_hw_kwargs(df, "gemm_universal")["lds_capacity"]
        assert type(value) is int

    def test_an_all_nan_column_falls_back_to_the_default(self):
        from train import build_hw_kwargs

        df, _ = self._frame(**self.HW)
        df["hw_lds_capacity"] = np.nan
        assert "lds_capacity" not in build_hw_kwargs(df, "gemm_universal")

    #: constant -> features computed from it besides its hw_ passthrough.
    DERIVED_USES = {
        "lds_capacity": ["lds_usage_ratio"],
        "num_cus": ["cu_utilization", "hw_total_simds"],
        "simds_per_cu": ["hw_total_simds"],
    }

    #: Keys extract() reads from the problem dict.
    _PROBLEM_KEYS = ("m", "n", "k", "layout", "dtype", "split_k")

    def _features(self, engine, df, path):
        if path == "extract_batch":
            return engine.extract_batch(df)
        row = df.iloc[0].to_dict()
        problem = {k: row[k] for k in self._PROBLEM_KEYS}
        kernel = {k: v for k, v in row.items() if k not in problem}
        return engine.extract(problem, kernel)

    def test_the_two_paths_agree(self):
        from feature_engine import GemmUniversalFeatureEngine

        df, _ = self._frame(**self.HW)
        df.loc[0, "split_k"] = 4
        engine = GemmUniversalFeatureEngine()
        np.testing.assert_allclose(
            self._features(engine, df, "extract"),
            self._features(engine, df, "extract_batch")[0],
        )

    @pytest.mark.parametrize("path", ["extract", "extract_batch"])
    @pytest.mark.parametrize("constant", sorted(DEFAULTS))
    def test_each_constant_reaches_its_passthrough_feature(self, constant, path):
        from feature_engine import GemmUniversalFeatureEngine

        df, _ = self._frame(**self.HW)
        names = GemmUniversalFeatureEngine().get_feature_names()
        col = names.index(f"hw_{constant}")
        base = self._features(GemmUniversalFeatureEngine(), df, path)
        moved = self._features(
            GemmUniversalFeatureEngine(**{constant: self.DEFAULTS[constant] * 2}),
            df,
            path,
        )
        assert base[..., col] != pytest.approx(moved[..., col])

    @pytest.mark.parametrize("path", ["extract", "extract_batch"])
    @pytest.mark.parametrize("constant", sorted(DEFAULTS))
    def test_each_derived_feature_still_uses_its_constant(self, constant, path):
        from feature_engine import GemmUniversalFeatureEngine

        df, _ = self._frame(**self.HW)
        names = GemmUniversalFeatureEngine().get_feature_names()
        base = self._features(GemmUniversalFeatureEngine(), df, path)
        moved = self._features(
            GemmUniversalFeatureEngine(**{constant: self.DEFAULTS[constant] * 2}),
            df,
            path,
        )
        changed = {
            names[j]
            for j in range(len(names))
            if base[..., j] != pytest.approx(moved[..., j])
        }
        expected = {f"hw_{constant}"} | set(self.DERIVED_USES.get(constant, []))
        assert changed == expected, (
            f"the features computed from {constant} on the {path} path have "
            f"changed; DERIVED_USES says {sorted(expected)}, measured "
            f"{sorted(changed)}"
        )

    def test_the_constructor_defaults_have_not_moved(self):
        from feature_engine import GemmUniversalFeatureEngine

        assert GemmUniversalFeatureEngine().hardware_config == self.DEFAULTS

    def test_a_recorded_spec_rebuilds_the_same_engine(self):
        from predict import _engine_for_spec
        from train import build_feature_spec, build_hw_kwargs

        df, Engine = self._frame(**self.HW)
        fe = Engine(**build_hw_kwargs(df, "gemm_universal"))
        spec = build_feature_spec(
            "gemm_universal", fe, "bf16", "gfx950", ["tflops"], [], {}
        )
        np.testing.assert_allclose(
            fe.extract_batch(df), _engine_for_spec(spec).extract_batch(df)
        )

    def test_the_constants_actually_move_the_features(self):
        df, Engine = self._frame(**self.HW)
        a = Engine(**self.HW).extract_batch(df)
        b = Engine().extract_batch(df)
        assert not np.allclose(a, b), (
            "the hardware constants do not change any feature, so the "
            "round-trip test proves nothing"
        )

    def test_a_spec_without_the_key_rebuilds_the_defaults(self):
        from predict import _engine_for_spec

        df, Engine = self._frame(**self.HW)
        legacy = {
            "op_type": "gemm_universal",
            "feature_engine": "GemmUniversalFeatureEngine",
        }
        np.testing.assert_allclose(
            _engine_for_spec(legacy).extract_batch(df), Engine().extract_batch(df)
        )

    @pytest.mark.parametrize(
        "engine_name",
        [
            "GemmUniversalFeatureEngine",
            "GemmUniversalVecFeatureEngine",
            "GroupedConvFeatureEngine",
        ],
    )
    def test_derived_values_are_not_fed_back(self, engine_name):
        from feature_engine import feature_engine_class

        cls = feature_engine_class(engine_name)
        fe = cls()
        cls(**fe.hardware_config)
        assert "total_simds" not in fe.hardware_config

    def test_an_engine_hiding_constants_behind_kwargs_raises(self):
        from feature_engine import GemmUniversalFeatureEngine

        class Hidden(GemmUniversalFeatureEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        with pytest.raises(TypeError, match="cannot be resolved from the signature"):
            Hidden().hardware_config

    def test_an_unregistered_stored_constant_raises(self):
        from feature_engine import GemmUniversalFeatureEngine

        fe = GemmUniversalFeatureEngine()
        fe._hw["invented_constant"] = 7
        with pytest.raises(TypeError, match="does not accept"):
            fe.hardware_config

    def test_every_shipped_model_can_still_be_scored(self):
        import json

        import numpy as np

        from predict import Predictor, _engine_for_spec

        models = sorted((Path(__file__).resolve().parents[1] / "models").glob("*/"))
        assert models, "no shipped models found"
        checked = 0
        for d in models:
            spec_path = d / "feature_spec.json"
            if not spec_path.exists():
                continue
            checked += 1
            spec = json.loads(spec_path.read_text())
            engine = _engine_for_spec(spec)
            names = engine.get_feature_names()
            missing = set(spec["feature_names"]) - set(names)
            assert not missing, f"{d.name} needs features the engine lacks: {missing}"
            width = (
                Predictor(str(d)).select_features(np.zeros((1, len(names)))).shape[1]
            )
            assert width == len(spec["feature_names"]), d.name
        assert checked == len(models), f"skipped {len(models) - checked} model(s)"
        assert len(models) >= 6

    def test_a_disagreeing_hardware_column_raises(self):
        from train import build_hw_kwargs

        df, _ = self._frame(**self.HW)
        mixed = pd.concat([df, df], ignore_index=True)
        mixed.loc[1, "hw_lds_capacity"] = 65536
        with pytest.raises(ValueError, match="distinct values"):
            build_hw_kwargs(mixed, "gemm_universal")

    def test_a_nan_first_row_still_finds_the_value(self):
        from train import build_hw_kwargs

        df, _ = self._frame(**self.HW)
        mixed = pd.concat([df, df], ignore_index=True)
        mixed.loc[0, "hw_lds_capacity"] = np.nan
        assert build_hw_kwargs(mixed, "gemm_universal")["lds_capacity"] == 163840

    def test_a_non_numeric_constant_names_its_column(self):
        from train import build_hw_kwargs

        df, _ = self._frame(**self.HW)
        df["hw_num_cus"] = df["hw_num_cus"].astype(object)
        df.loc[0, "hw_num_cus"] = "unknown"
        with pytest.raises(ValueError, match="hw_num_cus is 'unknown'"):
            build_hw_kwargs(df, "gemm_universal")

    def test_a_mixed_type_column_reports_rather_than_crashing(self):
        from train import build_hw_kwargs

        df, _ = self._frame(**self.HW)
        mixed = pd.concat([df, df], ignore_index=True)
        mixed["hw_num_cus"] = pd.Series([256, "unknown"], dtype=object)
        with pytest.raises(ValueError, match="distinct values"):
            build_hw_kwargs(mixed, "gemm_universal")

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_constant_raises(self, bad):
        from train import build_hw_kwargs

        df, _ = self._frame(**self.HW)
        df.loc[0, "hw_num_cus"] = bad
        with pytest.raises(ValueError, match="non-positive"):
            build_hw_kwargs(df, "gemm_universal")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
