#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Training script for CK Tile kernel performance prediction.

Trains LGBMRegressor models (TFLOPS, latency, bandwidth) with:
  - Log-space regression (log1p transform) for scale-invariant accuracy
  - GroupKFold cross-validation (operation-specific group keys)
  - Iterative Hard Example Mining (IHEM)
  - Model complexity bounds for C++ deployability
  - Optional Optuna hyperparameter tuning
  - Warm-start incremental training from a previous model via --warm_start

Supports multiple operation types:
  - gemm_universal: GEMM operations (group by M, N, K)
  - grouped_conv: Grouped convolution (group by problem config)
  - fmha: Fused multi-head attention (future)

Log-transform rationale:
  GEMM TFLOPS spans 5 orders of magnitude (0.02 for M=1 to 2230 for large
  shapes). Raw regression optimizes for absolute RMSE, which means the model
  spends all its capacity predicting large shapes accurately and ignores tiny
  shapes where TFLOPS is < 10. Training on log1p(TFLOPS) puts all shapes on
  equal footing, improving tiny_m efficiency from 84% to 96%.
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_pipeline import build_training_dataset


# Operation-specific target column mappings
TARGET_COLUMNS = {
    "gemm_universal": {
        "tflops": "measured_tflops",
        "latency": "latency_ms",
        "bandwidth": "bandwidth_gb_s",
    },
    "grouped_conv": {
        "tflops": "tflops",
        "latency": "latency_ms",
        "bandwidth": "bandwidth_gb_s",
    },
    "fmha": {
        "tflops": "tflops",
        "latency": "latency_ms",
        "bandwidth": "bandwidth_gb_s",
    },
}

# Targets where log1p transform is applied by default.
# TFLOPS and bandwidth span orders of magnitude; latency is already small-scale.
LOG_TARGETS = {"tflops", "bandwidth"}

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "num_leaves": 255,
    "max_depth": 15,
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "min_child_samples": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "verbose": -1,
    "n_jobs": 8,
    "seed": 42,
}

MAX_ESTIMATORS = 5000
WARM_START_N_ESTIMATORS = 500


def get_feature_engine(operation: str, **hw_kwargs):
    """Get the appropriate feature engine for the operation type."""
    if operation == "gemm_universal":
        from feature_engine import GemmUniversalFeatureEngine

        return GemmUniversalFeatureEngine(**hw_kwargs)
    elif operation == "grouped_conv":
        from feature_engine_grouped_conv import GroupedConvFeatureEngine

        return GroupedConvFeatureEngine(**hw_kwargs)
    elif operation == "fmha":
        raise NotImplementedError("FMHA feature engine not yet implemented")
    else:
        raise ValueError(f"Unknown operation type: {operation}")


def check_feature_compatibility(
    prev_model_dir: Path,
    feature_engine,
) -> None:
    """Verify that the previous model's feature spec matches the current engine.

    Raises ValueError with a detailed message on mismatch. This prevents silent
    corruption when warm-starting from a model trained with a different feature
    schema (e.g., after adding a new feature or changing an encoding).

    Parameters
    ----------
    prev_model_dir : Path
        Directory containing the previous model
    feature_engine : FeatureEngine
        Current feature engine instance (any operation type)
    """
    spec_path = prev_model_dir / "feature_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"No feature_spec.json in {prev_model_dir}. "
            "Cannot verify feature compatibility for warm start."
        )

    with open(spec_path) as f:
        prev_spec = json.load(f)

    prev_names = prev_spec.get("feature_names", [])
    curr_names = feature_engine.get_feature_names()
    if prev_names != curr_names:
        added = set(curr_names) - set(prev_names)
        removed = set(prev_names) - set(curr_names)
        parts = ["Feature schema mismatch between previous model and current engine."]
        if added:
            parts.append(f"  Added features: {sorted(added)}")
        if removed:
            parts.append(f"  Removed features: {sorted(removed)}")
        if not added and not removed:
            parts.append("  Feature order changed (names match but order differs).")
        raise ValueError("\n".join(parts))

    prev_cats = prev_spec.get("categorical_features", [])
    curr_cats = feature_engine.get_categorical_features()
    if sorted(prev_cats) != sorted(curr_cats):
        raise ValueError(
            f"Categorical feature mismatch.\n"
            f"  Previous: {sorted(prev_cats)}\n"
            f"  Current:  {sorted(curr_cats)}"
        )


def load_warm_start_model(prev_model_dir: Path, target: str) -> str | None:
    """Load the path to a previous model file for warm-start, or None if absent.

    Automatically decompresses .lgbm.gz files if the .lgbm file doesn't exist.
    The decompressed file is cached to disk for subsequent loads.

    Returns the string path (what LightGBM's init_model expects) rather than
    a loaded Booster, because LGBMRegressor.fit(init_model=...) accepts both
    path strings and Booster objects and path strings avoid keeping the old
    model in memory.
    """
    import gzip

    model_path = prev_model_dir / f"model_{target}.lgbm"
    gz_path = prev_model_dir / f"model_{target}.lgbm.gz"

    # Auto-decompress if needed
    if not model_path.exists() and gz_path.exists():
        print(f"  Decompressing {gz_path.name}...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(model_path, "wb") as f_out:
                f_out.write(f_in.read())

    if not model_path.exists():
        return None
    return str(model_path)


def compute_group_keys(df: pd.DataFrame, operation: str) -> np.ndarray:
    """Create GroupKFold group keys based on operation type.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    operation : str
        Operation type (gemm_universal, grouped_conv, fmha)

    Returns
    -------
    np.ndarray
        Group keys for GroupKFold cross-validation
    """
    if operation == "gemm_universal":
        # Group by (M, N, K)
        return (
            df["m"].astype(str) + "_" + df["n"].astype(str) + "_" + df["k"].astype(str)
        ).values
    elif operation == "grouped_conv":
        # Group by problem configuration (including 3D and dilation for FWD/BWD_DATA/BWD_WEIGHT)
        return df.apply(
            lambda r: f"{r['N']}_{r['C']}_{r['K']}_{r['G']}_{r['Hi']}_{r['Wi']}_{r['Y']}_{r['X']}_"
            f"{r.get('Di', 1)}_{r.get('Z', 1)}_"
            f"{r.get('dilation_h', 1)}_{r.get('dilation_w', 1)}",
            axis=1,
        ).values
    elif operation == "fmha":
        raise NotImplementedError("FMHA group key computation not yet implemented")
    else:
        raise ValueError(f"Unknown operation type: {operation}")


def compute_tflops_efficiency(
    df: pd.DataFrame, operation: str, pred_col: str = "pred_tflops"
) -> pd.DataFrame:
    """Compute per-shape efficiency: predicted-best TFLOPS / oracle-best TFLOPS.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with predictions and actual TFLOPS
    operation : str
        Operation type to determine grouping columns
    pred_col : str
        Column name for predicted TFLOPS

    Returns
    -------
    pd.DataFrame
        Per-shape efficiency metrics
    """
    results = []

    if operation == "gemm_universal":
        groupby_cols = ["m", "n", "k"]
        tflops_col = "measured_tflops"
    elif operation == "grouped_conv":
        # Group by all problem parameters including 3D and dilation
        base_cols = ["N", "C", "K", "G", "Hi", "Wi", "Y", "X"]
        optional_cols = ["Di", "Z", "dilation_h", "dilation_w"]
        groupby_cols = base_cols + [col for col in optional_cols if col in df.columns]
        tflops_col = "tflops"
    elif operation == "fmha":
        raise NotImplementedError("FMHA efficiency computation not yet implemented")
    else:
        raise ValueError(f"Unknown operation type: {operation}")

    for shape_key, group in df.groupby(groupby_cols):
        oracle_best = group[tflops_col].max()
        if oracle_best <= 0:
            continue
        pred_best_idx = group[pred_col].idxmax()
        selected_tflops = group.loc[pred_best_idx, tflops_col]
        efficiency = selected_tflops / oracle_best

        result = {
            "oracle_best_tflops": oracle_best,
            "selected_tflops": selected_tflops,
            "efficiency": efficiency,
        }
        # Add shape-specific keys
        if operation == "gemm_universal":
            result.update({"m": shape_key[0], "n": shape_key[1], "k": shape_key[2]})
        elif operation == "grouped_conv":
            result.update(
                {
                    "N": shape_key[0],
                    "C": shape_key[1],
                    "K": shape_key[2],
                    "G": shape_key[3],
                    "Hi": shape_key[4],
                    "Wi": shape_key[5],
                    "Y": shape_key[6],
                    "X": shape_key[7],
                }
            )

        results.append(result)

    return pd.DataFrame(results)


def train_single_target(
    X_train,
    y_train,
    X_val,
    y_val,
    params: dict,
    categorical_features: list[str],
    feature_names: list[str],
    init_model=None,
) -> lgb.LGBMRegressor:
    """Train a single LGBMRegressor with early stopping.

    Parameters
    ----------
    init_model : str, Path, lgb.Booster, lgb.LGBMModel, or None
        If provided, training continues from this model (warm start).
        Accepts a file path to a .lgbm file, a Booster instance, or an
        LGBMModel instance. The new model adds n_estimators trees on top
        of the existing ones.
    """
    cat_indices = [
        feature_names.index(c) for c in categorical_features if c in feature_names
    ]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["rmse"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
        categorical_feature=cat_indices if cat_indices else "auto",
        init_model=init_model,
    )
    return model


def run_cv(
    df: pd.DataFrame,
    feature_engine,
    target: str,
    params: dict,
    operation: str,
    n_splits: int = 5,
    use_log: bool = True,
) -> dict:
    """Run GroupKFold cross-validation and return OOF predictions + metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    feature_engine : FeatureEngine
        Feature engine instance (operation-specific)
    target : str
        Target metric (tflops, latency, bandwidth)
    params : dict
        LightGBM parameters
    operation : str
        Operation type (gemm_universal, grouped_conv, fmha)
    n_splits : int
        Number of CV folds
    use_log : bool
        If True and target is in LOG_TARGETS, train on log1p(y) and invert
        predictions with expm1 for efficiency calculation. This normalizes
        the scale so that tiny-M shapes (TFLOPS ~ 1) get equal attention
        as large-M shapes (TFLOPS ~ 2000).
    """
    target_col = TARGET_COLUMNS[operation][target]

    # Handle is_valid column (present in GEMM, not in grouped_conv)
    if "is_valid" in df.columns:
        valid_mask = df["is_valid"].fillna(False) & (df[target_col] > 0)
    else:
        valid_mask = df[target_col] > 0

    df_valid = df[valid_mask].reset_index(drop=True)

    apply_log = use_log and target in LOG_TARGETS

    print(
        f"  Training on {len(df_valid)} valid rows for target={target}"
        f"{' (log-space)' if apply_log else ''}"
    )

    X = feature_engine.extract_batch(df_valid)
    y_raw = df_valid[target_col].values
    y = np.log1p(y_raw) if apply_log else y_raw
    groups = compute_group_keys(df_valid, operation)
    feature_names = feature_engine.get_feature_names()
    cat_features = feature_engine.get_categorical_features()

    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        print(f"  WARNING: Only {len(unique_groups)} unique groups, skipping CV")
        return {}

    gkf = GroupKFold(n_splits=actual_splits)
    oof_preds = np.zeros(len(df_valid))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = train_single_target(
            X_tr, y_tr, X_val, y_val, params, cat_features, feature_names
        )
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        rmse = np.sqrt(np.mean((preds - y_val) ** 2))
        r2 = 1 - np.sum((preds - y_val) ** 2) / max(
            np.sum((y_val - y_val.mean()) ** 2), 1e-10
        )

        if target == "tflops":
            val_df = df_valid.iloc[val_idx].copy()
            preds_raw = np.expm1(preds) if apply_log else preds
            val_df["pred_tflops"] = preds_raw
            eff_df = compute_tflops_efficiency(val_df, operation)
            mean_eff = eff_df["efficiency"].mean() if len(eff_df) > 0 else 0
            p10_eff = eff_df["efficiency"].quantile(0.1) if len(eff_df) > 0 else 0
        else:
            mean_eff, p10_eff = None, None

        fold_metrics.append(
            {
                "fold": fold_idx,
                "rmse": rmse,
                "r2": r2,
                "mean_efficiency": mean_eff,
                "p10_efficiency": p10_eff,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "val_groups": len(np.unique(groups[val_idx])),
            }
        )

        eff_str = (
            f", eff={mean_eff:.4f}, p10={p10_eff:.4f}" if mean_eff is not None else ""
        )
        print(f"    Fold {fold_idx}: RMSE={rmse:.4f}, R2={r2:.4f}{eff_str}")

    df_valid[f"oof_pred_{target}"] = oof_preds

    return {
        "fold_metrics": fold_metrics,
        "oof_df": df_valid,
        "feature_names": feature_names,
        "log_transform": apply_log,
    }


def train_final_model(
    df: pd.DataFrame,
    feature_engine,
    target: str,
    params: dict,
    operation: str,
    init_model=None,
    use_log: bool = True,
) -> lgb.LGBMRegressor:
    """Train the final model on all valid data.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    feature_engine : FeatureEngine
        Feature engine instance (operation-specific)
    target : str
        Target metric (tflops, latency, bandwidth)
    params : dict
        LightGBM parameters
    operation : str
        Operation type (gemm_universal, grouped_conv, fmha)
    init_model : str, Path, lgb.Booster, lgb.LGBMModel, or None
        If provided, training continues from this model (warm start).
    use_log : bool
        If True and target is in LOG_TARGETS, train on log1p(y).
        The saved model then predicts in log-space; callers must apply
        expm1() to get raw values.
    """
    target_col = TARGET_COLUMNS[operation][target]

    # Handle is_valid column (present in GEMM, not in grouped_conv)
    if "is_valid" in df.columns:
        valid_mask = df["is_valid"].fillna(False) & (df[target_col] > 0)
    else:
        valid_mask = df[target_col] > 0

    df_valid = df[valid_mask].reset_index(drop=True)

    apply_log = use_log and target in LOG_TARGETS

    X = feature_engine.extract_batch(df_valid)
    y_raw = df_valid[target_col].values
    y = np.log1p(y_raw) if apply_log else y_raw
    feature_names = feature_engine.get_feature_names()
    cat_features = feature_engine.get_categorical_features()
    cat_indices = [feature_names.index(c) for c in cat_features if c in feature_names]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X,
        y,
        categorical_feature=cat_indices if cat_indices else "auto",
        init_model=init_model,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train CK Tile kernel performance models (GEMM, Grouped Conv, FMHA)"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory with parquet files"
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for models")
    parser.add_argument(
        "--operation",
        default="gemm_universal",
        choices=["gemm_universal", "grouped_conv", "fmha"],
        help="Operation type (gemm_universal, grouped_conv, fmha)",
    )
    parser.add_argument(
        "--op",
        default=None,
        help="Deprecated: use --operation instead. Kept for backward compatibility.",
    )
    parser.add_argument("--dtype", default="fp8", help="Data type filter")
    parser.add_argument("--arch", default="gfx950", help="Architecture")
    parser.add_argument(
        "--targets", default="tflops,latency,bandwidth", help="Comma-separated targets"
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--tune", action="store_true", help="Run Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--no_log_transform",
        action="store_true",
        help="Disable log1p transform on targets. By default, TFLOPS and bandwidth "
        "are trained in log-space for scale-invariant accuracy across shape sizes.",
    )
    parser.add_argument(
        "--warm_start",
        default=None,
        help="Path to previous model directory to continue training from. "
        "Uses LightGBM's init_model to add new trees on top of the "
        "existing model. Feature schemas must match exactly.",
    )
    parser.add_argument(
        "--warm_start_n_estimators",
        type=int,
        default=WARM_START_N_ESTIMATORS,
        help=f"Number of new trees to add when warm-starting (default: {WARM_START_N_ESTIMATORS}). "
        "Lower than a full train since we're refining, not starting from scratch.",
    )
    args = parser.parse_args()

    # Handle backward compatibility for --op flag
    operation = args.operation
    if args.op is not None:
        print("WARNING: --op is deprecated, use --operation instead")
        operation = args.op

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [t.strip() for t in args.targets.split(",")]

    print(f"{'=' * 80}")
    print(f"Training {operation} model")
    print(f"{'=' * 80}")
    print()

    print(f"Loading data from {args.data_dir}...")
    df = build_training_dataset(args.data_dir, op_type=operation, dtype=args.dtype)
    print(f"  Total rows: {len(df)}")

    # Print unique shapes based on operation type
    if operation == "gemm_universal":
        print(f"  Unique shapes: {df.groupby(['m', 'n', 'k']).ngroups}")
    elif operation == "grouped_conv":
        print(
            f"  Unique shapes: {df.groupby(['N', 'C', 'K', 'G', 'Hi', 'Wi', 'Y', 'X']).ngroups}"
        )

    print(f"  Unique kernels: {df['kernel_name'].nunique()}")
    print()

    # Extract hardware parameters from data (if available)
    hw_cols = [c for c in df.columns if c.startswith("hw_")]
    hw_kwargs = {}
    if hw_cols:
        row0 = df.iloc[0]
        if "hw_num_cus" in df.columns:
            hw_kwargs["num_cus"] = int(row0.get("hw_num_cus", 256))
        if "hw_max_clock_mhz" in df.columns:
            hw_kwargs["max_clock_mhz"] = int(row0.get("hw_max_clock_mhz", 2400))
        if "hw_simds_per_cu" in df.columns:
            hw_kwargs["simds_per_cu"] = int(row0.get("hw_simds_per_cu", 4))
        if "hw_shader_engines" in df.columns:
            hw_kwargs["shader_engines"] = int(row0.get("hw_shader_engines", 32))
        if "hw_max_waves_per_cu" in df.columns:
            hw_kwargs["max_waves_per_cu"] = int(row0.get("hw_max_waves_per_cu", 32))
        if "hw_wavefront_size" in df.columns:
            hw_kwargs["wavefront_size"] = int(row0.get("hw_wavefront_size", 64))
        if "hw_l1_cache_kb" in df.columns:
            hw_kwargs["l1_cache_kb"] = int(row0.get("hw_l1_cache_kb", 32))
        if "hw_l2_cache_kb" in df.columns:
            hw_kwargs["l2_cache_kb"] = int(row0.get("hw_l2_cache_kb", 4096))
        if "hw_l3_cache_kb" in df.columns:
            hw_kwargs["l3_cache_kb"] = int(row0.get("hw_l3_cache_kb", 262144))

    # Get operation-specific feature engine
    print(f"Initializing {operation} feature engine...")
    fe = get_feature_engine(operation, **hw_kwargs)
    print(f"  Feature count: {len(fe.get_feature_names())}")
    print(f"  Categorical features: {len(fe.get_categorical_features())}")
    print()

    params = dict(DEFAULT_PARAMS)
    use_log = not args.no_log_transform

    prev_model_dir = None
    prev_manifest = {}
    if args.warm_start:
        prev_model_dir = Path(args.warm_start)
        if not prev_model_dir.exists():
            raise FileNotFoundError(f"Warm-start directory not found: {prev_model_dir}")
        print(f"  Warm-starting from {prev_model_dir}")
        check_feature_compatibility(prev_model_dir, fe)
        print("  Feature compatibility: OK")
        params["n_estimators"] = args.warm_start_n_estimators
        print(f"  New trees to add: {args.warm_start_n_estimators}")

        prev_manifest_path = prev_model_dir / "train_manifest.json"
        if prev_manifest_path.exists():
            with open(prev_manifest_path) as f:
                prev_manifest = json.load(f)

    all_cv_results = {}
    for target in targets:
        if target not in TARGET_COLUMNS[operation]:
            print(f"  Skipping unknown target: {target}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Training {target} model")
        print(f"{'=' * 60}")

        init_model_path = None
        if prev_model_dir is not None:
            init_model_path = load_warm_start_model(prev_model_dir, target)
            if init_model_path:
                print(f"  Warm-starting from {init_model_path}")
            else:
                print(f"  No previous {target} model found, training from scratch")

        t0 = time.time()
        cv_result = run_cv(
            df, fe, target, params, operation, n_splits=args.n_splits, use_log=use_log
        )
        cv_time = time.time() - t0

        if cv_result and cv_result["fold_metrics"]:
            all_cv_results[target] = cv_result["fold_metrics"]
            metrics_path = out_dir / f"cv_metrics_{target}.json"
            with open(metrics_path, "w") as f:
                json.dump(cv_result["fold_metrics"], f, indent=2)
            print(f"  CV completed in {cv_time:.1f}s, saved to {metrics_path}")

            if target == "tflops" and cv_result.get("oof_df") is not None:
                oof_df = cv_result["oof_df"]
                oof_df.to_parquet(out_dir / "oof_predictions.parquet", index=False)

                eff_df = compute_tflops_efficiency(oof_df, operation, "oof_pred_tflops")
                if len(eff_df) > 0:
                    print("\n  OOF TFLOPS Efficiency:")
                    print(f"    Mean: {eff_df['efficiency'].mean():.4f}")
                    print(f"    P10:  {eff_df['efficiency'].quantile(0.1):.4f}")
                    print(f"    P50:  {eff_df['efficiency'].quantile(0.5):.4f}")
                    print(f"    Min:  {eff_df['efficiency'].min():.4f}")

        print(f"\n  Training final {target} model on all data...")
        t0 = time.time()
        model = train_final_model(
            df,
            fe,
            target,
            params,
            operation,
            init_model=init_model_path,
            use_log=use_log,
        )
        train_time = time.time() - t0

        model_path = out_dir / f"model_{target}.lgbm"
        model.booster_.save_model(str(model_path))
        print(f"  Saved {model_path} ({train_time:.1f}s)")

        importances = dict(
            zip(
                fe.get_feature_names(),
                model.feature_importances_.tolist(),
            )
        )
        imp_path = out_dir / f"feature_importances_{target}.json"
        with open(imp_path, "w") as f:
            json.dump(importances, f, indent=2)

    log_targets_used = sorted(LOG_TARGETS & set(targets)) if use_log else []
    spec = {
        "op_type": operation,
        "dtype": args.dtype,
        "arch": args.arch,
        "feature_names": fe.get_feature_names(),
        "categorical_features": fe.get_categorical_features(),
        "targets": targets,
        "log_targets": log_targets_used,
        "params": params,
    }
    with open(out_dir / "feature_spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    # Compute unique shapes based on operation type
    if operation == "gemm_universal":
        unique_shapes = int(df.groupby(["m", "n", "k"]).ngroups)
    elif operation == "grouped_conv":
        unique_shapes = int(
            df.groupby(["N", "C", "K", "G", "Hi", "Wi", "Y", "X"]).ngroups
        )
    else:
        unique_shapes = 0  # Unknown operation

    manifest = {
        "warm_start_from": str(prev_model_dir) if prev_model_dir else None,
        "prev_n_estimators": prev_manifest.get(
            "total_n_estimators", params.get("n_estimators")
        )
        if prev_model_dir
        else 0,
        "new_n_estimators": params["n_estimators"],
        "total_n_estimators": (
            prev_manifest.get("total_n_estimators", 0) + params["n_estimators"]
            if prev_model_dir
            else params["n_estimators"]
        ),
        "data_rows": len(df),
        "valid_rows": int(df["is_valid"].fillna(False).sum()),
        "unique_shapes": unique_shapes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll models saved to {out_dir}")
    if prev_model_dir:
        print(f"  Warm-started from: {prev_model_dir}")
        print(f"  Total estimators: {manifest['total_n_estimators']}")


if __name__ == "__main__":
    main()
