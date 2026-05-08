# Learnings — Grouped-Conv Heuristic (Forward, 2D + 3D)

Empirical findings from building the grouped-convolution kernel performance
predictor for **gfx950**. Specific to the forward path (NHWGC × GKYXC →
NHWGK); backward variants share the same architecture but have not been
re-trained against the latest feature schema (see §6).

These notes inform the current defaults in `feature_engine_grouped_conv.py`,
`predict.py`, and `train.py`, and explain why certain approaches were chosen.

## 1. Kernel-Name Aliasing Was the Top-1 Accuracy Ceiling

**Problem**: Grouped-conv kernel names look like
`grouped_conv_forward_bf16_2d_64x64x64_compv3_intrawave_dsb_si`, but the
original parser in `convert_csv_to_parquet.py` matched only up to the
pipeline token and discarded the wave-mode / dsb / si suffix. Every
`(tile, pipeline)` bucket aliased to a single feature row, even though the
benchmark contained up to 8 distinct kernels per bucket
(`{intrawave, interwave} × {∅, dsb, si, dsb_si}`). With the 2D vs 3D ndim
split, **up to 16 physical kernels collapsed into one feature signature**.

**Evidence** (forward 2D+3D holdout, ~80 unique physical problems):

| Model                        | Features | Mean Eff   | Top-1      | Top-5      |
| ---------------------------- | -------- | ---------- | ---------- | ---------- |
| Pre-suffix (aliased)         | 91       | 88.0%      | ~5–10%     | ~30%       |
| **Suffix-aware (current)**   | **97**   | **92.5%**  | **27.9%**  | **70.6%**  |

**Solution**: Three new kernel-side numeric flags (mirroring `is_compv*`):
`is_intrawave`, `has_dsb`, `has_si`. Plus three pipeline one-hots that were
missing (`is_basic`, `is_compv6`, `is_mem`). Total feature count went from
**83 → 91 → 97** in two stages (3D + dilation in the 91-step; suffix-aware
flags in the 97-step). The 30 valid `(pipeline, wave_mode, dsb, si)`
combinations live in `dispatcher/codegen/grouped_config_rules.py::PIPELINE_VARIANTS`
as the single source of truth used by both the candidate-pool generator and
the codegen harness.

**Why log-target alone wasn't enough**: log-transform fixes scale, not
discrimination. With aliased kernels the model literally cannot rank the 8
intra/inter × dsb/si variants of one tile against each other, no matter
what loss you train against. Top-1 accuracy was bounded by `1/8 = 12.5%`
even with a perfect regressor on the aliased schema.

## 2. Combined 2D+3D Beats Per-Dim Models

We trained three forward models in sequence:

| Model                                            | Features | Training data        | Status                          |
| ------------------------------------------------ | -------- | -------------------- | ------------------------------- |
| `grouped_conv_forward_bf16_gfx950`               | 83       | 2D only, no suffix   | Legacy. Kept for back-compat.   |
| `grouped_conv_forward_2d3d_bf16_gfx950`          | 91       | 2D + 3D, no suffix   | Pre-suffix baseline.            |
| `grouped_conv_forward_2d3d_suffix_bf16_gfx950`   | 97       | 2D + 3D + suffix     | **Current best.**               |

**Finding**: The combined-2D+3D model does **not** hurt 2D performance — both
share the same feature engine and the model learns to gate 3D features on
`Di > 1`. Don't bother training separate 2D-only and 3D-only models unless
you have a strong reason; the combined model wins on holdout.

**Critical features for 3D**: `dilation_d/h/w` in the 91/97-feature schemas
are essential for 3D shapes. Without them the model cannot distinguish
between shapes that share `(N,C,K,Hi,Wi,Y,X)` but differ in dilation, and
its predictions for dilated 3D problems are meaningless. Always include
dilation columns when re-converting CSVs that contain 3D shapes.

## 3. Model Coexistence via Version-Aware Predictor

After the 83 → 91 → 97 feature progression, **all** older models would have
crashed on load with:

```
LightGBMError: The number of features in data (97) is not the same as
it was in training data (83/91)
```

We need to keep the old `forward`, `bwd_data`, and `bwd_weight` models
loadable because we don't have the benchmark data to re-train backward
variants from scratch.

**Solution**: `predict.py::Predictor.__init__` reads
`feature_spec.json["feature_names"]` and builds an index map into the
engine's emit order, so old models pull only the columns they were trained
on. If the engine matches the spec exactly (e.g. the suffix model with the
current engine, or any GEMM model), the index map is `None` and the predict
path is a no-op fast path. If a model expects features the engine no longer
supplies (renamed or removed), `__init__` raises with a clear error rather
than silently predicting garbage.

**Constraint for future engine changes**: the current engine must remain a
**superset** of every deployed model's feature set, or you must retrain.
Adding new features is safe; renaming or removing one is a breaking change.

## 4. What Did Not Matter as Much as Expected

- **Hyperparameter tuning**. Default LightGBM params got within ~1% of any
  tuned configuration we tried. The suffix-aware feature change was ~10x
  more impactful than any HP move.
- **Number of CV folds**. `n_splits=5` and `n_splits=10` gave
  indistinguishable holdout numbers.
- **`use_log` for tflops target on grouped-conv**. Marginal (~0.5%)
  improvement, in contrast to the dramatic effect on GEMM (see
  `LEARNINGS.md` §1). Grouped-conv TFLOPS span a narrower range, so scale
  normalization helps less. Left on by default for stability of the
  warm-start path.

## 5. What Did Matter

- **De-aliasing kernel names** via the suffix-aware feature/parser change
  (§1) — by far the largest single improvement.
- **Group-aware CV** (`GroupKFold` keyed on the dim tuple). Without it,
  the same physical problem with different kernels ends up in both train
  and val, and the CV metric is wildly optimistic.
- **Including dilation columns** for 3D shapes (§2).
- **Joining ML and oracle results by dimension tuple, not `problem_idx`**.
  Index columns in benchmark CSVs are an artifact of generation order and
  cannot be trusted across files; always re-key on the dim tuple.

## 6. Backward Variants Not Yet Upgraded

`grouped_conv_bwd_data_bf16_gfx950` and `grouped_conv_bwd_weight_bf16_gfx950`
are still 83-feature, pre-suffix models. They load via the version-aware
Predictor but inherit the same aliasing problem the forward model used to
have. To upgrade:

1. Re-benchmark (the existing CSVs do not encode wave_mode / dsb / si in
   the kernel names — verify before you start).
2. Re-run `convert_csv_to_parquet.py` (suffix-aware regex) to get parquets
   with `wave_mode`, `has_dsb`, `has_si` columns.
3. Train with `--op grouped_conv --targets tflops --n_splits 5`.

Expect the same magnitude of top-1 accuracy jump that the forward model saw.

## Summary of Defaults

Based on these findings, the current defaults for grouped-conv are:

- **Feature engine**: `GroupedConvFeatureEngine` emits 97 features (38
  problem + extended kernel block with suffix flags + 18 interaction + 12
  hardware).
- **Pipeline variant set**: `dispatcher/codegen/grouped_config_rules.PIPELINE_VARIANTS`
  is the single source of truth for the 30 valid
  `(pipeline, wave_mode, dsb, si)` combinations used by both codegen and
  the candidate-pool generator.
- **Predictor loading**: version-aware feature filtering in
  `predict.py::Predictor` allows old (83/91-feature) models to coexist with
  the new (97-feature) suffix model under the same engine.
- **CV**: 5-fold GroupKFold with the group key including all spatial dims
  and dilation.
- **Target transform**: log1p on tflops (consistent with GEMM defaults
  even though the marginal gain on grouped-conv is small).
