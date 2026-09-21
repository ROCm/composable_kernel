# Learnings and Design Decisions

Empirical findings from building the CK Tile kernel performance prediction system.
These inform the current defaults and explain why certain approaches were chosen.

## 1. Log-Transform is Essential for Cross-Scale Accuracy

**Problem**: GEMM TFLOPS spans 5 orders of magnitude across different problem
sizes. When training on raw TFLOPS, the regression loss (RMSE) is dominated by
large shapes where absolute errors are biggest. The model learns to predict
large shapes accurately but ignores tiny shapes where the TFLOPS values are
much lower.

**Evidence** (168 shapes, 626K rows, 5-fold GroupKFold CV):


| Model                         | Mean Eff   | P10 Eff    | tiny_m Eff | Min Eff    |
| ----------------------------- | ---------- | ---------- | ---------- | ---------- |
| Raw TFLOPS (500 trees)        | 92.73%     | 80.24%     | 84.55%     | 4.26%      |
| **log1p(TFLOPS)** (500 trees) | **96.92%** | **94.34%** | **94.89%** | **60.27%** |
| log1p(TFLOPS) (2000 trees)    | 97.51%     | 93.89%     | 96.04%     | 63.56%     |


**Solution**: Train on `log1p(measured_tflops)` and apply `expm1()` to
predictions. This is now the default in `train.py`. Pass `--no_log_transform`
to revert to raw regression (not recommended).

**Why log1p, not log**: `log1p(x) = log(1 + x)` handles zero and near-zero
TFLOPS gracefully, whereas `log(x)` produces -inf for x=0.

## 2. Tiny-M Shapes are the Hardest Case

M=1 (single-token inference) shapes are fundamentally different from batch shapes:

- Most kernel configurations produce very low TFLOPS
- The "best" kernel is often only marginally better than the rest
- The oracle performance itself is very low, so any prediction error tanks efficiency
- Many kernels fail outright (tile_m=128 with M=1 wastes 127/128 of the tile)

The bottom shapes in our evaluation are all M=1, with efficiencies in the
63-70% range. These shapes have such low absolute performance that the model's
noise floor exceeds the performance difference between kernels.

**Mitigation**: Log-transform helps significantly (tiny_m improved from 84% to
96%). For production use with M=1, consider a dedicated fallback (e.g.,
hardcoded kernel selection for M < 4 based on known-good configs).

## 3. IHEM (Hard Example Mining) Hurts When Scale is the Issue

We tried Iterative Hard Example Mining with sample reweighting (2x-5x weight
on hard shapes). Result: it made things **worse**, degrading mean efficiency
from 94.31% to 92.90% over 3 iterations.

**Why**: The hard shapes are hard because of scale mismatch, not because the
model lacks capacity. Reweighting amplifies the small-TFLOPS rows, which
distorts the learned relationship between features and performance for the
majority of shapes. The log-transform was the correct fix -- it addresses the
root cause (scale) rather than the symptom (bad predictions on tiny shapes).

**Lesson**: IHEM is useful when the model has capacity gaps (e.g., certain
pipeline types are underrepresented). It is counterproductive when the issue
is target-variable scale. Always try target transforms before reweighting.

## 4. GroupKFold Key = (M, N, K) Forces Generalization

The validation uses `GroupKFold` where the group key is `(M, N, K)` -- all
kernels for the same shape go to the same fold. This means:

- The model is always evaluated on shapes it has **never seen** during training
- Layout is excluded from the key, forcing the model to generalize across layouts
- Since models are per-arch, `arch` is implicit (constant within one training run)

This is much stricter than random row splitting, where the model would see some
kernels for each shape during training. Our efficiency numbers are conservative
estimates of real-world performance on unseen shapes.

## 5. Model Size vs Accuracy Tradeoff


| Config             | Trees    | Leaves  | LR       | Mean Eff   | P10 Eff    | Train Time    |
| ------------------ | -------- | ------- | -------- | ---------- | ---------- | ------------- |
| Small (default v1) | 500      | 127     | 0.05     | 96.92%     | 94.34%     | ~20s          |
| **Big (current)**  | **2000** | **255** | **0.02** | **97.51%** | **93.89%** | **~25s/fold** |


The bigger model improved mean efficiency by 0.6% but P10 didn't improve
(actually slightly worse). The extra capacity helps on medium shapes but
doesn't crack the tiny-M floor. This suggests the feature set, not model
capacity, is the limiting factor for the hardest shapes.

For C++ deployment, the bigger model (2000 trees, 255 leaves) is still fast
enough -- LightGBM inference is O(trees * log(leaves)) per sample, which is
~microseconds even at 2000 trees.

## 6. N=1 and K=1 Shapes are Degenerate

We generated benchmark data for 546 edge-case shapes (N=1, K=1, small N/K).
Result: **zero valid kernel results** across 94 shapes. All 4608 kernels either
fail or produce 0 TFLOPS for these degenerate dimensions.

This means:

- The tile engine kernels have hard minimum dimension requirements
- N=1 / K=1 shapes cannot be handled by the current kernel set
- These shapes need dedicated kernels (e.g., BLAS-1/BLAS-2 fallbacks)
- The ML model should not be expected to handle them -- they should be filtered
out before reaching the heuristic

## 7. Feature Engineering Insights

From LightGBM feature importances on the log-target model:

**Top features** (by split count):

- `M, N, K` -- raw dimensions are always the most important
- `tile_m, tile_n, tile_k` -- the tile shape is the primary kernel differentiator
- `overall_tile_efficiency` -- how well the shape fits the tile (the interaction)
- `num_tiles_m, total_output_tiles` -- work decomposition
- `arithmetic_intensity` -- compute vs memory bound regime
- `pipeline` -- pipeline type (compv3 vs compv4 vs mem) significantly affects perf

**Low-importance features**:

- Hardware constants (CUs, clock, caches) -- they're constant within one arch
model, so they provide no discriminative signal. They'll become important when
training cross-arch models.
- `split_k` -- always 1 in current data
- `persistent` -- rarely True in current kernel set

## 8. Warm-Start Works for Incremental Updates

LightGBM's `init_model` parameter successfully continues training from an
existing model. New trees are added on top of existing ones. Key considerations:

- Feature schema must match exactly (enforced by `check_feature_compatibility`)
- Use fewer new trees (200-500) since we're refining, not starting fresh
- The `train_manifest.json` tracks the full lineage (total trees, data sizes)
- Quality should be at least as good as the base model (tested)

## 9. Data Volume Matters More Than Model Complexity


| Dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Shapes | Rows | Mean Eff (log, 500 trees)     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ---- | ----------------------------- |
| Original (DeepSeek only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 108    | 418K | 98.28% (on seen distribution) |
| + Wide coverage M=1 distribution. Adding 60 diverse shapes (many M=1) exposed the model's weakness on tiny shapes. More diverse training data is always better than a bigger model on narrow data.Summary of DefaultsBased on these findings, the current defaults in `train.py` are:- **Target transform**: `log1p` for TFLOPS and bandwidth (scale normalization)- **Model**: 2000 trees, 255 leaves, max depth 15, LR 0.02- **Validation**: 5-fold GroupKFold, key = (M, N, K)- **Early stopping**: patience 100 (let trees fully converge)- **Warm start**: 500 new trees (was 200, increased for bigger base model) | 168    | 626K | 96.92% (harder distribution)  |


The original 108-shape model looked great (98.28%) but was overfitting to the
DeepSeek LLM inference

