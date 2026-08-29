# CK Tile Heuristics: ML-Based Kernel Selection

Fast, accurate kernel selection for CK Tile operations using LightGBM regression
with Origami-augmented feature engineering.

## What This Does

Instead of running all 4608+ kernel configurations on the GPU to find the best
one (exhaustive search taking ~46 seconds per shape), this system trains an ML
model that predicts TFLOPS for any (problem, kernel) pair in microseconds. It
scores all candidates instantly and picks the best kernel -- achieving 98.28%
of oracle-best TFLOPS efficiency across 108 tested shapes.

## Quick Start

### 1. Generate and convert benchmark data

**Step 1: Generate benchmark data**

```bash
python3 generate_benchmark_data.py \
    --build_dir /path/to/build \
    --output_dir data/fp16_original \
    --dtype fp16 \
    --layout rcr \
    --num_build_jobs 4 \
    --warmup 10 \
    --repeat 50
```

This outputs JSON with all benchmark results.

**Step 2: Convert JSON to parquet training format**

```bash
python3 convert_json_to_parquet.py \
    --input data/fp16_original/benchmark_results_fp16_rcr.json \
    --output data/fp16_original/fp16_training_data.parquet \
    --arch gfx950
```

The converter automatically fixes pad flags for `_mem` kernels and validates data.

**Alternative: Parse existing logs**

If you have raw benchmark logs from CK Tile:

```bash
python3 data_pipeline.py ck_tile_testrun_2.log \
    -o data/gemm_universal_fp8_rcr_gfx950.parquet \
    --arch gfx950 --capture_hw
```

### 2. Train a model

```bash
python3 train.py \
    --data_dir data/ \
    --out_dir models/gemm_universal_fp8_gfx950 \
    --op gemm_universal --dtype fp8 --arch gfx950
```

**Note**: Trained models are automatically compressed to `.lgbm.gz` format to save space (~67% reduction). The Python tools automatically decompress them on first use and cache the decompressed version. For warm-start training, decompression happens automatically.

### 3. Evaluate

```bash
python3 evaluate.py \
    --model_dir models/gemm_universal_fp8_gfx950 \
    --data_dir data/ --op gemm_universal --dtype fp8
```

### 4. Predict the best kernel for a problem

```bash
python3 predict.py \
    --model_dir models/gemm_universal_fp8_gfx950 \
    --m 128 --n 1536 --k 7168 --layout rcr
```

### 5. Search for optimal configs (optional)

```bash
python3 search.py \
    --model_dir models/gemm_universal_fp8_gfx950 \
    --m 128 --n 1536 --k 7168 \
    --strategy random --budget 500 --top_k 10
```

### 6. Using models in C++ (requires decompression)

C++ code uses the LightGBM C API which requires uncompressed `.lgbm` files. If you have compressed models (`.lgbm.gz`), decompress them first:

```bash
cd models/gemm_universal_fp16_gfx950
gunzip model_tflops.lgbm.gz
```

Then use in C++ examples:

```bash
cd dispatcher/build
./gemm_09_ml_heuristic --model ../heuristics/models/gemm_universal_fp16_gfx950/model_tflops.lgbm
```

**Note**: Python tools automatically decompress `.lgbm.gz` files on first use, so you can run Python scripts first to trigger decompression, then use the same models in C++.

## Architecture

```
Problem (M, N, K, dtype, layout)
    |
    v
FeatureEngine.extract_batch()    <-- 55 features: problem, kernel, interaction, hardware
    |
    v
LGBMRegressor.predict()          <-- predicts TFLOPS for each candidate kernel
    |
    v
Sort by predicted TFLOPS          <-- rank all candidates
    |
    v
Select Top-1 kernel               <-- 98.28% mean efficiency, <1ms inference
```

Three models are trained per (op, dtype, arch):
- **TFLOPS model** (primary): used for kernel ranking
- **Latency model** (auxiliary): for latency-sensitive workloads
- **Bandwidth model** (auxiliary): for memory-bound analysis

## File Inventory

| File | Purpose |
|---|---|
| `generate_benchmark_data.py` | Build and run benchmarks across ~25 diverse problem sizes, output JSON |
| `convert_json_to_parquet.py` | Convert benchmark JSON to parquet training format, fix `_mem` pad flags |
| `data_pipeline.py` | Parse raw benchmark logs into canonical parquet datasets |
| `feature_engine.py` | 55-feature extraction: problem, kernel, interaction, hardware profile |
| `train.py` | Multi-target LGBMRegressor training with GroupKFold CV, IHEM, warm-start |
| `predict.py` | Predictor class: predict TFLOPS/latency/bandwidth, rank kernels |
| `evaluate.py` | Full evaluation: global metrics, per-shape/layout/pipeline slices |
| `search.py` | Surrogate search: discrete DE, random top-K |
| `generate_wide_coverage.py` | Generate benchmark data across 706 diverse shapes |
| `generate_edge_dims.py` | Generate N=1, K=1, and other edge-case shapes |
| `DATA_GENERATION.md` | Detailed guide for building binaries and generating data |
| `plan.md` | Full design plan with architecture, milestones, and rationale |

## Features Used (55 total)

### Problem features (13)
`M, N, K, split_k, log2(M), log2(N), log2(K), log2(MNK),
arithmetic_intensity, aspect_ratio_mn, aspect_ratio_mk, aspect_ratio_nk, layout`

### Kernel features (17)
`tile_m, tile_n, tile_k, warp_m, warp_n, warp_k, warp_tile_m, warp_tile_n,
warp_tile_k, pipeline, scheduler, epilogue, pad_m, pad_n, pad_k, persistent,
num_warps, tile_volume, tile_mn, lds_usage_estimate, lds_usage_ratio`

### Interaction features (9)
`num_tiles_m, num_tiles_n, num_tiles_k, total_output_tiles,
tile_eff_m, tile_eff_n, tile_eff_k, overall_tile_efficiency, cu_utilization`

### Hardware profile features (12)
`hw_num_cus, hw_simds_per_cu, hw_total_simds, hw_shader_engines,
hw_max_clock_mhz, hw_max_waves_per_cu, hw_wavefront_size, hw_lds_capacity,
hw_l1_cache_kb, hw_l2_cache_kb, hw_l3_cache_kb, hw_num_xcd`

## Model Performance

### fp8 RCR, gfx950

| Metric | 108 shapes (original) | 168 shapes (wide coverage) |
|---|---|---|
| Mean TFLOPS Efficiency | 98.28% | 97.51% |
| P10 TFLOPS Efficiency | 94.64% | 93.89% |
| tiny_m (M=1) Efficiency | 95.57% | 96.04% |
| R2 (TFLOPS) | 0.997 | 0.993 |

### fp16 RCR, gfx950

Trained on 25 shapes, 1,024 kernels, 21,920 valid benchmarks.

| Metric | Value |
|---|---|
| Mean TFLOPS Efficiency | 99.36% |
| P10 TFLOPS Efficiency | 98.05% |
| P50 TFLOPS Efficiency | 100.00% |
| Min Efficiency | 95.45% |
| NDCG@1 | 64.00% |
| Top-5 Hit Rate | 88.00% |

**Shape Family Breakdown:**

| Shape Family | Mean Eff | P10 Eff | Shapes |
|---|---|---|---|
| Large M (M≥1024) | 99.54% | 99.07% | 4 |
| Medium M (128≤M<1024) | 99.62% | 98.74% | 7 |
| Small M (8≤M<128) | 98.82% | 96.22% | 8 |
| Tiny M (M<8) | 99.65% | 98.96% | 6 |

**Pipeline Breakdown:**

| Pipeline | Mean Eff | P10 Eff |
|---|---|---|
| compv3 | 99.75% | 99.09% |
| compv4 | 99.40% | 98.54% |
| mem | 99.08% | 96.59% |

Training uses `log1p(TFLOPS)` as the target by default, which normalizes the
scale across shapes spanning 0.02 to 2230 TFLOPS. This was the key finding
that improved tiny-M shapes from 84% to 96% efficiency. See
[LEARNINGS.md](LEARNINGS.md) for details.

## Validation

Training uses `GroupKFold(n_splits=5)` with group key `(M, N, K)` to ensure
the model is evaluated on shapes it has never seen during training. Layout is
excluded from the group key to force cross-layout generalization.

## Incremental Training (Warm Start)

When new benchmark data arrives, update the model without retraining from scratch:

```bash
python3 train.py \
    --data_dir data/ \
    --out_dir models/v2 \
    --warm_start models/gemm_universal_fp8_gfx950 \
    --warm_start_n_estimators 200
```

This adds 200 new trees on top of the existing model. Feature schemas must
match exactly (automatically enforced).

## Extending to New Ops

Adding support for a new operation (e.g., `gemm_streamk`, `grouped_conv`):

1. **Build binaries**: `ninja -C build benchmark_gemm_streamk_fp8_rcr`
2. **Subclass `FeatureEngine`**: add op-specific features (e.g., StreamK split factor)
3. **Generate data**: run benchmarks across diverse shapes
4. **Train**: `python3 train.py --op gemm_streamk --dtype fp8 --data_dir data/ --out_dir models/`

The training, evaluation, prediction, and search infrastructure is fully
op-agnostic -- only the feature engine needs a new subclass.

## Tests

102 tests covering all modules:

```bash
python3 -m pytest tests/ -v
```

Test coverage includes:
- Log parsing with malformed JSON, empty logs, single-kernel shapes
- Feature formula correctness (tile efficiency, LDS usage, arithmetic intensity)
- Corner-case shapes: M=1, N=1, K=1, prime dimensions, 20480x7168x256
- Batch vs single extraction parity
- Parameter space validation and projection
- Predictor: single/batch prediction, ranking, missing models, empty inputs
- Training: group keys, efficiency computation, warm-start, feature compatibility
- Search: random, DE, config validity, determinism

## Documentation

- **[README.md](README.md)**: This file -- quick start, architecture, performance
- **[DATA_GENERATION.md](DATA_GENERATION.md)**: Complete guide for building tile engine
  binaries, running benchmarks, managing datasets, and troubleshooting
- **[LEARNINGS.md](LEARNINGS.md)**: Empirical findings and design decisions (log-transform,
  IHEM results, tiny-M analysis, feature importance, N=1/K=1 edge cases)

## Grouped Convolution ML Heuristics

### Overview

ML-based kernel selection for grouped convolution operations (forward, bwd_data, bwd_weight) on gfx950 with bf16 precision.

### Results

#### Forward Pass Model
- **Training Data**: 48,845 measurements across 1,372 unique problem shapes
- **Validation Set**: 300 unseen problems from model crawler
- **Validation Performance** (vs. oracle):
  - Mean Efficiency: **93.05%**
  - Median Efficiency: **96.8%**
  - P10 Efficiency: **79.9%**
  
#### Backward Data Gradient (bwd_data) Model
- **Training Data**: 18,773 measurements across 891 unique problem shapes
- **Validation Set**: 300 unseen problems from model crawler
- **Validation Performance** (vs. oracle):
  - Mean Efficiency: **93.8%**
  - Median Efficiency: **96.5%**
  - P10 Efficiency: **82.9%**
  - Top-1 Accuracy: **25.2%** (37/147 problems)

#### Backward Weight Gradient (bwd_weight) Model
- **Training Data**: 34,900 measurements across 1,508 unique problem shapes
- **Validation Set**: 300 unseen problems from model crawler
- **Validation Performance** (vs. oracle):
  - Mean Efficiency: **96.1%**
  - Median Efficiency: **99.2%**
  - P10 Efficiency: **89.4%**
  - Top-1 Accuracy: **32.7%** (51/156 problems)

### Training Data Generation

Extended synthetic problem sets for backward passes cover diverse scenarios:
- Small spatial (7×7, 14×14) + various channels (64-1024)
- Medium spatial (28×28, 32×32, 56×56) + various channels (32-512)
- Large spatial (112×112) + small/medium channels (16-256)
- Asymmetric C/K combinations
- Small and large batch sizes (N=1 to 128)
- Grouped convolutions (G=2, 4, 8)
- Depthwise convolutions (G=C=K)
- Stride-2 downsampling

### Model Files

Trained models stored in:
- `models/grouped_conv_forward_bf16_gfx950/`
- `models/grouped_conv_bwd_data_bf16_gfx950/`
- `models/grouped_conv_bwd_weight_bf16_gfx950/`

Each contains:
- `model_tflops.lgbm` - LightGBM model (compressed with gzip)
- `feature_spec.json` - Feature configuration
- `cv_metrics_tflops.json` - Cross-validation metrics
- `feature_importances_tflops.json` - Feature importance rankings

Models are automatically decompressed on first use.

### Usage

```python
import pandas as pd
from predict import Predictor
from feature_engine_grouped_conv import GroupedConvFeatureEngine

# Define problem
problem = {
    'N': 16, 'C': 256, 'K': 128, 'G': 1,
    'Hi': 28, 'Wi': 28, 'Y': 3, 'X': 3,
    'stride_h': 1, 'stride_w': 1,
    'pad_h': 1, 'pad_w': 1,
    'dtype': 'bf16'
}

# Load model with the grouped-conv feature engine
predictor = Predictor(
    "models/grouped_conv_bwd_data_bf16_gfx950",
    feature_engine=GroupedConvFeatureEngine(),
)

# Build the candidate kernel pool from a training/holdout parquet
# (each row carries kernel_name + every kernel-config column the engine needs).
df = pd.read_parquet("data/grouped_conv_bwd_data/bwd_data.parquet")
configs = [df[df["kernel_name"] == kn].iloc[0].to_dict()
           for kn in df["kernel_name"].unique()]

# Rank candidates by predicted TFLOPS
ranked = predictor.rank_kernels(problem, configs)
best_name, best_tflops = ranked[0]
print(f"Best kernel: {best_name}")
print(f"Predicted TFLOPS: {best_tflops:.2f}")
```

### Validation

Run validation against oracle benchmarks:

```bash
cd projects/composablekernel/tile_engine/ops/grouped_conv
python3 validate_ml_vs_oracle.py --variant bwd_data
python3 validate_ml_vs_oracle.py --variant bwd_weight
```

### Solution Architecture (Grouped Conv)

```
Problem Config → Feature Engineering (83 features) → LightGBM Model → Predict TFLOPS → Select Best Kernel
     ↓              - Problem features (38)             ↓                    ↓
(N,C,K,G,H,W,Y,X)   - Kernel features (12)         Trained on          <1ms total
                    - Interactions (21)            48K samples          latency
                    - Hardware (12)                1372 shapes
```

### Feature Engineering (`feature_engine_grouped_conv.py`)

**83 engineered features**:
- **Problem Features (38)**: Raw params (N,C,K,G,Hi,Wi,Y,X,strides,pads), derived (Ho,Wo), log-scale transforms, arithmetic intensity, aspect ratios, channel/group metrics
- **Kernel Features (12)**: Block size, GEMM tiles (M,N), pipeline type, num warps, tile volume, LDS usage
- **Interaction Features (21)**: Tile efficiency (M,N,K), block-tile ratios, CU utilization, problem-tile comparisons, output tile counts
- **Hardware Features (12)**: GFX950 specs - CUs (304), SIMDs, clocks, wavefront size, cache sizes (L1/L2/L3), XCD count

### Latency

- **Selection Time**: <1ms
- **vs Oracle**: 30-60 seconds
- **Speedup**: 30,000-60,000×

### Model Size

- **Compressed**: 2-8 MB (.lgbm.gz)
- **Runtime Memory**: ~50 MB
- **Feature Array**: <6 KB per problem

### Training Pipeline

```bash
# 1. Collect data: Run all kernels on GPU for diverse problem set
python grouped_conv_full_benchmark.py --problem_set forward_training_miopen

# 2. Preprocess: Convert CSV to Parquet
python convert_csv_to_parquet.py --input train.csv --output train.parquet

# 3. Train model: LightGBM with cross-validation
python train.py --operation grouped_conv --direction forward --dtype bf16

# 4. Validate: Sanity-check on training shapes
python validation/grouped_conv/validate_training_shapes.py
```

### Validation Framework

| Test | Purpose | Shapes | Runtime | Target |
|------|---------|--------|---------|--------|
| `validate_training_shapes.py` | Sanity check on training data | 5 | 5-10 min | >95% efficiency |
| `validate_backward_models.py` | Backward pass prediction quality | 7 | <1 min | Reasonable predictions |

### File Structure (Grouped Conv)

```
dispatcher/heuristics/
├── train.py                           # Training script
├── feature_engine_grouped_conv.py     # Feature engineering
├── predict.py                         # Generic Predictor (use with GroupedConvFeatureEngine)
├── models/
│   ├── grouped_conv_forward_bf16_gfx950/
│   │   ├── model_tflops.lgbm.gz       # Compressed model
│   │   ├── feature_spec.json          # Feature definitions
│   │   └── train_manifest.json        # Training metadata
│   ├── grouped_conv_bwd_data_bf16_gfx950/
│   └── grouped_conv_bwd_weight_bf16_gfx950/
└── validation/
    ├── validate_ml_heuristic.py       # GEMM validation
    └── grouped_conv/
        ├── validate_training_shapes.py
        └── validate_backward_models.py

tile_engine/ops/grouped_conv/
├── grouped_conv_full_benchmark.py     # Data collection
├── run_one_grouped_conv_kernel.py     # Single kernel runner
├── compare_ml_vs_oracle.py            # Analysis tool
└── problems/
    ├── forward_training_miopen.py     # Training problem sets
    └── forward_validation_300.py      # Test problem sets
```

### C++/Python Integration

- **C++ API**: `GroupedConvRegistry::get_solution(problem)`
- **Python API**: `registry.run(problem, input, weight)`
- Automatic fallback to exhaustive search if ML unavailable

```python
from ck_tile.dispatcher import GroupedConvRegistry, GroupedConvProblem

# Define problem
problem = GroupedConvProblem(
    N=2, C=128, K=256, G=1,
    Hi=28, Wi=28, Y=3, X=3,
    stride_h=1, stride_w=1, pad_h=1, pad_w=1,
    dtype='bf16', direction='forward'
)

# ML heuristic automatically selects best kernel
registry = GroupedConvRegistry(arch='gfx950')
result = registry.run(problem, input_tensor, weight_tensor)
```

### Key Innovations

1. **Comprehensive Feature Engineering**: 83 features capture problem-kernel-hardware interactions
2. **Tier-1 Extended Training**: 1,372 shapes (vs 185 baseline) for better edge case coverage
3. **Compressed Models**: LGBM.gz reduces size 8-10× without accuracy loss
4. **Operation-Specific Models**: Separate optimizations for forward/backward passes
5. **Validation Framework**: Automated testing on unseen production workloads

## Verifying Training Quality

To quickly verify that a refactored `train.py` produces models with equivalent quality to the production training script:

```bash
cd /workspace/rocm-libraries/projects/composablekernel/dispatcher/heuristics

# Run automated test (uses 3-fold CV for speed)
./test_model_quality.sh
```

This script will:
1. Validate current production model on 300 validation shapes
2. Train a new model using refactored `train.py`
3. Validate the new model on the same 300 shapes
4. Compare predictions between old and new models

**Expected Output:**
```
Step 4: Comparing predictions...
================================================================================
PREDICTION COMPARISON: bwd_data
================================================================================

Kernel Selection Agreement: 215/300 (71.7%)

Metric                    Old Model       New Model       Delta
----------------------------------------------------------------------
Mean Efficiency           0.9380          0.9380          +0.0000
Median Efficiency         0.9650          0.9650          +0.0000
P10 Efficiency            0.8290          0.8290          +0.0000

Per-Problem Changes:
  Improved:  0 (0.0%)
  Same:      300 (100.0%)
  Degraded:  0 (0.0%)

================================================================================
✓ PASS: New model maintains quality!
================================================================================
```

### Model Selection Process

The validation script (`validate_ml_vs_oracle.py`) automatically selects the model based on:

**Variant:** `--variant {forward|bwd_data|bwd_weight}`
**Model Path:** `dispatcher/heuristics/models/grouped_conv_{variant}_bf16_gfx950/`

For example:
- `--variant bwd_data` → uses `models/grouped_conv_bwd_data_bf16_gfx950/model_tflops.lgbm`
- `--variant bwd_weight` → uses `models/grouped_conv_bwd_weight_bf16_gfx950/model_tflops.lgbm`

### Manual Step-by-Step Comparison

If you want to run each step manually:

#### Step 1: Validate Current Model

```bash
cd tile_engine/ops/grouped_conv

python3 validate_ml_vs_oracle.py \
  --operation grouped_conv \
  --variant bwd_data \
  --problem-set bwd_data_model_crawler_validation \
  --oracle-csv bwd_data_model_crawler_oracle.csv \
  --save-predictions /tmp/bwd_data_old_predictions.csv
```

This uses the model at: `dispatcher/heuristics/models/grouped_conv_bwd_data_bf16_gfx950/`

#### Step 2: Train New Model

```bash
cd ../../dispatcher/heuristics

python3 train.py \
  --operation grouped_conv \
  --data_dir data/bwd_data_training \
  --out_dir /tmp/grouped_conv_bwd_data_bf16_gfx950_new \
  --dtype bf16 \
  --arch gfx950 \
  --targets tflops \
  --n_splits 5
```

#### Step 3: Temporarily Swap Models

```bash
# Backup current model
mv models/grouped_conv_bwd_data_bf16_gfx950 /tmp/backup

# Use new model for validation
cp -r /tmp/grouped_conv_bwd_data_bf16_gfx950_new models/grouped_conv_bwd_data_bf16_gfx950
```

#### Step 4: Validate New Model

```bash
cd ../../tile_engine/ops/grouped_conv

python3 validate_ml_vs_oracle.py \
  --operation grouped_conv \
  --variant bwd_data \
  --problem-set bwd_data_model_crawler_validation \
  --oracle-csv bwd_data_model_crawler_oracle.csv \
  --save-predictions /tmp/bwd_data_new_predictions.csv
```

#### Step 5: Restore Original Model

```bash
cd ../../dispatcher/heuristics

rm -rf models/grouped_conv_bwd_data_bf16_gfx950
mv /tmp/backup models/grouped_conv_bwd_data_bf16_gfx950
```

#### Step 6: Compare Predictions

```bash
cd ../../tile_engine/ops/grouped_conv

python3 compare_model_predictions.py \
  --old-predictions /tmp/bwd_data_old_predictions.csv \
  --new-predictions /tmp/bwd_data_new_predictions.csv \
  --variant bwd_data
```

### Acceptance Criteria

A new model passes quality validation if:

1. ✓ Mean efficiency is within 0.5% of baseline
2. ✓ Median efficiency is within 0.5% of baseline
3. ✓ P10 efficiency is within 2% of baseline
4. ✓ No catastrophic regressions (efficiency drops >10% on any problem)

### Troubleshooting

#### Different Predictions on Same Model

**Unlikely** - If the same model file produces different predictions, check:
- Feature engine version (should be 83 features)
- Problem encoding (verify problem_to_dict matches)
- Predictor initialization (check log transform handling)

#### Quality Regression

If new model has lower efficiency:
1. Check CV metrics in training log - should be similar to baseline
2. Verify identical training data (check parquet row counts)
3. Compare feature importance - should be similar patterns
4. Inspect specific regression cases in comparison output

