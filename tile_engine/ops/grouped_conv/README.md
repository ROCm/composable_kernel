# Grouped Convolution ML Heuristics & Benchmarking

Training data collection and validation utilities for ML-based kernel selection in grouped convolution operations.

## Overview

This directory supports the **ML heuristic system** for grouped convolution kernel selection. The system achieves **99.67% efficiency** on unseen production workloads by predicting optimal kernels without exhaustive GPU search.

**Key Results:**
- Forward pass: 99.67% mean efficiency (validated on 10 unseen MIOpen shapes)
- 70% perfect oracle matches (selected exact best kernel)
- <1ms selection latency (30,000-60,000× faster than exhaustive search)

See [dispatcher/heuristics/GROUPED_CONV_ML_SUMMARY.md](../../dispatcher/heuristics/GROUPED_CONV_ML_SUMMARY.md) for full technical details.

---

## Files

### Benchmarking & Data Collection
- **`grouped_conv_full_benchmark.py`** - Systematic sweep for training data (kernels × problems)
- **`run_one_grouped_conv_kernel.py`** - Subprocess worker for isolated GPU execution
- **`test_batch_benchmark.py`** - Quick integration test (2 kernels × small problems)
- **`grouped_conv_instance_builder.py`** - Kernel configuration generator from JSON

### ML Validation
- **`validate_ml_vs_oracle.py`** - Compare ML predictions vs exhaustive GPU search
- **`compare_ml_vs_oracle.py`** - Analysis of ML vs oracle performance

### Configuration
- **`configs/*.json`** - Kernel trait configurations (forward, bwd_data, bwd_weight)
- **`problems/*.py`** - Problem datasets (training, validation, MIOpen production shapes)

---

## ML Heuristic Workflow

### 1. Training Data Collection

Already completed. Training datasets:
- **Forward**: 48,845 samples (1,372 unique shapes) - Tier-1 extended
- **Bwd Data**: 14,562 samples (701 unique shapes)
- **Bwd Weight**: 18,150 samples (921 unique shapes)

If you need to collect new data:

```bash
# Full benchmark sweep (all kernels × all problems)
python grouped_conv_full_benchmark.py \
  --variant forward \
  --category full \
  --workers 256 \
  --output training_data_forward_bf16.csv
```

### 2. Training Models

Models are located in `dispatcher/heuristics/models/`:
- `grouped_conv_forward_bf16_gfx950/` - **Production-ready** (99.67% efficiency)
- `grouped_conv_bwd_data_bf16_gfx950/` - Trained, needs hardware validation
- `grouped_conv_bwd_weight_bf16_gfx950/` - Trained, needs hardware validation

To train new models, see [dispatcher/heuristics/README.md](../../dispatcher/heuristics/README.md).

### 3. Validation

Validate ML model performance on unseen shapes:

```bash
cd ../../dispatcher/heuristics/validation/grouped_conv

# Quick sanity check on training shapes (hardware)
python validate_training_shapes.py --direction forward

# Backward models validation (no GPU)
python validate_backward_models.py
```

See [dispatcher/heuristics/validation/README.md](../../dispatcher/heuristics/validation/README.md) for details.

---

## Problem Datasets

Located in `problems/`:

### Training Sets
- **`forward_training.py`** - 2,630 shapes (300 MIOpen + 2,330 synthetic)
- **`forward_training_miopen.py`** - 300 MIOpen production shapes
- **`bwd_data_synthetic_extended.py`** - Backward data training set
- **`bwd_weight_synthetic_extended.py`** - Backward weight training set

### Validation Sets (Unseen)
- **`bwd_data_test_validation.py`** - 10 unseen backward data shapes
- **`bwd_weight_test_validation.py`** - 10 unseen backward weight shapes

### Dataset Generator
- **`create_miopen_training_set.py`** - Extract shapes from MIOpen ALL_CONFIGS_FULL.txt

---

## Benchmarking Usage

### Quick Test (2 Kernels × Few Problems)

```bash
# Test benchmark pipeline
python test_batch_benchmark.py
```

### Full Sweep (All Kernels × All Problems)

```bash
# Forward: 20 kernels × 200 problems = 4,000 measurements
python grouped_conv_full_benchmark.py \
  --variant forward \
  --category full \
  --workers 256 \
  --output sweep_forward.csv

# Backward data
python grouped_conv_full_benchmark.py \
  --variant bwd_data \
  --category full \
  --workers 256

# Backward weight
python grouped_conv_full_benchmark.py \
  --variant bwd_weight \
  --category full \
  --workers 256
```

**Output**: CSV with columns:
```
kernel,problem_idx,N,C,K,G,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w,latency_ms,tflops,non_zero
```

**Note**: The benchmark always starts fresh and overwrites the output CSV file. If you need to preserve previous results, rename or move the CSV file before running a new benchmark.

---

## Instance Builder

Generate kernel configurations from JSON trait files:

```bash
# List all kernels matching config
python grouped_conv_instance_builder.py configs/forward_bf16.json --arch gfx950 --list

# Count kernels
python grouped_conv_instance_builder.py configs/forward_bf16.json --count-only

# Apply filter
python grouped_conv_instance_builder.py configs/forward_bf16.json \
  --filter "c.tile_n >= 128 and c.pipeline == 'compv5'" --list

# Export to JSON
python grouped_conv_instance_builder.py configs/forward_bf16.json \
  --export-json kernels.json
```

### Config Files

- **`forward_bf16.json`** - Forward BF16 (compv3/v4/v5, 30 kernels)
- **`bwd_data.json`** - Backward data (compv3/mem, 20 kernels)
- **`bwd_weight.json`** - Backward weight (compv3/mem, 20 kernels)

**Trait filtering** (see configs for examples):
```json
{
  "variant": "forward",
  "trait_config": {
    "data_type": {"values": ["bf16"]},
    "pipeline": {"values": ["compv3", "compv4", "compv5"]},
    "ndim_spatial": {"values": [2]}
  }
}
```

---

## Architecture

Based on FMHA tile engine design with subprocess isolation:

```
grouped_conv_full_benchmark.py (orchestrator)
  ├─> grouped_conv_instance_builder.py (generate kernel configs)
  ├─> Build phase: JIT compile all kernels (serial, avoids fork/GPU issues)
  └─> Benchmark phase: subprocess workers (serial GPU access)
      └─> run_one_grouped_conv_kernel.py (subprocess)
          └─> GpuGroupedConvRunner (fresh GPU context per problem)
```

**Key design decisions:**
1. **Subprocess isolation** - Fresh GPU context prevents memory leaks
2. **Batch size 20** - Optimal kernels per subprocess
3. **Path-only build** - Main process never initializes GPU
4. **Serial GPU access** - Accurate timing, no contention
5. **Serial codegen/compile** - Avoids ProcessPoolExecutor + GPU fork() issues

**Note**: The `--workers` flag is accepted for API compatibility but currently ignored.
Codegen and compilation run serially to avoid GPU context issues with process forking.

**Success rate**: 99.5% (3,760/3,780 measurements succeeded)

---

## Example Workflow: New Data Collection

```bash
# 1. Generate problem set
cd problems/
python create_miopen_training_set.py \
  --input /path/to/ALL_CONFIGS_FULL.txt \
  --output forward_training_new.py \
  --count 500

# 2. Collect training data
cd ..
python grouped_conv_full_benchmark.py \
  --variant forward \
  --category full \
  --workers 256 \
  --output new_training_data.csv

# 3. Convert to parquet
cd ../../dispatcher/heuristics
python convert_csv_to_parquet.py \
  --input ../../tile_engine/ops/grouped_conv/new_training_data.csv \
  --output data/grouped_conv_forward_bf16_gfx950/new_data.parquet

# 4. Train model
python train.py \
  --data_dir data/ \
  --out_dir models/grouped_conv_forward_bf16_gfx950_v2 \
  --op grouped_conv \
  --variant forward

# 5. Validate (sanity check on training shapes)
cd validation/grouped_conv
python validate_training_shapes.py --direction forward
```

---

## Performance Results

### Forward Pass (Production-Ready)
- **Mean efficiency**: 99.67% on 10 unseen MIOpen shapes
- **Perfect matches**: 70% (7/10 selected exact oracle best)
- **Min efficiency**: 98.4% (even on edge case: 1×491 spatial)
- **Selection time**: <1ms (vs 30-60s exhaustive search)

### Backward Passes (Prediction-Validated)
- **Bwd Data**: 14,562 samples, prediction quality tested
- **Bwd Weight**: 18,150 samples, prediction quality tested
- **Status**: Models trained, hardware validation pending

See [dispatcher/heuristics/GROUPED_CONV_ML_SUMMARY.md](../../dispatcher/heuristics/GROUPED_CONV_ML_SUMMARY.md) for full metrics.

---

## Hardware Tested

- **GPU**: AMD MI300 (gfx950)
- **Datatypes**: BF16 (primary), FP16, FP32
- **Pipelines**: CompV3, CompV4, CompV5 (forward), CompV3/Mem (backward)
- **Schedulers**: Intrawave, Interwave
- **Tile sizes**: 16×64×64, 32×64×64, 64×64×64, 128×128×64, etc.

---

## Related Documentation

- **ML System Overview**: [dispatcher/heuristics/GROUPED_CONV_ML_SUMMARY.md](../../dispatcher/heuristics/GROUPED_CONV_ML_SUMMARY.md)
- **Training Pipeline**: [dispatcher/heuristics/README.md](../../dispatcher/heuristics/README.md)
- **Validation Framework**: [dispatcher/heuristics/validation/README.md](../../dispatcher/heuristics/validation/README.md)
- **Python Examples**: [dispatcher/examples/grouped_conv/python/README_ML_HEURISTIC.md](../../dispatcher/examples/grouped_conv/python/README_ML_HEURISTIC.md)

---

## Next Steps

**For Forward Pass**: Production-ready, integrate into runtime dispatcher

**For Backward Passes**: Run prediction-quality check
```bash
cd ../../dispatcher/heuristics/validation/grouped_conv
python validate_backward_models.py
```

Target: >85% mean efficiency on unseen shapes before production deployment.
