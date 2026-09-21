# ML Heuristic Validation Tools

This directory contains validation scripts for testing ML-based kernel selection heuristics.

## Directory Structure

```
validation/
├── README.md                          # This file
├── validate_ml_heuristic.py           # GEMM universal validation
└── grouped_conv/                      # Grouped convolution specific
    ├── validate_training_shapes.py    # Training data sanity check
    └── validate_backward_models.py    # Backward pass prediction quality
```

## Scripts Overview

### 1. `validate_ml_heuristic.py` - GEMM Universal Validation

**Purpose**: Validate ML heuristic for GEMM universal operations (not grouped conv).

**Usage**:
```bash
python validate_ml_heuristic.py --dtype fp16 --layout rcr
python validate_ml_heuristic.py --dtype bf16 --model_dir models/gemm_universal_bf16_gfx950
```

**What it does**:
- Loads benchmark data (oracle-best results for each GEMM shape)
- Uses ML model to predict best kernel for each shape
- Compares ML selection with oracle-best to compute efficiency
- Outputs mean/median/P10/P90 efficiency statistics

**When to use**: Testing GEMM universal ML models on new training data or architectures.

---

## Grouped Convolution Validation

### 2. `grouped_conv/validate_training_shapes.py` - Training Data Sanity Check

**Purpose**: Quick sanity check on shapes WITH multiple kernels in training data.

**Usage**:
```bash
cd dispatcher/heuristics/validation/grouped_conv
python validate_training_shapes.py
```

**What it does**:
1. Selects 5 random training shapes with ≥5 kernels each
2. For each shape:
   - Gets oracle-best from training data
   - Uses ML to predict best kernel
   - Builds BOTH kernels (oracle + ML)
   - Runs both on hardware
   - Compares actual TFLOPS

**Output**:
- Per-shape efficiency (ML vs Oracle on hardware)
- Prediction accuracy (ML predicted TFLOPS vs actual)
- Mean efficiency across test shapes

**Runtime**: ~5-10 minutes (builds 10 kernels, runs on hardware)

**When to use**:
- Quick sanity check after model training
- Verify model isn't overfitting to training data
- Debug prediction accuracy issues

---

### 3. `grouped_conv/validate_backward_models.py` - Backward Pass Prediction Quality

**Purpose**: Quick prediction quality check for bwd_data and bwd_weight ML models.

**Usage**:
```bash
cd dispatcher/heuristics/validation/grouped_conv
python validate_backward_models.py
```

**What it does**:
1. Loads bwd_data and bwd_weight ML models
2. Tests on 5-7 hardcoded representative problems
3. For each problem:
   - Predicts TFLOPS for all backward kernels (compv3, mem pipelines)
   - Shows top-3 predicted kernels
   - Reports prediction statistics

**Output**:
- Top-3 predicted kernels for each problem
- Average predicted TFLOPS
- Pipeline preference (compv3 vs mem)
- Prediction confidence (gap between best and 3rd)

**Runtime**: <1 minute (NO hardware - prediction only)

**When to use**:
- Quick check after training backward models
- Verify model predictions are reasonable
- Debug backward pass heuristic issues

**Note**: This does NOT run on hardware - it only checks prediction quality.

---

## Comparison Matrix

| Script | Operation | Hardware? | Shapes Tested | Runtime | Use Case |
|--------|-----------|-----------|---------------|---------|----------|
| `validate_ml_heuristic.py` | GEMM universal | ✗ | All training | <1 min | GEMM model validation |
| `validate_training_shapes.py` | Grouped conv fwd | ✓ | 5 training | 5-10 min | Quick sanity check |
| `validate_backward_models.py` | Grouped conv bwd | ✗ | 5-7 hardcoded | <1 min | Backward prediction quality |

## Typical Workflow

1. **After training forward model**:
   ```bash
   # Quick check
   python grouped_conv/validate_training_shapes.py
   ```

2. **After training backward models**:
   ```bash
   python grouped_conv/validate_backward_models.py
   ```

## Target Metrics

### Forward Pass (Tier-1 Model)
- **Mean efficiency**: >90% (currently 93.05%)
- **P10 efficiency**: >75% (currently 79.21%)
- **Kernel match rate**: >70%

### Backward Pass
- **Mean efficiency**: >85%
- **Prediction accuracy**: >90%

## Dependencies

All scripts require:
- Trained ML models in `../models/`
- Training data in `../data/`
- Python packages: pandas, numpy, LightGBM, matplotlib (for plotting)

Grouped conv hardware validation scripts additionally require:
- GPU hardware (gfx950 default)
- Compiled kernels or JIT compilation support
- `tile_engine/ops/grouped_conv/` utilities
