# ML-Based Auto-Tuner Guide

## Overview

The ML-based auto-tuner uses **XGBoost** to learn from historical tile_engine benchmark data and predict the best kernel configuration for any problem size. 

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Auto-Tuner Pipeline                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────────┐                    ┌──────────────────────┐
│ Data Collection   │                    │ Feature Engineering  │
│                   │                    │                      │
│ • Run benchmarks  │                    │ • 50+ features       │
│ • tile_engine     │                    │ • Problem size       │
│ • Sweep configs   │                    │ • Tile config        │
│ • Collect metrics │                    │ • Arithmetic int.    │
└───────────────────┘                    │ • Cache efficiency   │
        │                                └──────────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                    ┌──────────────────────┐
│ Training Data     │                    │ XGBoost Model        │
│                   │                    │                      │
│ • JSON/CSV        │───────────────────>│ • Train on data      │
│ • Problem sizes   │                    │ • Predict GFLOPS     │
│ • Configurations  │                    │ • Feature importance │
│ • Performance     │                    │ • Model persistence  │
└───────────────────┘                    └──────────────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │ Inference            │
                                         │                      │
                                         │ • Predict perf       │
                                         │ • Recommend config   │
                                         │ • Real-time tuning   │
                                         └──────────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install xgboost pandas numpy scikit-learn
```

### 2. Collect Training Data

```bash
# Collect benchmarks from tile_engine
python collect_training_data.py \
    --tile-engine-path /path/to/tile_engine/build \
    --output-dir ./training_data \
    --problem-sizes ml \
    --num-configs 50 \
    --max-workers 8 \
    --export-csv
```

**Output**: `training_data/training_data.json` and `training_data/training_data.csv`

### 3. Train Model

```bash
# Train XGBoost model
python ml_autotuner.py train \
    --data-dir ./training_data \
    --output ./models/autotuner.pkl \
    --target gflops \
    --test-split 0.2
```

**Output**: Trained model saved to `models/autotuner.pkl`

### 4. Use Model for Prediction

```bash
# Predict performance for a configuration
python ml_autotuner.py predict \
    --model ./models/autotuner.pkl \
    --problem-size 1024 1024 1024 \
    --config kernel_config.json
```

### 5. Get Recommendations

```bash
# Recommend best configuration
python ml_autotuner.py recommend \
    --model ./models/autotuner.pkl \
    --problem-size 2048 2048 2048 \
    --candidates candidate_configs.json
```

---

## Detailed Workflow

### Step 1: Data Collection

The data collection script runs tile_engine benchmarks systematically:

**Problem Size Strategies**:
- `power2`: Powers of 2 (64, 128, 256, ...)
- `ml`: Common ML workload sizes (BERT, GPT, etc.)
- `random`: Random sizes for diversity

**Tile Configuration Sweep**:
- Tile sizes: 64x64 to 256x256
- Warp configs: 2x2, 4x4, etc.
- Warp tile sizes: 16x16, 32x32
- Pipelines: compv3, compv4, mem
- Epilogues: cshuffle, default
- Schedulers: intrawave, interwave

**Example**:
```bash
python collect_training_data.py \
    --tile-engine-path ~/ck/build \
    --output-dir ./data \
    --problem-sizes ml \
    --num-configs 100 \
    --max-workers 16 \
    --warmup 10 \
    --iterations 50 \
    --export-csv
```

**Expected Runtime**: 2-8 hours depending on configurations

**Output Format** (JSON):
```json
{
  "metadata": {
    "num_benchmarks": 5000,
    "timestamp": "2025-10-31 12:00:00"
  },
  "benchmarks": [
    {
      "problem": {"M": 1024, "N": 1024, "K": 1024},
      "config": {
        "tile_m": 128, "tile_n": 128, "tile_k": 32,
        "warp_m": 2, "warp_n": 2, "warp_k": 1,
        "pipeline": "compv4",
        "epilogue": "cshuffle"
      },
      "performance": {
        "execution_time_ms": 0.523,
        "gflops": 4096.5,
        "memory_bandwidth_gb_s": 850.2,
        "occupancy": 0.95
      }
    }
  ]
}
```

---

### Step 2: Feature Engineering

The ML model uses **50+ engineered features**:

**Problem Features** (12):
- M, N, K dimensions
- Problem size (M×N×K)
- Dimension ratios (M/N, N/K, M/K)
- Max/min dimensions
- Arithmetic intensity

**Tile Features** (15):
- Tile dimensions (tile_m, tile_n, tile_k)
- Tile size
- Number of tiles needed
- Tile efficiency (how well tiles fit)
- Warp configuration
- Warp tile configuration

**Performance Features** (10):
- Cache efficiency estimate
- Expected occupancy
- Memory access patterns
- Arithmetic intensity
- Block utilization

**Categorical Features** (13):
- Pipeline (one-hot: compv3, compv4, mem)
- Epilogue (one-hot: cshuffle, default)
- Scheduler (one-hot: intrawave, interwave)
- Datatype (one-hot: fp16, bf16, fp32, int8)
- Persistent kernel flag

**Example Feature Vector**:
```python
{
    'M': 1024.0,
    'N': 1024.0,
    'K': 1024.0,
    'problem_size': 1073741824.0,
    'M_div_N': 1.0,
    'arithmetic_intensity': 341.33,
    'tile_m': 128.0,
    'tile_n': 128.0,
    'tile_k': 32.0,
    'num_tiles_m': 8.0,
    'tile_efficiency_m': 1.0,
    'pipeline_compv4': 1.0,
    'epilogue_cshuffle': 1.0,
    # ... 40 more features
}
```

---

### Step 3: Model Training

**XGBoost Configuration**:
```python
{
    'n_estimators': 100,        # Number of trees
    'max_depth': 6,             # Tree depth
    'learning_rate': 0.1,       # Learning rate
    'subsample': 0.8,           # Sample fraction
    'colsample_bytree': 0.8,    # Feature fraction
    'objective': 'reg:squarederror',
    'random_state': 42
}
```

**Training Process**:
1. Load benchmark data
2. Extract features for each configuration
3. Split into train/test (80/20)
4. Normalize features (z-score)
5. Train XGBoost regressor
6. Evaluate on test set
7. Save model + scaler parameters

**Example Training**:
```bash
python ml_autotuner.py train \
    --data-dir ./training_data \
    --output ./models/autotuner_v1.pkl \
    --target gflops \
    --test-split 0.2
```

**Output**:
```
Training XGBoost model on 4500 samples
Training complete. Test R²: 0.9234, Test MAE: 125.43

Training Metrics:
  train_mse: 15234.23
  test_mse: 18456.78
  train_mae: 98.45
  test_mae: 125.43
  train_r2: 0.9456
  test_r2: 0.9234

Top 10 Important Features:
  1. tile_m: 0.1523
  2. tile_n: 0.1456
  3. problem_size: 0.1234
  4. arithmetic_intensity: 0.0987
  5. tile_k: 0.0876
  6. num_tiles_m: 0.0765
  7. M: 0.0654
  8. pipeline_compv4: 0.0543
  9. warp_m: 0.0432
  10. tile_efficiency_m: 0.0321

Model saved to ./models/autotuner_v1.pkl
```

---

### Step 4: Inference

**Predict Performance**:
```python
from ml_autotuner import XGBoostAutoTuner, KernelPerformanceData

# Load model
tuner = XGBoostAutoTuner()
tuner.load_model(Path("./models/autotuner.pkl"))

# Create configuration
config = KernelPerformanceData(
    M=2048, N=2048, K=2048,
    tile_m=256, tile_n=256, tile_k=32,
    warp_m=4, warp_n=4, warp_k=1,
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    pipeline="compv4",
    epilogue="cshuffle",
    scheduler="intrawave"
)

# Predict
predicted_gflops = tuner.predict(config)
print(f"Predicted: {predicted_gflops:.2f} GFLOPS")
```

**Recommend Best Configuration**:
```python
# Load candidate configurations
candidates = [
    KernelPerformanceData(tile_m=128, tile_n=128, tile_k=32, ...),
    KernelPerformanceData(tile_m=256, tile_n=256, tile_k=32, ...),
    # ... more candidates
]

# Get recommendation
best_config, best_perf = tuner.recommend_best_config(
    problem_size=(2048, 2048, 2048),
    candidate_configs=candidates
)

print(f"Best: {best_config.tile_m}x{best_config.tile_n}x{best_config.tile_k}")
print(f"Predicted: {best_perf:.2f} GFLOPS")
```

---

## Integration with Unified Codegen

### Option 1: Pre-generate Optimal Kernels

```bash
# 1. Train model on tile_engine data
python ml_autotuner.py train --data-dir ./data --output ./models/tuner.pkl

# 2. Use model to select best configs for common sizes
python -c "
from ml_autotuner import XGBoostAutoTuner
from preselected_kernels import get_preselected_set

tuner = XGBoostAutoTuner()
tuner.load_model('models/tuner.pkl')

# Get candidates
candidates = get_preselected_set('fp16_rcr_all')

# Recommend for common sizes
for M, N, K in [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]:
    best, perf = tuner.recommend_best_config((M, N, K), candidates)
    print(f'({M}, {N}, {K}): {best.tile_m}x{best.tile_n}x{best.tile_k} -> {perf:.2f} GFLOPS')
"

# 3. Generate only the recommended kernels
python unified_gemm_codegen.py \
    --output-dir ./generated \
    --config ml_recommended_configs.json
```

### Option 2: Runtime Selection

```python
# In dispatcher runtime
from ml_autotuner import XGBoostAutoTuner

class MLDispatcher:
    def __init__(self, model_path):
        self.tuner = XGBoostAutoTuner()
        self.tuner.load_model(model_path)
        self.available_kernels = load_all_kernels()
    
    def dispatch(self, problem):
        # Use ML model to select best kernel
        best_config, predicted_perf = self.tuner.recommend_best_config(
            problem_size=(problem.M, problem.N, problem.K),
            candidate_configs=self.available_kernels
        )
        
        # Find matching kernel
        kernel = find_kernel_by_config(best_config)
        return kernel
```

---

## Advanced Usage

### Custom Feature Engineering

```python
from ml_autotuner import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    @staticmethod
    def extract_features(data):
        features = FeatureEngineer.extract_features(data)
        
        # Add custom features
        features['custom_metric'] = compute_custom_metric(data)
        features['special_ratio'] = data.M / (data.tile_m * data.warp_m)
        
        return features
```

### Ensemble Models

```python
# Train multiple models
models = []
for seed in range(5):
    tuner = XGBoostAutoTuner()
    tuner.train(data, random_state=seed)
    models.append(tuner)

# Ensemble prediction (average)
predictions = [model.predict(config) for model in models]
final_prediction = np.mean(predictions)
```

### Online Learning

```python
# Collect new data
new_data = collect_recent_benchmarks()

# Retrain model
tuner.train(old_data + new_data)
tuner.save_model("models/autotuner_v2.pkl")
```

---

## Troubleshooting

### Issue: Low R² Score

**Causes**:
- Insufficient training data
- High variance in benchmarks
- Poor feature engineering

**Solutions**:
- Collect more data (aim for >2000 samples)
- Increase warmup/iterations
- Add more features
- Try different XGBoost parameters

### Issue: Poor Generalization

**Causes**:
- Overfitting
- Training data not representative

**Solutions**:
- Increase test split
- Add regularization (max_depth, min_child_weight)
- Collect more diverse problem sizes

### Issue: Slow Prediction

**Causes**:
- Too many trees
- Large feature set

**Solutions**:
- Reduce n_estimators
- Feature selection
- Use GPU XGBoost

---

## Future Enhancements

- [ ] Multi-objective optimization (GFLOPS + memory)
- [ ] Uncertainty quantification
- [ ] Active learning (select most informative benchmarks)
- [ ] Transfer learning across GPUs
- [ ] Neural network models (MLP, Transformer)
- [ ] Reinforcement learning for adaptive tuning

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [AutoTVM Paper](https://arxiv.org/abs/1805.08166)
- [Halide Auto-Scheduler](https://halide-lang.org/papers/autoscheduler2019.html)

---

**The ML auto-tuner provides state-of-the-art kernel selection with minimal overhead!**

*Last Updated: 2025-10-31*
*Version: 1.0.0*

