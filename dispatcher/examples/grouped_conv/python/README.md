# Grouped Convolution — Python Examples

Examples and test harnesses for the grouped convolution dispatcher (forward,
bwd_data, bwd_weight) using the Python JIT codegen + hipcc workflow.

Run scripts from this directory:

```bash
cd dispatcher/examples/grouped_conv/python
python3 -u <script.py>          # use -u for unbuffered logs
```

GPU arch is auto-detected (`detect_gpu_arch()`); pass `--arch gfx950` to override.

## Examples

| Script | Purpose |
|---|---|
| `01_basic_grouped_conv.py` | End-to-end smoke test: build + run forward kernel, verify output. |
| `02_forward.py` | Forward variant (NHWGC / GKYXC), small 2D problem. |
| `03_bwd_data.py` | Backward-data variant. Runner contract: `run(dY, W, prob)`. |
| `04_bwd_weight.py` | Backward-weight variant. Runner contract: `run(X, dY, prob)`. |
| `05_benchmark.py` | Multi-kernel sweep + timing (slow; runs many configs). |
| `06_registry_json.py` | Build a registry from a JSON config file. |
| `09_ml_heuristic.py` | Demo of LightGBM heuristic (requires `lightgbm`); see *ML heuristic* below. |
| `10_test_all_pipelines.py` | For each variant, test all 8 pipelines with `intrawave`. |
| `11_test_schedulers.py` | For each variant, test all 8 pipelines × {intrawave, interwave}. |
| `12_test_config_options.py` | Test the 5 config options (see *Config-options harness* below). |

## Runner argument contract

`runner.run(input_np, weight_np, prob)` — order matters per variant:

| Variant | `input_np` | `weight_np` |
|---|---|---|
| `forward` | `X` (NHWGC) | `W` (GKYXC) |
| `bwd_data` | `dY` | `W` |
| `bwd_weight` | `X` | `dY` |

## Pipelines & schedulers

All 8 pipelines: `basic_v1, mem, compv3, compv4, compv5, compv6, comp_async,
basic_async_v1`.

* `compv4` and `comp_async` require `double_smem_buffer=True` (loud
  `static_assert` otherwise).
* Not every pipeline supports both `intrawave` and `interwave`. `11_test_schedulers.py`
  treats a pipeline as supported if **at least one** scheduler runs successfully.

## Config-options harness (`12_test_config_options.py`)

Verifies the 5 `GroupedConvKernelConfig` options:

1. `double_smem_buffer` — LDS ping-pong (required for compv4 / comp_async).
2. `num_groups_to_merge` — fuse groups into one tile.
3. `split_image` — split spatial dims for large tensors.
4. `explicit_gemm` — explicit GEMM path (experimental).
5. `two_stage` — two-stage bwd_weight with fp32 workspace.

Each test is run in its **own subprocess** (`--single-test '<json>'` mode) so a
single GPU page fault doesn’t take down the whole sweep — failing combinations
are reported as `CRASH` and the run continues.

Test problem sizes are kept small (e.g. 2D: `N=1, G=2, C=K=64, Hi=Wi=8, 3×3`)
to avoid OOM / aperture violations on the test GPU.

## ML heuristic (`09_ml_heuristic.py`)

LightGBM regression model that predicts kernel TFLOPS and selects a kernel for
a given problem. Requires the `lightgbm` Python package.

* Models live in `dispatcher/heuristics/models/grouped_conv_<variant>_bf16_<arch>/`
  (forward, bwd_data, bwd_weight all available).
* Feature engine: `dispatcher/heuristics/feature_engine_grouped_conv.py`.
* Training entry point: `dispatcher/heuristics/train.py`.
* Prediction: `dispatcher/heuristics/predict.py` (use `Predictor` with
  `GroupedConvFeatureEngine`; build the candidate kernel pool from a
  training/holdout parquet via `df["kernel_name"].unique()`).

Typical training flow:

```bash
# 1. Benchmark to CSV (slow)
cd tile_engine/ops/grouped_conv
python3 -u grouped_conv_full_benchmark.py configs/forward_bf16.json \
  --arch gfx950 --problems forward_training \
  --csv benchmark_forward_bf16_gfx950.csv --workers 8

# 2. CSV → Parquet
cd ../../../dispatcher/heuristics
python3 convert_csv_to_parquet.py \
  --input ../../tile_engine/ops/grouped_conv/benchmark_forward_bf16_gfx950.csv \
  --output data/grouped_conv_forward_bf16_gfx950.parquet --arch gfx950

# 3. Train
python3 train.py --data_dir data \
  --out_dir models/grouped_conv_forward_bf16_gfx950 \
  --op grouped_conv --dtype bf16 --arch gfx950 --targets tflops --n_splits 5
```

To add a new pipeline (e.g. `compv6`) update:
`dispatcher/codegen/grouped_config_rules.py` (`VARIANT_PIPELINES`),
`dispatcher/heuristics/feature_engine_grouped_conv.py` (add the `is_<name>`
flag), and the relevant `tile_engine/ops/grouped_conv/configs/*.json`. Then
re-run the benchmark + train flow above.

## Notes

* Use `python3 -u` for any long-running script so logs aren’t buffered.
* Kernels are compiled once and cached under `/tmp/dispatcher/`; subsequent
  runs reuse the cached `.so`.
* This repo has 1 GPU — do not run benchmarks in parallel.