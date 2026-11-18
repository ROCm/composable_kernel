# Benchmarking GEMM kernels chosen by the ML heuristic

This runbook covers measuring GEMM Universal kernels with the tile_engine
driver (`tile_engine/ops/gemm/gemm_full_benchmark.py`) when a trained model
chooses which kernels to measure, and judging how good its choices are.
[DATA_GENERATION.md](DATA_GENERATION.md) covers the other route, the CMake
benchmark binaries; the two enumerate kernels differently, so keep a dataset
to one route.

## When this applies

A full sweep builds every kernel of a configuration space and times it on every
problem it can run. Its cost grows with the number of kernels compiled and the
number of `(kernel, problem)` pairs timed, and a large space over many problems
can take days. A model trained on earlier measurements ranks each problem's
candidates, so only its top K per problem are built and timed. The selection is
only as good as the model, so a full sweep of the same problems, where one is
affordable, is the reference that measures it.

## The workflow at a glance

```text
config files + problems file + dtype + layout + arch      (identical in every step)
 |
 |-- A. reference sweep ------------------------------------------- sweep.csv
 |      gemm_full_benchmark.py --verify --repeat-top ...          sweep.repeats.csv
 |                                                                 sweep.failures.csv
 |-- B. training data and model
 |      convert_csv_to_parquet.py sweep.csv -> data/*.parquet
 |      train.py --operation gemm_universal_vec ...              -> models/<name>/
 |
 |-- C. candidates and selection
 |      gemm_full_benchmark.py --list-pairs candidates.json      (no build, no GPU)
 |      select_kernels.py --candidates candidates.json ...       -> picks.json, picks.csv
 |
 |-- D. measure the picks
 |      gemm_full_benchmark.py --pairs picks.json --repeat-top K -> measured.csv
 |                                                                 measured.repeats.csv
 '-- E. score the picks against A
```

Steps A and B are needed once per model; C and D run for every new problem set.
E needs A on the same problems.

## Prerequisites

- The dispatcher built for the target arch (see `dispatcher/README.md`), so the
  driver can compile kernels: `libck_tile_dispatcher.a` under
  `dispatcher/build/`.
- The Python packages in `dispatcher/requirements-ml.txt` for steps B, C and E.
- A problems file: `{"problems": [{"M": ..., "N": ..., "K": ...}, ...]}`.
- One or more sweep config files (`tile_engine/ops/gemm/configs/` has
  examples). They define the candidate space; the model can only choose among
  kernels the configs produce.

**Use the same config files, problems file, `--dtype`, `--layout` and `--arch`
in every step.** The driver expands fixed-vector-width kernels for the problems
in the problems file, so a different problems file is a different candidate
set, and a `--pairs` file only names kernels its own expansion produces. The
driver takes one layout per run: repeat every step per layout.

## Step A: the reference sweep

```bash
cd tile_engine/ops/gemm
python3 gemm_full_benchmark.py sweep_config.json \
    --problems problems.json --dtype bf16 --layout rcr --arch gfx950 \
    --verify --repeat-top 10 --repeat-count 5 \
    --failures-csv sweep.failures.csv --build-timeout 1200 --kernel-timeout 180 \
    --workers 64 --csv sweep.csv
```

Why each option:

- `--verify` compares every kernel's output with an fp32 reference. A
  numerically wrong kernel is often fast, so without it a wrong kernel can be
  the "best" kernel, and a training label.
- `--repeat-top 10 --repeat-count 5` re-times the 10 fastest kernels of each
  problem 5 more times, each launch in its own subprocess, into
  `sweep.repeats.csv`. Each sweep measurement is a single timing, and the
  fastest of many single timings is biased towards a lucky one. Take the best
  kernel and its time from the medians of the repeats instead.
- `--failures-csv` records every failed measurement and build with its reason.
  The main CSV only has successful rows; this file tells "could not run" (for
  example the kernel refused the problem) apart from "never tried".
- `--build-timeout` raises the per-step build limit. The limits are wall-clock,
  so a build with many `--workers` on a busy host can hit them, and a kernel
  that times out is missing from the results like any other build failure.
- `--kernel-timeout` bounds each measurement; a batch of `--batch-size` kernels
  is stopped after `kernel-timeout x batch-size` seconds.

The kernels time themselves with `CK_TILE_BENCH_WARMUP` (default 50) warm-up and
`CK_TILE_BENCH_REPEAT` (default 100) timed iterations, flushing the cache
(`CK_TILE_BENCH_FLUSH`, default 1) and rotating `CK_TILE_BENCH_ROTATING`
(default 1000) input buffers. Set them in the environment if needed, and record
them with the results: timings taken with different settings are not
comparable.

**Resuming.** If a sweep stops part-way, list the problems it did not reach as
`"*"` entries in a `--pairs` file (`{"1024x512x256": "*", ...}`) and run again
with the same inputs and a new `--csv`. Each `"*"` problem is measured exactly
as the full run would have measured it.

## Step B: training data and model

```bash
cd dispatcher/heuristics
python3 convert_csv_to_parquet.py --input sweep.csv \
    --output data/gemm_universal_bf16_rcr_gfx950.parquet --arch gfx950
python3 train.py --data_dir data/ --out_dir models/gemm_universal_vec_bf16_gfx950 \
    --operation gemm_universal_vec --dtype bf16 --arch gfx950
```

- The converter takes every kernel field from the kernel name, drops the rows
  that failed `--verify`, and adds the hardware constants of `--arch`;
  `train.py` records them in the model.
- The label is the sweep's single timing per pair. Do not mix in the repeat
  medians: within a problem, every candidate should be measured the same way.
- Keep one arch per data directory: `train.py` does not filter rows by
  `--arch`, and refuses data that carries another arch. Several layouts and
  sweeps of the same arch can share one directory.
- `--operation gemm_universal_vec` adds the fixed-vector-width features; use it
  whenever the space includes fixed-vector-width kernels.

## Step C: candidates and selection

```bash
cd tile_engine/ops/gemm
python3 gemm_full_benchmark.py sweep_config.json \
    --problems problems.json --dtype bf16 --layout rcr --arch gfx950 \
    --list-pairs candidates.json

cd dispatcher/heuristics
python3 select_kernels.py --candidates candidates.json \
    --model_dir models/gemm_universal_vec_bf16_gfx950 --top_k 5 \
    --out_pairs picks.json --out_ranked picks.csv
```

- `--list-pairs` builds nothing and needs no GPU. It writes exactly the pairs a
  full run would measure, that is after the vector-width and tile rules, with
  each kernel's configuration. Every candidate can therefore run its problem;
  without that filter a model trained only on successful launches readily
  picks kernels that cannot run.
- `select_kernels.py` ranks each problem's candidates and keeps the top K. It
  refuses a model trained for another dtype or arch.
- If the picks will be scored (step E), keep `picks.csv` as written, for
  example by recording its checksum, before measuring anything, so the scored
  selection cannot be adjusted after seeing results.
- K trades measurement time against the chance that the best kernel is among
  the picks. Choose it from step E on problems that have a reference.

## Step D: measure the picks

```bash
cd tile_engine/ops/gemm
python3 gemm_full_benchmark.py sweep_config.json \
    --problems problems.json --dtype bf16 --layout rcr --arch gfx950 \
    --pairs picks.json --verify --repeat-top 5 --repeat-count 5 \
    --failures-csv measured.failures.csv --build-timeout 1200 --kernel-timeout 180 \
    --csv measured.csv
```

- Only the union of the picks is built and only the listed pairs are timed. A
  pick that cannot be measured (build failed, or the driver does not pair it)
  is printed and recorded as `not-measured`.
- Use `--repeat-top` at least K so every pick is re-timed: a pick whose single
  timing was unlucky would otherwise be ranked on that timing.
- For each problem, the chosen kernel is the verified pick with the lowest
  median latency in `measured.repeats.csv`.

## Step E: score the picks against the reference

For one problem, the efficiency at K is the best of the model's first K picks
divided by the best kernel of the reference sweep, both as median latencies of
the repeats:

```python
import pandas as pd


def medians(path):
    df = pd.read_csv(path, float_precision="round_trip")
    return df[df.ok == 1].groupby(["M", "N", "K", "kernel"]).latency_ms.median()


best = medians("sweep.repeats.csv").groupby(["M", "N", "K"]).min()
measured = medians("measured.repeats.csv")
picks = pd.read_csv("picks.csv")
picks[["M", "N", "K"]] = picks.problem.str.split("x", expand=True).astype(int)
for k in (1, 3, 5):
    top = picks[picks["rank"] <= k].join(measured, on=["M", "N", "K", "kernel"])
    eff = (best / top.groupby(["M", "N", "K"]).latency_ms.min()).dropna()
    print(f"K={k}: mean {eff.mean():.3f}, 10th percentile {eff.quantile(0.1):.3f}, "
          f"{len(eff)} problems")
```

- 1.000 means the picks contained the reference's best kernel. Values slightly
  above 1 happen when the two runs are separate jobs: compare the medians of a
  few kernels measured in both runs to see how far timings drift between them.
- Run A and D on the same GPU model with the same timing settings.
- Score problems the model was trained on separately from new ones; only new
  problems show how the model generalises.

## Cost

Build time grows with the number of distinct kernels compiled, measurement
time with the number of pairs timed plus `repeat-top x repeat-count` launches
per problem. `--list-pairs` gives both before anything runs: the sweep compiles
`len(kernels)` kernels and times every listed pair, while the picks need only
the distinct kernels in `picks.json` (printed by `select_kernels.py`) and K
pairs per problem.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `--pairs lists N kernels this run does not have` | The configs, problems file, dtype, layout or arch differ from the run that wrote the list. |
| `Listed pair not measured: ... not paired` | The kernel's vector widths or tile do not fit that problem; it cannot run it. |
| `Listed pair not measured: ... not built` | The kernel failed to build; `--failures-csv` has the reason. |
| `Timeout: compile exceeded N s` in the build log | Raise `--build-timeout` or lower `--workers`. |
| `select_kernels.py`: model trained for another dtype or arch | Train on data of the run's dtype and arch, or use a model that was. |
| The model refuses to load (feature encoding version) | The model was trained with another version of the feature engine; retrain it. |
