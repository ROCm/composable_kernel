# FMHA Tile Engine

Benchmarking and kernel enumeration for Fused Multi-Head Attention (FMHA) via the CK dispatcher's pipelined JIT compilation.

Covers all 9 FMHA kernel families: Forward, Split-KV (main + combine), Paged-KV, Append-KV, Batch Prefill, and Backward (dot\_do\_o, dq\_dk\_dv, convert\_dq) -- totaling 33,541 unique kernel specializations on gfx950.

## Directory Layout

```
fmha/
  fmha_instance_builder.py     Kernel enumeration from JSON config + pipeline rules
  fmha_benchmark.py            Single-config JIT compile and GPU benchmark
  fmha_full_benchmark.py       Full sweep: compile all kernels, benchmark across test shapes
  ck_fmha_testing_matrix.yaml  Test shapes (smoke / full / nightly)
  CMakeLists.txt               CMake targets
  README.md                    This file
  configs/                     Sweep definitions (JSON)
    receipt0_fwd.json             Full receipt-0 forward: ~12K kernels
    fwd.json                      Forward variants
    fwd_ci.json                   Minimal CI subset
    bwd.json                      Backward variants
    splitkv.json                  Split-KV
    appendkv.json                 Append-KV
    pagedkv.json                  Paged-KV
    batch_prefill.json            Batch prefill
  filters/                     Sample Python filter scripts
    h128_no_dropout.py            Keep only h128 without dropout
```

## Quick Start

```bash
# Count kernels without compiling
python fmha_instance_builder.py configs/receipt0_fwd.json --count-only

# Minimal CI build + run (~16 kernels, <1 min)
python fmha_benchmark.py configs/fwd_ci.json --workers 128 --verify

# Full forward receipt-0 compile-only (12K kernels, ~10 min with 256 workers)
python fmha_benchmark.py configs/receipt0_fwd.json --workers 256 --compile-only

# Full sweep: compile every fwd kernel, benchmark against all smoke shapes
python fmha_full_benchmark.py --category smoke --variant fwd --workers 256

# Quick end-to-end test (2 kernels, 1 shape)
python fmha_full_benchmark.py --category smoke --variant fwd --max-kernels 2 --workers 4
```

## How It Works

### Kernel Enumeration

```
JSON config (variant + trait_config allow-list)
  --> fmha_instance_builder.py
    --> fmha_pipeline_rules.py (self-contained CK parity logic)
      --> fmha_arch_specs.json (tile tables per arch / dtype / hdim)
        --> list of FmhaKernelConfig (33,541 total on gfx950)
          --> optional --filter / --filter-file
```

The pipeline rules in `dispatcher/codegen/fmha_pipeline_rules.py` reproduce the exact kernel enumeration from CK Tile's `01_fmha/codegen/`, including per-arch tile constraints, pipeline selection, padding variants, and feature products. Parity is verified by `dispatcher/tests/validate_arch_specs_parity.py`.

### Benchmark Tools

**`fmha_benchmark.py`** -- single-config benchmark. Input: one JSON config (kernel definitions). JIT-compiles all matching kernels, runs each on a given problem size, reports per-kernel timing and optional CPU validation. Optionally writes `--csv` output.

**`fmha_full_benchmark.py`** -- full sweep benchmark. Input: `ck_fmha_testing_matrix.yaml` (test shapes) + JSON configs (kernel definitions). Compiles all kernel variants for selected families, then iterates over test shapes, matching each shape to compatible compiled kernels and benchmarking every match. Writes `--csv` and `--json` output.

### JIT Compilation Pipeline

Both tools use the dispatcher's `setup_multiple_fmha_dispatchers()` which implements a 3-stage pipelined build:

1. **Codegen** (parallel) -- generate C++ kernel specializations and ctypes wrappers
2. **Compile** (parallel) -- `hipcc` compile each kernel and ctypes lib
3. **Link + Load** (parallel) -- produce `.so` libraries, load via ctypes

With 256 workers, throughput is roughly 5-10 kernels/sec depending on kernel complexity.

## JSON Config Format

Each config specifies a `variant` and an optional `trait_config` that acts as an allow-list filter:

```json
{
  "variant": "fwd",
  "trait_config": {
    "data_type": {"values": ["fp16", "bf16"]},
    "pipeline": {"values": ["qr_async"]},
    "mode": {"values": ["batch"]},
    "mask": {"values": ["no"]},
    "bias": {"values": ["no"]},
    "lse": {"values": [false]},
    "dropout": {"values": [false]},
    "logits": {"values": [false]},
    "sink": {"values": [false]}
  }
}
```

If a trait key is absent, all values pass. The `receipt0_fwd.json` config only restricts `data_type` to exclude fp32, giving the full ~12K forward kernel set.

## Filtering

### CLI expression

```bash
python fmha_benchmark.py configs/receipt0_fwd.json \
    --filter "c.hdim_q == 128 and c.pipeline == 'qr_async'"

python fmha_full_benchmark.py --variant fwd \
    --filter "c.hdim_q == 128 and c.hdim_v == 128 and c.data_type == 'fp16'"
```

The expression accesses `c` (an `FmhaKernelConfig` dataclass) with fields: `data_type`, `mode`, `hdim_q`, `hdim_v`, `pipeline`, `tile_m0`, `tile_n0`, `tile_k0`, `pad_s`, `pad_sk`, `pad_d`, `pad_dv`, `mask`, `bias`, `lse`, `dropout`, `logits`, `sink`, `skip_min_seqlen_q`, `qscale`, `paged_kv`, `rope`, `deterministic`, `dbias`, `dropout_variant`.

### Python file filter

```bash
python fmha_benchmark.py configs/receipt0_fwd.json --filter-file filters/h128_no_dropout.py
```

The file must define `filter_config(c) -> bool`. Both `--filter` and `--filter-file` combine with AND logic.

## Test Shape Matrix

`ck_fmha_testing_matrix.yaml` defines test problems in three tiers:

| Category | Purpose | Shapes |
|----------|---------|--------|
| `smoke`  | Pre-submit sanity, <5 min | ~365 |
| `full`   | Post-submit validation | smoke + ~1,500 |
| `nightly`| Exhaustive sweep | all |

Shapes cover representative configurations: GQA ratios, asymmetric head dims, non-power-of-2 sequences, FP8 variants, long sequences, and cross-attention patterns.

## Output Format

### CSV

```
problem_name,batch,seqlen_q,seqlen_k,nhead_q,nhead_k,hdim_q,hdim_v,dtype,
kernel,family,mode,pipeline,tile_m0,tile_n0,tile_k0,...,
latency_ms,tflops,bandwidth_gb_s
```

Every column needed to fully reconstruct the kernel identity is included. TFLOPS and latency come directly from CK's internal HIP event timing.

### JSON

```json
{
  "metadata": {
    "arch": "gfx950",
    "category": "smoke",
    "total_kernels": 600,
    "shapes_benchmarked": 42,
    "total_measurements": 12600
  },
  "results": [...]
}
```

## CMake Targets

```bash
make benchmark_fmha          # Forward sweep
make benchmark_fmha_ci       # Quick CI validation
make benchmark_fmha_bwd      # Backward sweep
make benchmark_fmha_all      # All variants
make benchmark_fmha_splitkv  # Split-KV only
```

## Parity Verification

```bash
python dispatcher/tests/validate_arch_specs_parity.py --arch gfx950 --receipt 0
# PASS: 33,541 kernels across all 9 families
```

This confirms the dispatcher's self-contained enumeration exactly matches CK Tile's upstream codegen.

## Example: Single-Shape All-Kernel Benchmark

Run every compiled fwd fp16 h128 kernel against one shape:

```bash
python fmha_full_benchmark.py \
    --category smoke --variant fwd --workers 256 \
    --filter "c.hdim_q == 128 and c.hdim_v == 128 and c.data_type == 'fp16'" \
    --csv results.csv
```
