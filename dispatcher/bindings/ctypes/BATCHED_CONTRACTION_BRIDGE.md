<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Batched Contraction: Tile Engine -> Dispatcher Bridge

This document describes the **batched_contraction** variant of the Tile Engine (TE)
-> Dispatcher bridge. It is the batched-tensor-contraction counterpart of the regular
GEMM bridge (#8997) and the batched/grouped/stream-K/multi-D siblings (#9306, #9000,
#9028, #9308).

## What the bridge is

In the bridge model the **Dispatcher is the single source of truth** for codegen,
build, and runtime; **Tile Engine only generates configs and benchmarks** them.

Batched contraction computes a generalized tensor contraction:

```
E[G.., M.., N..] = epilogue( sum_{K..} A[G.., M.., K..] * B[G.., N.., K..], D0.. )
```

with independent multi-dimensional **G/M/N/K** index groups (`NUM_DIM_G/M/N/K`).
Collapsed, it is a batched `E = A * B^T` (B holds K on its trailing axis).

## Why it needs a dedicated ctypes lib

The generated `launch()` takes
`ck_tile::BatchedContractionHostArgs<NumDTensor>`, which carries **variable-length**
dim and stride vectors (`A_dims`/`B_dims`/`E_dims`, `A_strides`/...). The dispatcher
registry only knows the single-pointer `GemmHostArgs` signature and its generic
backend builds a `GemmHostArgs` — it cannot express the contraction args. So this lib
**bypasses the registry** and calls `SelectedKernel::launch(BatchedContractionHostArgs<N>, stream)`
directly, building the HostArgs from plain C arrays.

## Components

| Layer | File | Role |
|---|---|---|
| Codegen | `dispatcher/codegen/unified_batched_contraction_codegen.py` | one `.hpp` per config; mirrors the Old-TE instance (BatchedContractionProblem/Kernel, UniversalGemmPipeline, CShuffle/Default epilogue); `make_batched_contraction_kernel_name` is the single source of the kernel name |
| C API | `dispatcher/bindings/ctypes/batched_contraction_ctypes_lib.cpp` | flat C ABI; device alloc/copy; packed row-major stride derivation; builds `BatchedContractionHostArgs`; direct launch; warmup/repeat timing |
| Python | `dispatcher/python/batched_contraction_utils.py` | `BatchedContractionKernelConfig` (byte-exact `.name`), `BatchedContractionProblem`, `BatchedContractionDispatcherLib`, `GpuBatchedContractionRunner`, `setup_multiple_batched_contraction_dispatchers`, `expand_sweep` |
| Tests | `dispatcher/tests/test_batched_contraction_bridge.py` | CPU-only: name contract, codegen-JSON projection, problem flops, sweep dedup |
| Build | `dispatcher/bindings/ctypes/CMakeLists.txt` | `dispatcher_batched_contraction_lib` target |
| TE driver | `tile_engine/ops/gemm/batched_contraction_full_benchmark.py` + `run_one_batched_contraction_kernel.py` | 3-phase driver + isolated per-GPU worker with fp32 `--verify` |

## C ABI

```c
int dispatcher_init(void);
int dispatcher_get_num_dim_g(void);   // compiled-in NUM_DIM_G (also m/n/k)
int dispatcher_get_num_d_tensors(void);
int dispatcher_run_batched_contraction(
    const void* A, const void* B, void* E,          // host, row-major packed
    const void** d_ptrs, int num_d,                 // D-tensor host ptrs; num_d must == compiled-in NUM_D_TENSORS
                                                    //   (d_ptrs may be NULL only when num_d==0)
    const int64_t* g_dims, const int64_t* m_dims,
    const int64_t* n_dims, const int64_t* k_dims,
    int num_dim_g, int num_dim_m, int num_dim_n, int num_dim_k,  // must == compiled-in
    int k_batch,
    float* time_ms);                                // avg kernel time (may be NULL)
// returns 0 ok, -1 HIP/bad-args/throw, -2 unsupported args
```

Layouts: `A=[G..,M..,K..]`, `B=[G..,N..,K..]`, `E=[G..,M..,N..]`; the lib derives packed
row-major strides (matches the Old-TE `HostTensorDescriptor(dims)`), allocates/copies
each buffer, launches with `k_batch`, copies E back. Supports `NUM_D_TENSORS`
`0..8`: the `run()` also accepts the D-tensor pointers (D byte-size keyed off the
codegen `DBaseDataType` typedef) for the `MultiDAdd`/`MultiDMultiply` epilogue.

## Coverage (v1) — GPU-verified on gfx950

- **dtype:** `fp16`, `bf16`, `fp32` — all numerically verified vs the fp32 reference
  (max_rel: fp16 ~5e-4, bf16 ~4e-3, fp32 ~1e-4). Each needs a dtype-appropriate MFMA
  warp tile (fp16/bf16: `32x32x16`/`16x16x16`/`16x16x32`; fp32: `16x16x4`/`16x16x16`/`32x32x8`).
- **layout:** `rcr` only. Column-major A/B (`rrr`/`ccr`/`crr`) trip kernel
  `static_assert`s ("B block window has incorrect lengths for defined BLayout") and do
  not compile for these tiles, so v1 scopes to `rcr` (enforced in `is_valid()`).
- **dims:** arbitrary `num_dim_g/m/n/k` (default 1/1/1/1) — the ABI marshals the
  variable-length dim/stride vectors; multi-dim g/m/n/k verified.
- **pipeline/scheduler:** `{compv3,compv4,mem} x {intrawave,interwave}` (all verified).
- **epilogue:** `cshuffle` (v1); `default` is emitted by codegen but not swept.
- **num_d_tensors:** `0..8`. `num_d==0` is a plain contraction (`PassThrough`);
  `num_d>0` runs the D-tensor epilogue (`MultiDAdd` = `C + D0 + D1 + ...`,
  `MultiDMultiply` = `C * D0 * D1 * ...`), matching Old-TE
  `reference_batched_contraction.hpp` / `ck_tile::element_wise::MultiD*`. Each D
  tensor has E's shape `[G,M,N]` and the A/B dtype; the runner constructs them,
  marshals them through the ABI, and `reference()` applies the same epilogue in
  fp32. GPU-verified on gfx950 vs fp32 reference: num_d=1 MultiDAdd max_rel 7.14e-4,
  num_d=2 MultiDAdd 7.07e-4, num_d=1 MultiDMultiply 8.18e-4. `is_valid()` gates the
  count (0..8) and enforces num_d<->elementwise consistency.
- **k_batch:** `1` only. Split-K (`k_batch>1`) is a **shared Old-TE kernel defect**,
  not a bridge gap. The batched-contraction CShuffle epilogue is hard-wired to
  `memory_operation_enum::set` (no atomic accumulation), while the grid launches
  `k_batch` `blockIdx.z` K-split blocks that all write the **same** E tile with no
  atomic. Driving the exact Old-TE kernel at `k_batch=2` faults with an illegal
  memory access on gfx950 (`k_batch=1` is correct, max_rel ~4e-4). The bridge
  hard-rejects `k_batch>1` (returns -1, never silently-wrong) — out of scope until
  the shared kernel gains atomic accumulation.
- **problem sizes:** tile-multiple M/N/K. Non-multiples (e.g. 130) are rejected by the
  kernel's `IsSupportedArguments` (surfaced as rc=-2), even with padding flags.

## Note on warp tiles

The Old-TE `configs/default_config.json` lists `warp_tile 32x32x64`; that k=64 warp
tile is not in the fp16 XDL allow-list and does not build. The bridge configs
(`configs/bridge_default*.json`) use the validated fp16 point `32x32x16`, and
`is_valid()` gates warp tiles by dtype.

## Building and running

```bash
# CPU-only unit tests (no GPU)
python3 -m pytest dispatcher/tests/test_batched_contraction_bridge.py -v

# Codegen smoke (no GPU): one config
python3 dispatcher/codegen/unified_batched_contraction_codegen.py \
    --output-dir /tmp/bc --config-json '{"datatype":"fp16","layout":"rcr",
      "tile_config":{"tile_m":128,"tile_n":128,"tile_k":64,"warp_m":2,"warp_n":2,"warp_k":1,
      "warp_tile_m":32,"warp_tile_n":32,"warp_tile_k":16},"num_dim_g":1,"num_dim_m":1,
      "num_dim_n":1,"num_dim_k":1,"num_d_tensors":0}'

# Codegen smoke with a D-tensor epilogue (num_d>0, MultiDAdd)
python3 dispatcher/codegen/unified_batched_contraction_codegen.py \
    --output-dir /tmp/bc_d --config-json '{"datatype":"fp16","layout":"rcr",
      "tile_config":{"tile_m":128,"tile_n":128,"tile_k":64,"warp_m":2,"warp_n":2,"warp_k":1,
      "warp_tile_m":32,"warp_tile_n":32,"warp_tile_k":16},"num_dim_g":1,"num_dim_m":1,
      "num_dim_n":1,"num_dim_k":1,"num_d_tensors":1,"elementwise":"MultiDAdd"}'

# End-to-end bridge sweep + verify on GPU
python3 tile_engine/ops/gemm/batched_contraction_full_benchmark.py \
    --arch gfx942 --verify --csv batched_contraction_results.csv
```
