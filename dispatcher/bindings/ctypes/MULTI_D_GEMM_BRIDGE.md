<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Multi-D GEMM: Tile Engine -> Dispatcher Bridge

This document describes the **gemm_multi_d** variant of the Tile Engine (TE) ->
Dispatcher bridge. It is the multi-D counterpart of the regular-GEMM bridge
(#8997), the grouped bridge (#9000), and the Stream-K bridge (#9028).

## What the bridge is

In the bridge model the **Dispatcher is the single source of truth** for
codegen, build, and runtime; **Tile Engine only generates configs and
benchmarks** them. TE no longer carries its own kernel-instance build path — it
shells out to the dispatcher codegen and runs the resulting kernel.

Multi-D fuses extra D operands into the GEMM epilogue, so — like grouped and
Stream-K — it takes a dedicated ctypes library that **bypasses the registry** and
calls the generated `SelectedKernel::launch(...)` directly.

## Why multi-D needs special handling

Multi-D computes `E = elementwise_op(A @ B, D0, D1, ...)`: a fixed number
(`NumDTensor`) of extra device pointers are fused into the CShuffle epilogue. Two
consequences:

1. The single-problem run path (`g_dispatcher->run` / `GemmHostArgs`) cannot
   carry the D-pointer array or per-D strides.
2. The generated registry wrapper (`generated_tile_backend.hpp::run()`) ignores
   `d_ptrs` and calls the plain `SelectedKernel::launch(GemmHostArgs, ...)`
   overload (empty D tensors), so it cannot exercise real D operands.

So the multi-D kernel header exposes a different launch signature

```cpp
static float launch(const GemmMultiDArgs& args, const stream_config& stream);
// GemmMultiDArgs == GemmMultiDHostArgs<NumDTensor>
```

and the multi-D ctypes lib force-includes one generated kernel header
(`-include ..._multid_....hpp` with `CK_TILE_SINGLE_KERNEL_INCLUDE`), calls that
`launch` directly, and reports the kernel name from the compile-time
`KERNEL_NAME` macro.

## Components

| Layer | File | Role |
|---|---|---|
| Codegen | `dispatcher/codegen/unified_gemm_codegen.py` | `GemmVariant.MULTI_D` (already present): `_multi_d_types`, `_launch_function_multi_d`, `_epilogue_code` (CShuffle multi-D). This PR adds `_multi_d_single_include`: re-exports `NumDTensor`/`DsDataType`/`DsLayout`/`DLayout`/`ElementWiseFn`/`GemmMultiDArgs` + `ALayout`/`BLayout`/`CLayout` and the `GEMM_KEY_MULTI_D`/`GEMM_KEY_NUM_D_TENSORS`/`GEMM_KEY_ELEMENTWISE_OP`/`GEMM_KEY_D_LAYOUT` macros under `CK_TILE_SINGLE_KERNEL_INCLUDE`. |
| C API | `dispatcher/bindings/ctypes/multi_d_gemm_ctypes_lib.cpp` | Multi-D ABI; A/B/C + D device alloc/copy; layout-derived strides; fair-by-default timing (flush_cache=true, rotating_count=1000 to match Old-TE; env-tunable via `CK_TILE_BENCH_WARMUP`/`REPEAT`/`FLUSH`/`ROTATING`); `dispatcher_get_num_d_tensors`. |
| Python | `dispatcher/python/gemm_utils.py` | `MultiDGemmProblem` / `MultiDGemmResult`, `GpuMultiDGemmRunner`, `run_multi_d`, multi_d fields on `GemmKernelConfig` (`elementwise_op`/`num_d_tensors`/`d_layout`), 4-char `codegen_layout`, `_ctypes_source_name`, `expand_sweep(variant="multi_d")`. |
| TE driver | `tile_engine/ops/gemm/gemm_multi_d_full_benchmark.py` | Generates configs, builds `.so`s in parallel, benchmarks in disposable per-GPU workers. |
| TE worker | `tile_engine/ops/gemm/run_one_gemm_multi_d_kernel.py` | Runs one multi_d kernel; num-D read off the .so; fp32 reference for `--verify` (op(A@B, Ds)). |

## C ABI

```c
int dispatcher_init(void);                 // lightweight no-op (no registry)
int dispatcher_get_num_d_tensors(void);    // compiled-in NumDTensor
int dispatcher_run_multi_d_gemm(
    const void*  A,                        // host A (MxK)
    const void*  B,                        // host B (KxN)
    const void** d_ptrs,                   // num_d host D buffers, each MxN
    int          num_d,                    // MUST equal dispatcher_get_num_d_tensors()
    void*        C,                        // host E/C out (MxN)
    int64_t M, int64_t N, int64_t K,
    float*  time_ms);                      // out: average kernel time
// returns 0 ok, -1 HIP/bad-args/throw, -2 arguments unsupported by the kernel
```

The lib `hipMalloc`s A/B/C and each D, copies A/B/D host->device, memsets C,
builds `GemmMultiDArgs` with strides derived from the compile-time
`ALayout`/`BLayout`/`CLayout`/`DLayout` of the `-include`d header, launches with
`k_batch=1` (multi-D requires k_batch==1), then copies C back. The ABI is
`void*` + element-size; the Python runner owns the numpy codecs.

## Coverage (matches Old-TE gemm_multi_d exactly)

| Layout \ Op | MultiDAdd | MultiDMultiply | PassThrough |
|---|---|---|---|
| rcrr | ✓ | ✓ | ✓ |
| rrrr | ✓ | ✓ | ✓ |
| ccrr | ✓ | ✓ | ✓ |
| crrr | ✓ | ✓ | ✓ |

- **dtype:** fp16 only (the TE `gemm_multi_d_instance_builder.py` argparse
  restricts `--datatype` to `fp16`).
- **layouts:** 4-char, `rcrr`/`rrrr`/`ccrr`/`crrr` — A/B vary, C and D are always
  row-major (last two chars `r`), matching the TE builder.
- **num D tensors:** swept from `multi_d_config.num_d_tensors` (default `[1, 2]`).
  The **apples-to-apples parity default is num_d=2**: Old-TE
  `gemm_multi_d_benchmark_single.cpp` bakes `DsDataType = tuple<D0, D1>`
  (`DsDataType::size() == 2`), so num_d=2 is the byte-identical comparison and
  the headline parity slice. (Any num_d=1 claim is *not* the fair slice — Old-TE
  never builds a single-D multi_d kernel.)
- **elementwise ops:** `MultiDAdd`, `MultiDMultiply` (Add/Multiply are the
  multi-D-signature ops; `PassThrough` is also supported). Unary ops
  (Relu/Gelu) are excluded on both sides — wrong signature for multi-D.

## Known follow-ups

- **TODO (rocprof):** On the num_d=2 parity slice, 14/640 shapes show |gap|>15%
  at 4096^3 (memory-pipeline-bound). Both sides run the *same* shared-codegen
  kernel, so this is a large-shape mem-pipeline characteristic, not a bridge
  regression; a rocprof root-cause is deferred and does not block parity.

## Building and running

```bash
# Codegen smoke (no GPU): one layout/op/num_d
python3 dispatcher/codegen/unified_gemm_codegen.py \
    --output-dir /tmp/md --datatype fp16 --layout rcrr \
    --variants multi_d --config dispatcher/codegen/default_config.json

# Full A/B parity sweep vs Old-TE (per GPU, subprocess-isolated, --verify checks
# each kernel against an fp32 op(A@B, Ds) reference):
python3 tile_engine/ops/gemm/gemm_multi_d_full_benchmark.py \
    --layout rcrr --verify --csv gemm_multi_d_results.csv
```
