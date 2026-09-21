<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Grouped GEMM: Tile Engine -> Dispatcher Bridge

This document describes the **grouped_gemm** variant of the Tile Engine (TE) ->
Dispatcher bridge (PR #8130). It is the grouped counterpart of the regular-GEMM
bridge (#8123/#8479), the fp8/bf8/int8 bridge (#8887), and the Stream-K bridge
(#8136).

## What the bridge is

In the bridge model the **Dispatcher is the single source of truth** for
codegen, build, and runtime; **Tile Engine only generates configs and
benchmarks** them. TE no longer carries its own kernel-instance build path —
it shells out to the dispatcher codegen and runs the resulting kernel.

For most variants the dispatcher runs a kernel through its registry/backend.
Grouped GEMM cannot use that path (see below), so the grouped bridge takes the
same approach as Stream-K: a dedicated ctypes library that **bypasses the
registry** and calls the generated `SelectedKernel::launch(...)` directly.

## Why grouped needs special handling

Grouped GEMM is **multi-problem**: a single launch runs a *list* of `(M, N, K)`
sub-problems, each with its own A/B/C device pointers. Two consequences:

1. The single-problem run path (`g_dispatcher->run` / `GemmHostArgs`) cannot
   express a list of problems.
2. The generated registry wrapper (`generated_tile_backend.hpp::run()`)
   hard-codes the single-problem `SelectedKernel::launch(GemmHostArgs, ...)`
   signature and will not compile against a grouped `SelectedKernel`.

So the grouped kernel header exposes a different launch signature

```cpp
static float launch(const std::vector<ck_tile::GroupedGemmHostArgs<>>& descs,
                    const stream_config& stream);
```

and the grouped ctypes lib force-includes one generated kernel header
(`-include ..._grouped.hpp` with `CK_TILE_SINGLE_KERNEL_INCLUDE`), calls that
`launch` directly, and reports the kernel name from the compile-time
`KERNEL_NAME` macro.

## Components

| Layer | File | Role |
|---|---|---|
| Codegen | `dispatcher/codegen/unified_gemm_codegen.py` | `GemmVariant.GROUPED`; `_launch_function_grouped` (DeviceMem internal workspace, `MakeKargs`, persistent/non-persistent grid). Kept in lockstep with PR #8075. |
| Codegen | `dispatcher/codegen/arch_filter.py` | `GEMM_GROUPED` operator tile constraints. |
| C API | `dispatcher/bindings/ctypes/grouped_gemm_ctypes_lib.cpp` | Multi-problem ABI; per-group device alloc/copy; layout-derived strides; warmup/repeat timing. |
| Python | `dispatcher/python/gemm_utils.py` | `GroupedGemmProblem` / `GroupedGemmResult`, `GpuGroupedGemmRunner`, `run_grouped`, `build_grouped`, dtype/layout codecs. |
| Python | `dispatcher/python/ctypes_utils.py` | Threads the `grouped` variant into the codegen `--variants` flag. |
| TE driver | `tile_engine/ops/gemm/grouped_gemm_full_benchmark.py` | Generates configs, builds `.so`s in parallel, benchmarks in disposable workers. |
| TE worker | `tile_engine/ops/gemm/run_one_grouped_gemm_kernel.py` | Runs one grouped kernel; dtype/layout-aware operand generation. |

## C ABI

```c
int dispatcher_init(void);                 // lightweight no-op (no registry)
int dispatcher_run_grouped_gemm(
    int            group_count,
    const int64_t* Ms,                     // [group_count]
    const int64_t* Ns,                     // [group_count]
    const int64_t* Ks,                     // [group_count]
    const void**   A_ptrs,                 // host A buffers, one per group
    const void**   B_ptrs,                 // host B buffers, one per group
    void**         C_ptrs,                 // host C out buffers, one per group
    float*         time_ms);               // out: average kernel time
// returns 0 ok, -1 HIP/throw, -2 arguments unsupported by the kernel
```

The lib `hipMalloc`s A/B/C per group, copies A and B host->device, memsets C,
builds `std::vector<ck_tile::GroupedGemmHostArgs<>>` with **strides derived from
the compile-time `ALayout`/`BLayout`/`CLayout`** of the `-include`d header
(`std::is_same_v<…, RowMajor>`), launches once, then copies each C back. The ABI
is `void*` + element-size, so it is dtype-agnostic; the Python runner owns the
numpy codecs.

## Coverage

The bridge runnable set is exactly the Old-TE grouped_gemm runnable set on
`develop` — no more, no less:

| Layout \ Dtype | fp16 | bf16 | fp8 (E4M3) | bf8 (E5M2) |
|---|---|---|---|---|
| rcr | ✓ | ✓ | ✓ | ✓ |
| rrr | ✓ | ✓ | ✓ | ✓ |
| ccr | ✓ | ✓ | ✓ | ✓ |
| crr | ✓ | ✓ | ✓ | ✓ |

- **Matrix C is always row-major** (grouped builder constraint), so the layout
  string varies A/B only.
- **Excluded:** `int8` (rejected by the TE grouped builder), `fp32`/`fp64`
  (no MFMA warp tiles). These are excluded on both sides.
- fp8/bf8 use the **FNUZ** encoding on gfx942 (matches the regular #8887 path);
  the Python codecs require `ml_dtypes`.

## Building and running

Generate + build one grouped `.so` and run the A/B parity sweep vs Old-TE:

```bash
# Codegen smoke (no GPU): one variant/dtype/layout
python3 dispatcher/codegen/unified_gemm_codegen.py \
    --output-dir /tmp/grp --datatype bf16 --layout ccr \
    --variants grouped --config dispatcher/codegen/default_config.json

# Full TE-driven parity sweep (build + benchmark)
python3 tile_engine/ops/gemm/grouped_gemm_full_benchmark.py <config.json> \
    --arch gfx942 --dtype fp16 --layout rcr --csv grouped_results.csv
```

Timing knobs `CK_TILE_BENCH_WARMUP` (default 50) and `CK_TILE_BENCH_REPEAT`
(default 100) are honored by **both** the grouped ctypes lib and the registry
backend, so bridge-vs-Old-TE A/B comparisons stay matched. For fair parity keep
`flush_cache=false`, `rotating_count=1`, run on a single GPU, and re-measure any
`|gap|>15%` outlier standalone.
