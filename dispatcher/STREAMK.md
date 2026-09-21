# Stream-K GEMM (Dispatcher Deep-Core Path)

Stream-K is a single GEMM that splits the **K** dimension across compute units (CUs)
and reduces the partial results, instead of giving each CU a whole output tile. It
keeps every CU busy on shapes where a classic data-parallel tiling would leave some
idle (tall-skinny / large-K problems), at the cost of a reduction step.

This document explains how to **generate**, **build**, **run**, and **test** the
Stream-K kernels through the CK Tile dispatcher.

> **Validated platform:** AMD Instinct MI300X (gfx942). See [Known limitations](#known-limitations)
> for gfx950 (MI350) status.

---

## Why Stream-K needs its own path

A plain GEMM rides `Dispatcher::run(A, B, C, problem)`. Stream-K cannot use that
signature unchanged: it needs a **reduction workspace** and a **reduction strategy**,
so its host args type (`ck_tile::StreamKHostArgs`) is ABI-incompatible with the
regular `GemmHostArgs`. The deep-core path makes Stream-K ride the registry anyway:

```
codegen (unified_gemm_codegen.py)
   -> generated Stream-K kernel + dispatcher wrapper
   -> Registry::register_kernel(GeneratedStreamKKernelInstance)
   -> Dispatcher::select_kernel(Problem.streamk + reduction_strategy)
   -> GeneratedStreamKKernelInstance::run()  (Dispatcher owns the workspace)
   -> SelectedKernel::launch(StreamKHostArgs, cfg, workspace)
```

### Reduction strategies

The reduction strategy is a **compile-time** property, so each strategy is a
*distinct kernel*. The registry holds all three side by side and the dispatcher
selects by `Problem::reduction_strategy`:

| Strategy | Workspace | Identifier suffix | Notes |
|---|---|---|---|
| `atomic` | none | `_streamk` | partials accumulate directly into C via atomics |
| `linear` | yes | `_streamk_linear` | partials reduced through a device workspace, in order |
| `tree`   | yes | `_streamk_tree`   | tree reduction through a device workspace |

### Supported datatypes / layouts

- **Datatypes:** `fp16`, `bf16`, `fp8`, `bf8`. (`fp32`/`fp64` have no MFMA warp tiles;
  `int8` Stream-K is out of scope for this path.)
- **Layouts:** `rcr`, `rrr`, `ccr`, `crr` — A/B in either order, **C is row-major**
  (the atomic C-reset relies on it).

---

## Prerequisites

A full ROCm toolchain with HIP headers (`hip/hip_runtime.h`) and `hipcc`. Bare SLURM
compute nodes on the cluster often ship an incomplete ROCm, so build inside the CK
ROCm container, e.g.:

```bash
# on a GPU node (pyxis/enroot), mounting your home:
srun --jobid=<JOBID> --overlap \
  --container-image=/cluster/images/ck/ck_rocm7.1.1_therock_<date>.sqsh \
  --container-mounts=$HOME:$HOME \
  bash -lc '<commands below>'
```

---

> All commands below are run from the dispatcher root
> (`projects/composablekernel/dispatcher`).

## 1. Generate a Stream-K kernel

The codegen emits all three reduction-strategy headers from one tile config:

```bash
python3 codegen/unified_gemm_codegen.py \
    --datatype fp16 --layout rcr \
    --gpu-target gfx942 \
    --variants stream_k \
    --tile-config-json '{
      "tile_config":  {"tile_m":[128],"tile_n":[128],"tile_k":[64],
                       "warp_m":[2],"warp_n":[2],"warp_k":[1],
                       "warp_tile_m":[32],"warp_tile_n":[32],"warp_tile_k":[16],
                       "block_size":[256]},
      "trait_config": {"pipeline":["compv3"],"epilogue":["cshuffle"],"scheduler":["intrawave"],
                       "pad_m":[false],"pad_n":[false],"pad_k":[false],"persistent":[false]},
      "streamk_config": {"reduction_strategy":["atomic","linear","tree"]}
    }' \
    --output-dir ./gen_fp16_rcr
```

This produces, per strategy, a header named:

```
gemm_<dtype>_<layout>_compv3_cshuffle_intrawave_<padM>_<padN>_<padK>_<persistent>_<TILE>_<variant>.hpp
# variant ∈ { streamk, streamk_linear, streamk_tree }
```

Each header force-includes into the global namespace: `SelectedKernel`,
`ADataType/BDataType/CDataType/AccDataType`, `ALayout/BLayout/CLayout`, `KERNEL_NAME`.

Omit `--tile-config-json` to generate the full arch-filtered tile set instead of a
single config. Use `--show-arch-info` to print what a target GPU supports.

---

## 2a. Run via the standalone driver (`03_streamk_gemm_driver.cpp`)

Calls `SelectedKernel::launch()` **directly** (bypasses the dispatcher). Use this for
apple-to-apple performance measurement against Tile Engine.

```bash
HDR=gen_fp16_rcr/gemm_fp16_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x64_2x2x1_32x32x16_streamk.hpp

hipcc -std=c++17 --offload-arch=gfx942 -O3 \
  -DCK_TILE_SINGLE_KERNEL_INCLUDE \
  -I ../include -I gen_fp16_rcr \
  -include "$HDR" \
  examples/gemm/cpp/03_streamk_gemm_driver.cpp -o streamk_gemm_driver

# performance (cold cache, TE-matched defaults):
./streamk_gemm_driver --m 4096 --n 4096 --k 4096 --validate 0
# correctness (single cold shot so C matches the reference):
./streamk_gemm_driver --m 4096 --n 4096 --k 4096 --validate 1
```

| Option | Default | Meaning |
|---|---|---|
| `--m/--n/--k` | 3840/4096/2048 | GEMM dims |
| `--warmup` | 50 | warmup iterations (timing) |
| `--repeat` | 100 | timed iterations |
| `--validate` | 1 | verify vs `reference_gemm`; forces 1 cold shot, no rotation |
| `--timer` | 1 | use the GPU timer |
| `--flush_cache` | 1 | flush L2 each iter (cold measurement, like Tile Engine) |
| `--rotating_count` | 1000 | rotating input copies to defeat cache (Tile Engine default) |

> **Methodology:** leaving the cache warm over-reports TFlops and is the entire
> source of spurious "dispatcher vs Tile Engine" perf gaps. Always measure perf with
> the cold-cache defaults (`--validate 0`); run correctness separately (`--validate 1`).

---

## 2b. Run via the registry/dispatcher (`04_streamk_registry_driver.cpp`)

Exercises the **full deep-core path**: registers the kernel, lets the dispatcher
select it by `Problem::reduction_strategy`, runs it (dispatcher owns the workspace),
and verifies vs the reference with a **split-K-aware tolerance**.

```bash
HDR=gen_fp16_rcr/gemm_fp16_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x64_2x2x1_32x32x16_streamk.hpp

# core objects (once, no force-include):
hipcc -std=c++17 --offload-arch=gfx942 -O3 -I ../include -I include -c src/dispatcher.cpp -o dispatcher.o
hipcc -std=c++17 --offload-arch=gfx942 -O3 -I ../include -I include -c src/registry.cpp   -o registry.o

# driver (force-include one strategy's header):
hipcc -std=c++17 --offload-arch=gfx942 -O3 \
  -DCK_TILE_SINGLE_KERNEL_INCLUDE -DGFX_ARCH='"gfx942"' \
  -I ../include -I include -I gen_fp16_rcr -include "$HDR" \
  -c examples/gemm/cpp/04_streamk_registry_driver.cpp -o drv04.o
hipcc --offload-arch=gfx942 drv04.o dispatcher.o registry.o -o streamk_registry_driver

./streamk_registry_driver --m 3840 --n 4096 --k 2048 --strategy atomic --validate 1
```

| Option | Default | Meaning |
|---|---|---|
| `--m/--n/--k` | 3840/4096/2048 | GEMM dims |
| `--strategy` | atomic | `atomic` / `linear` / `tree` (must match the force-included header) |
| `--validate` | 1 | verify vs `reference_gemm` (split-K-aware rtol/atol) |

> The registry `run()` path is a functional dispatch path; its `Perf:` line is a
> cold-but-**non-rotated** measurement, **not** the calibrated apple-to-apple surface.
> Use the `03` driver (`--validate 0`) for Tile-Engine-comparable numbers.

---

## 3. Test (CTest)

The deep-core path is guarded by `test_streamk_registry.py`, which generates, builds,
dispatches, and verifies every `datatype × layout × strategy` against two shapes
(the default plus a small-M/large-K shape that stresses the split-K tolerance). It
**SKIPs** (exit 77) when no GPU or `hipcc` is present.

```bash
# directly:
python3 tests/test_streamk_registry.py --arch gfx942
python3 tests/test_streamk_registry.py --arch gfx942 --datatypes fp16,bf16 --layouts rcr,ccr

# via ctest (from your dispatcher build dir):
ctest -R dispatcher_test_streamk_registry --output-on-failure
```

---

## Verification tolerance (why Stream-K is special)

Stream-K reduces `kbatch` partial products into each output element, so the
accumulation error is larger than a single-pass GEMM. The drivers use the same
split-K-aware tolerance as Tile Engine (`calculate_rtol_atol`): `kbatch` is taken
from the kernel's own tile partitioner, and the tolerance is
`max(per-split threshold, split-K-reduction threshold)`. Using the plain
`get_relative/absolute_threshold(K)` here spuriously FAILs correct atomic results on
small-M/N, large-K shapes.

---

## Known limitations

- **gfx950 (MI350) fp8/bf8 not validated.** On CDNA4 the fp8/bf8 host reference/codec
  hits an FNUZ-vs-OCP format mismatch; those combos currently fail verification. fp16
  and bf16 are fine on gfx950. Validate/gate before enabling fp8/bf8 there.
- **Tile coverage is narrower than Tile Engine.** The dispatcher emits fewer Stream-K
  tiles than TE (e.g. fp16 `rcr` TE=180 vs DISP=73). Numeric+perf parity is validated
  per matched tile config, not over the whole TE tile surface. See the coverage note
  at the `STREAM_K` variant in `codegen/unified_gemm_codegen.py`.

---

## File map

| Path | Role |
|---|---|
| `codegen/unified_gemm_codegen.py` | generates Stream-K kernels + dispatcher wrappers (`--variants stream_k`) |
| `include/ck_tile/dispatcher/backends/generated_tile_backend_streamk.hpp` | `GeneratedStreamKKernelInstance` (registry/workspace/launch glue) |
| `include/ck_tile/dispatcher/kernel_key.hpp` | registry key carrying `streamk` + `reduction_strategy` |
| `examples/gemm/cpp/03_streamk_gemm_driver.cpp` | standalone driver (direct `launch`, perf surface) |
| `examples/gemm/cpp/04_streamk_registry_driver.cpp` | deep-core driver (Registry → Dispatcher → verify) |
| `tests/test_streamk_registry.py` | CTest `dispatcher_test_streamk_registry` |
