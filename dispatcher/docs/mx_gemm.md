# MX GEMM bridge

The bridge runs CK-Tile microscaling GEMM with FP8 (OCP E4M3) or packed FP4
(E2M1) inputs, one E8M0 scale per 32 K elements, FP32 accumulation, and FP16
output. Layout is RCR: A is stored as `[M, K]`, B as `[N, K]`, and C as `[M, N]`.
FP4 stores the even K element in the low nibble and the odd K element in the high
nibble, so its input storage is `[M, K/2]` and `[N, K/2]` bytes.

The bridge exposes all five pipeline choices in the native MX GEMM selector:

| Target | Pipeline | Native implementation | Epilogue |
| --- | --- | --- | --- |
| gfx950 | `comp_async` | `GemmPipelineAgBgCrCompAsync` | CShuffle |
| gfx950 | `comp_async_eight_waves` | `GemmPipelineAgBgCrCompAsyncEightWaves` | CShuffle |
| gfx950 | `weight_preshuffle` | `MXGemmPreshufflePipelineAGmemBGmemCRegV1` | CShuffle |
| gfx1250 | `comp_tdm` | `GemmPipelineAgBgCrCompTDMV1` | TDM |
| gfx1250 | `comp_tdm_v2` | `GemmPipelineAgBgCrCompTDMV2` | TDM |
| gfx1250 | `comp_async` | `GemmPipelineAgBgCrCompAsync` | CShuffle |
| gfx1250 | `comp_async_eight_waves` | `GemmPipelineAgBgCrCompAsyncEightWaves` | CShuffle |
| gfx1250 | `weight_preshuffle` | `MXGemmPreshufflePipelineAGmemBGmemCRegV1` | CShuffle |

These are the five choices in the native `MxGemmPipelineType` selector.
CK-Tile also has a separate `MXFlatmmKernel` family, including
`MXFlatmmPipelineAGmemBGmemCRegV1` and
`WeightPreshufflePipelineAGmemBGmemCRegTDM`. Those MXFlatMM paths are not
currently exposed by the old Tile Engine or this bridge.

All use intrawave scheduling. The default warp tile is 16 × 16 × 128;
gfx1250 TDM V1/V2 also expose 32 × 32 × 128 with either input type and
32 × 16 × 128 for FP4. Set `warp_tile_m=32` and `warp_tile_n=32` or `16`
in `MxGemmKernelConfig`, or in a custom Tile Engine JSON configuration.
gfx950 and the three CShuffle pipelines retain the 16 × 16 × 128 warp tile.
The default pipeline remains
`comp_async` on gfx950 and `comp_tdm` on gfx1250. Select another pipeline with
`default_fp8_config("gfx950", pipeline="weight_preshuffle")`,
`default_fp4_config("gfx950", pipeline="comp_async_eight_waves")`, or
`default_fp8_config("gfx1250", pipeline="comp_async")`. These helpers choose
compatible block tiles and warp counts for the selected pipeline.

Eight-wave async requires `4 × 2 × 1` warps and M/N block tiles divisible by
128, with a valid CShuffle distribution. Weight preshuffle requires
`1 × 4 × 1` warps and block tiles divisible by `32 × 128 × 256`; FP4 also
requires block N divisible by 512. The host reshuffles B for weight preshuffle,
so callers supply the same packed RCR input buffers for every pipeline.
On gfx1250, weight preshuffle requires problem N divisible by 16; the host
rejects other N sizes before reshuffling. Async and TDM require `2 × 2 × 1`
warps on gfx1250.
The generator and Python configuration validation apply the same architecture,
tile-distribution, and LDS limits.

The gfx1250 path uses WMMA without cluster launch and supports revision 0.
M and N may include partial tiles subject to the weight-preshuffle N alignment;
the host pads their scale buffers before
reshuffling. K must be divisible by both 128 and the selected block tile K.
Split-K, persistent execution, and K padding are not supported by this bridge's
gfx1250 path.

The FP8 32 × 32 warp tile is implemented by four 16 × 16 × 128 WMMA operations.
The dedicated FP4 32 × 32 specialization uses two scaled 32 × 16 × 128 FP4
operations. A 32 × 32 warp tile uses 32 FP32 accumulators per lane, compared
with eight for 16 × 16. Total register use depends on the block shape and
compiler; a larger warp tile does not guarantee higher throughput.

FP4 32 × 16 × 128 is an opt-in configuration to request the native
`v_wmma_scale_f32_32x16x128_f4` path. Tile Engine and the bridge do not
distinguish gfx1250 A0/B0 revisions; instruction support is left to the native
pipeline. The current C++ 32 × 16 trait has only the unscaled operation, and
both TDM V1/V2 reject MX builds with a missing scaled `wmma_intrinsic`
overload. Exposing the configuration does not imply a successful device
build. It is excluded from the default and CI sweeps.
All exposed warp tiles retain E8M0 scales per 32 K elements.

The dispatcher generator reuses `MxGemmKernelBuilder` from Tile Engine. Both
entry points use the same generated kernel, while their host code chooses the
architecture's native scale reshuffling helper. Select one GPU architecture per
Tile Engine build.

## Run the bridge

From `projects/composablekernel`, with NumPy installed and a matching ROCm
compiler/runtime available:

```bash
export CK_TILE_HIPCC=/opt/rocm/bin/hipcc
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export CK_TILE_BENCH_WARMUP=1
export CK_TILE_BENCH_REPEAT=2
PYTHONPATH=dispatcher/python python3 - <<'PYCODE'
from pathlib import Path
import numpy as np
from mx_gemm_utils import (
    GpuMxGemmRunner, MxGemmProblem, default_fp4_config, default_fp8_config,
    setup_multiple_mx_gemm_dispatchers,
)

for make_config in (default_fp8_config, default_fp4_config):
    config = make_config("gfx1250", pipeline="comp_tdm_v2")
    library = setup_multiple_mx_gemm_dispatchers(
        [config], output_dir=Path("build/mx_bridge"), gfx_arch="gfx1250",
        parallel=False,
    )[0]
    assert library is not None, "Kernel compilation failed"
    runner = GpuMxGemmRunner(library, dtype=config.datatype, arch="gfx1250")
    problem = MxGemmProblem(129, 257, 384)
    a_ref, b_ref, a, b, sa, sb = runner.make_inputs(problem, scale=1.0, seed=5)
    rng = np.random.default_rng(19)
    sa[:] = rng.integers(124, 130, size=sa.shape, dtype=np.uint8)
    sb[:] = rng.integers(124, 130, size=sb.shape, dtype=np.uint8)
    result = runner.run(problem, a, b, sa, sb)
    reference = runner.reference(a_ref, b_ref, sa, sb, problem).astype(np.float32)
    got = np.asarray(result.C, dtype=np.float32)
    denominator = np.abs(reference) + max(np.abs(reference).max() * 1e-2, 1e-6)
    error = np.max(np.abs(got - reference) / denominator)
    assert np.isfinite(got).all() and error <= 5e-2
    print(config.datatype, "PASS", "max_rel=", error, "time_ms=", result.time_ms)
PYCODE
```

## Regression tests

CPU tests cover architecture selection and target suffixes, invalid configurations,
LDS boundaries, scale/packing codecs, standalone Tile Engine entry points, the CI
configuration, and exact generated-header parity with Tile Engine:

```bash
python3 -m unittest discover -s dispatcher/tests -p test_mx_gemm_bridge.py -v
```

The CMake tests check MX target selection and effective compiler definitions.
They require CMake and a host C++ compiler, and skip when either is unavailable;
no GPU or HIP compiler is needed:

```bash
python3 -m unittest discover -s dispatcher/tests -p test_mx_gemm_cmake.py -v
```

Both suites are registered with CTest when dispatcher tests are enabled:

```bash
ctest --test-dir build -R '^dispatcher_test_mx_gemm_(bridge|cmake)$' --output-on-failure
```

The GPU suite builds all 16 gfx1250 TDM CI configurations, four larger TDM
configurations whose LDS allocation exceeds 64 KiB, six configurations covering
async, eight-wave async, and weight preshuffle in FP4 and FP8, and four TDM
configurations with the 32 × 32 × 128 warp tile: 30 configurations total. It tests
partial tiles and one through five and eight K-loop iterations with two seeds,
varies scales across rows and K blocks, and repeats eight-wave launches to catch
buffer-reuse races. It checks K-tail/split-K rejection and weight-preshuffle N
alignment. On gfx950
it tests all three pipelines with FP4 and FP8 over four block-relative shapes
and two seeds:

```bash
CK_TILE_BENCH_WARMUP=1 CK_TILE_BENCH_REPEAT=2 \
python3 -m unittest discover -s dispatcher/tests -p test_mx_gemm_gpu_correctness.py -v
```

The native C++ regression uses CK-Tile directly:

```bash
cmake -S . -B build/mx_native \
  -DBUILD_DEV=ON -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1250 \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build/mx_native --target test_ck_tile_mx_gemm_e8m0_gfx1250 --parallel 2
ctest --test-dir build/mx_native -R test_ck_tile_mx_gemm_e8m0_gfx1250 --output-on-failure
cmake --build build/mx_native --target test_ck_tile_mx_gemm_splitk_support --parallel 2
ctest --test-dir build/mx_native -R test_ck_tile_mx_gemm_splitk_support --output-on-failure
```

For Tile Engine, enable `BUILD_CK_TILE_ENGINE`, use `GPU_TARGETS=gfx1250`, and
build `benchmark_mx_gemm_all`. The MX operation selects
`default_config_gfx1250.json` automatically unless a custom MX config is
provided. It uses both `comp_tdm` and `comp_tdm_v2` with the TDM epilogue,
intrawave scheduling, `2 x 2 x 1` warps,
a `16 x 16 x 128` warp tile, and no padding or persistent execution. Block M/N
range from 64 to 256 in steps of 64; block K is 128 or 256. All 32 block shapes
fit gfx1250's 320 KiB LDS capacity with both FP4 and FP8, including TDM descriptor
padding and both staging buffers: 64 kernels per pipeline, for 128 kernels total.
Both the bridge and Tile Engine use this architecture-aware capacity check.
This default set contains TDM kernels. The three CShuffle pipelines
are available through explicit configuration or the Python helpers above.

For the smaller 16-kernel CI set covering both TDM pipelines, pass
`-DMX_GEMM_CONFIG_FILE=default_ci_config_gfx1250.json` to CMake. The
`MX_GEMM_CONFIG_FILE` environment variable takes precedence over the CMake
option. gfx950 selects `default_config.json`, which covers all three gfx950
pipelines with CShuffle. Each configuration file contains pipelines supported
by its target architecture.

Benchmark executables accept `-m=`, `-n=`, `-k=`, `-verify=1`, `-init=0`,
`-warmup=1`, and `-repeat=2` for a short correctness run with random inputs.
