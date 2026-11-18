# Changelog for Composable Kernel

Documentation for Composable Kernel available at [https://rocm.docs.amd.com/projects/composable_kernel/en/latest/](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/).

## (Unreleased) Composable Kernel

### Added

### Changed

### Removed

### Optimized

### Resolved issues

### Known issues

### Upcoming changes


## Composable Kernel 1.4.0 for ROCm 10.3.0

### Added

* Added support for the gfx1170 and gfx1171 architectures (`gfx11-7-generic`) to CK and CK Tile, following the gfx11 feature set (no FP8 or BF8 WMMA, MX, or split-K). CK Tile FMHA forward, backward, append-KV, split-KV, and paged-KV prefill are supported for fp16 and bf16, with the forward tiles tuned for gfx117.
* Added gfx1251 support to CK and CK Tile, which builds gfx1251 with the gfx1250 code paths and adds the fp64 16x16x4 WMMA instruction to the CK warp GEMM.
* Added the `CK_HEADERS_ONLY` CMake option, which installs only the headers and the CMake package without building instance libraries, the profiler, examples, or tests, and the `headers` package component with the `composable_kernel::headers` target for consumers that instantiate the CK templates themselves.
* Added the `LAYERNORM2D_FWD_INSTANCE_FILTER` and `GEMM_QUANT_INSTANCE_FILTER` CMake options, which build only the matching instances of the CK Tile LayerNorm2D forward and quantized GEMM examples.
* Added the `CK_ENABLE_POST_RA_SCHEDULING` and `CK_ENABLE_EXPERT_SCHEDULING` CMake options (both `OFF` by default), which enable the LLVM post-RA machine scheduler and the AMDGPU expert scheduling mode for CK builds.
* Added `comp_tdm_v3`, a CK Tile GEMM pipeline for gfx1250 that uses TDM with 4 or 16 waves and a split workgroup barrier, for fp8 and bf8 inputs (row-major A, column-major B), MX fp8 with E8M0 scales per 32 K elements, and fp16 and bf16 inputs in any A and B layout, with row-major C. It is also available, with the CShuffle epilogue, in the CK Tile dispatcher and Tile Engine.
* Added `comp_tdm_pc`, a producer-consumer TDM GEMM pipeline for gfx1250 in which two loader waves fill a ring of LDS buffers synchronized by hardware named barriers while the other waves compute. It is available in the CK Tile GEMM, batched GEMM, grouped GEMM, batched contraction, multiple D, and multiple ABD kernels, with a ring depth and publish lag that the CK Tile dispatcher and Tile Engine can configure.
* Added CK Tile GEMM Stream-K support on gfx1250 with atomic, linear, and tree reduction, including the comp_tdm, comp_tdm_v2, comp_tdm_v3, and comp_tdm_pc pipelines. All four are also available in the CK Tile dispatcher, its Stream-K benchmark (which adds a `--budget-pct` workgroup budget option), and the Tile Engine. C must be row-major, atomic reduction requires the CShuffle epilogue, and the TDM pipelines do not support padding.
* Added the gfx1250 TDM GEMM pipelines (comp_tdm and comp_tdm_v2) to the CK Tile grouped GEMM, GEMM multiple D, and batched contraction kernels, with a new `TdmMultiDEpilogue` for row-major D and E tensors. Padding, split-K, persistent mode, and cluster launch are not supported on this path.
* Added support for the row-column, tensor, A, B, and AB quantization modes of the CK Tile quantized GEMM kernel on the general GEMM pipelines (Mem, CompV3, CompV4, CompAsync, and the gfx1250 comp_tdm and comp_tdm_v2 pipelines), not only on the dedicated quantized pipelines, with the Intrawave scheduler and non-preshuffled quantization scales.
* Added gfx1250 support to the CK Tile dispatcher and Tile Engine preshuffle GEMM for fp16, bf16, fp8, and bf8 with the rcr layout, using the weight-preshuffle `preshufflev2` and `preshuffle_tdm` pipelines and the `comp_tdm`, `comp_tdm_v2`, and `comp_async` pipelines, which take an unshuffled B. The Tile Engine builds these kernels only when gfx1250 is the only GPU target.
* Added FP6 (E2M3) operands and K tails to the CK Tile microscaling (MX) GEMM (`MxGemmKernel`) on the gfx1250 TDM pipelines. FP6 must be used for both A and B, with a 16x16x128 warp tile and no padding; BF6 is not supported. With row-major A, column-major B, 32-element E8M0 scale blocks, and a K tile that is a multiple of 128, K no longer has to be a multiple of the tile size: any K for fp8, an even K for fp4, and a multiple of 16 for fp6. The MX grouped GEMM still requires whole K tiles.
* Added format-explicit FP8 types to CK Tile (`fp8_fnuz_t`, `bf8_fnuz_t`, `fp8_ocp_t`, and `bf8_ocp_t`), which the warp GEMM maps to native FP8 instructions on devices that support the format.
* Added an opt-in `WiderInstruction` template parameter to the CK Tile `get_k_warp_tile()` helper, which selects the 16x16x128 FP8 and BF8 WMMA instruction on gfx1250 instead of the default 16x16x64 instruction.
* Added the `qr_tdm_fp8_sched` CK Tile FMHA forward pipeline on gfx1250 for the `fp8`, `fp8bf16`, `mxfp8`, and new `mxfp8bf16` data types, with head dimensions 64, 128, 256, and 192/128, causal and sliding-window masks, elementwise bias (`fp8` and `fp8bf16` only), LSE, sink, GQA and MQA, and batch and group mode.
* Added head dimension 512, logits soft cap, dropout, group-mode `skip_min_seqlen_q`, and Q and V head dimensions that are not multiples of 8 to the gfx1250 fp16 and bf16 CK Tile FMHA forward TDM pipeline (`qr_tdm`).
* Added CK Tile FMHA batch prefill on gfx1100 in the CK Tile dispatcher for fp16 and bf16, with head dimensions 64, 96, and 128 and a linear paged KV cache. Dropout and logits soft cap are not supported.
* Added a PyTorch extension for CK Tile FMHA (`ck_tile_fmha`, in `example/ck_tile/01_fmha/python`) with fixed- and variable-length attention, KV cache, bias gradients, and FP8 forward, on gfx942, gfx950, and gfx1250.
* Added direct-load wavelet pipeline instances for XDL grouped convolution backward data and backward weight on gfx950 and gfx1250, for 2D and 3D NHWGC and NDHWGC layouts with fp16 and bf16, using TDM loads on gfx1250.
* Added an opt-in runtime clamp limit to the XDL MoE GEMMs. With the `RuntimeSwigluLimit` template parameter, `DeviceMoeGemm` (`silu_and_mul` and `swiglu_oai_and_mul`) and `DeviceMoeGemmBlockScale` (stage-1 `silu_and_mul` without split-K) clamp the gate and up values to the positive `swiglu_limit` argument of `MakeArgument`. This replaces the fixed 7.0 bound of `swiglu_oai_and_mul` and adds a clamp to `silu_and_mul`.
* Added N and K padding to the XDL two-stage MoE GEMM with preshuffled weights (`DeviceMoeGemm`) for bf16 and fp8, so instances with a padding `GemmSpecialization` accept model and intermediate sizes that are not multiples of the tile size, as long as N is a multiple of 16 (8 for combined gate and up weights) and K is a multiple of the 16-element (8-element for some FP8 weights) weight vector.
* Added gfx90c support to the FP32 DL GEMM (`DeviceGemmDl`) for column-major A and row-major B and C, for applications that instantiate the CK headers directly.
* Added an opt-in adaptive kernel search to the Tile Engine GEMM benchmark (`gemm_full_benchmark.py --search-config`), including the gfx1250 TDM pipelines. It builds and measures only the kernels a search strategy selects (random, exhaustive, differential evolution, hill climbing, TPE, Gaussian process, or a portfolio) and exports the verified winners as a `--pairs` file.
* Added Tile Engine GEMM benchmarking tools: `mx_gemm_full_benchmark.py` for MXFP4 and MXFP8 GEMM sweeps through the dispatcher on gfx950 and gfx1250, measurement options for `gemm_full_benchmark.py` (`--build-timeout`, `--failures-csv`, `--repeat-top`, `--list-pairs`, and `--pairs`), dispatcher heuristics scripts that turn its CSV output into training data and select the top-ranked kernels per problem, and a gfx950 bf16 batched GEMM model for the dispatcher heuristics.

### Changed

* The CK Tile FMHA forward `fp8` data type now writes an FP32 output, as `mxfp8` does, so callers must size the O tensor for FP32. FP8 output is no longer provided, including by the split-KV and paged-KV instances. On gfx9 and gfx12, `fp8` now supports only head dimension 128; use `fp8bf16` for head dimensions 64, 192/128, and 256.
* On gfx1250, CK Tile FMHA forward split-KV and paged-KV for fp16 and bf16 now require Q and V head dimensions that are multiples of 8.

### Removed

* Removed the CK Tile FMHA forward `fp8fp32` data type. Use `fp8`, which now has the same FP32 output.

### Optimized

* Improved CK Tile FMHA forward performance on gfx1250 for fp16 and bf16 short queries on large grids, with a new `qr_tdm_pingpong` pipeline for head dimension 128 (including a 16-row tile for queries of up to 16 rows) and a non-temporal cache hint on the K and V loads of small `qr_tdm` tiles.
* Improved CK Tile FMHA forward performance on gfx1250 for head dimension 128 prefill, especially sliding-window attention with small windows, by storing the output asynchronously from LDS and overlapping the Q, K, and V prologue loads.
* Improved CK Tile FMHA forward performance on gfx1250 for head dimension 256, which regressed in the previous release, by using a single V LDS buffer and double-buffered whole K tiles with bank-conflict padding.
* Improved CK Tile FMHA forward performance on gfx1250 for bf16 head dimension 128 causal attention in batch mode with equal query and key lengths of up to 2048, by assigning whole batch-head rows to each XCD and processing the heaviest Q tiles first.
* Improved CK Tile FMHA forward performance in batch mode with per-batch query lengths (`cu_seqlen_q`) by ending workgroups past each batch's effective length early.
* Improved CK Tile FMHA forward split-KV and paged-KV performance on gfx1250 for fp16 and bf16 with a TDM-based split-KV pipeline for head dimensions 32 to 256 that supports batch and group mode, sink, bias, ALiBi, and logits soft cap.
* Improved CK Tile FMHA backward performance on gfx1250 for bf16 head dimension 128 training, decode (`seqlen_q` up to 32), padded head dimensions, and deterministic mode, which now also computes a bitwise-reproducible sink gradient (`d_sink`).
* Improved CK Tile FMHA backward performance by splitting each row of the dO-times-O preprocessing kernel (`FmhaBwdOGradDotOKernel`) across lanes and reading O with non-temporal loads.
* Improved CK Tile FMHA forward and backward performance with dropout on gfx1250 by computing the Philox random number generator's products with 32-bit multiplies.
* Improved fp16 and bf16 CK Tile FMHA forward performance on gfx11, gfx1200, and gfx1201 by retuning the default tiles, using separate K and V LDS buffers, reducing register pressure for padded head dimensions, and restoring the faster software bf16 conversion.
* Improved grouped convolution forward performance on gfx1250 with wavelet direct-load instances that load the weight tile with TDM and synchronize the compute waves with named barriers, for fp16 and bf16 2D (NHWGC) and 3D (NDHWGC) convolutions.

### Resolved issues

* Fixed CK support for gfx1250-strict: LDS transpose loads now compile for it, and the CK device checks accept gfx1250-strict devices.
* Fixed wrong FP8 results in the XDL FP8 GEMM, convolution, MoE, elementwise, CK Tile FlatMM, and block-scale GEMM examples when built for both gfx942 and gfx950. They now use FNUZ types on gfx942 and OCP types on gfx950.
* Fixed CK Tile TF32 GEMM compilation failures in builds that target both gfx942 and gfx950.
* Fixed incorrect results in the CK Tile TDM GEMM pipelines on gfx1250 for tile shapes where the TDM loads and the LDS reads used different LDS padding layouts, such as tiles whose LDS rows are not a power of two.
* Fixed incorrect results in the CK Tile CompAsync and CompAsyncV2 GEMM pipelines on gfx1250 when K is not padded, in CompAsyncV2 when the number of K iterations is odd (for example K = 192 or 320), and in CompAsyncV2 on gfx1250 for blocks that extend past the A or B tensor.
* Fixed incorrect results in CK Tile block GEMMs whose K iterations per warp are not a multiple of the K sub-tile count, such as 384-deep fp8 TDM tiles, which skipped some warp GEMMs.
* Fixed silently wrong results in CK Tile GEMM with fp32 output and in grouped convolution backward data and backward weight on gfx1250 when the TDM epilogue was used with split-K. These kernels now reject `k_batch > 1` with the TDM epilogue.
* Fixed silently wrong results from the CK Tile MX GEMM on the gfx1250 TDM pipelines for unsupported arguments. `IsSupportedArgument` now rejects packed (FP4) operands whose stride or contiguous extent is not a whole number of packs, such as a column-major FP4 A with M = 1.
* Fixed memory faults in CK Tile TDM transfers and TDM prefetches on gfx1250 with newer compilers, caused by reading the tensor descriptor through a dangling reference.
* Fixed intermittent wrong results in CK Tile GEMM Stream-K with linear or tree reduction on gfx12 (gfx1200, gfx1201, and gfx12-generic), where partial tiles could be read before their stores completed.
* Fixed wrong and out-of-bounds output from the CK Tile multi-reduction kernels when the output length is not a multiple of the block tile or vector size, in both the direct-store and global-atomic paths.
* Fixed the CK Tile dispatcher rejecting GEMM Stream-K kernels with linear or tree reduction whose C store is narrower than 4 bytes, so fp16 and bf16 problems with an odd N (including N = 1) get a linear or tree reduction kernel.
* Fixed incorrect CK Tile FMHA forward results with an attention sink in the split-KV, paged-KV, batch prefill, and qr pipelines, when a window mask left the leading split empty or a logits soft cap was used.
* Fixed all-zero CK Tile FMHA forward output and wrong FMHA backward gradients on gfx11 with newer compilers, caused by out-of-bounds thread-buffer accesses and a strict-aliasing violation in `PermuteWarpGemmCToA`.
* Fixed data races in CK Tile FMHA backward on gfx1250, in the LDS reuse between jobs of the persistent kernel and in the dS LDS buffer of the `tdm_kr_ktr` pipeline.
* Fixed wrong bias gradients and wrong dQ, dK, and dV with an elementwise bias in the CK Tile FMHA backward short-sequence pipeline on gfx950 and gfx1250.
* Fixed CK Tile FMHA forward on gfx950 dispatching `fp8bf16` sliding-window masks to kernels that support only causal masks.
* Fixed a possible data race in the CK Tile Jenga sparse attention pipeline by adding LDS memory fences around the barrier that publishes the asynchronously loaded K and V tiles.
* Fixed reduced bf16 accuracy of CK Tile FMHA kernels compiled from `generate.py` output by an external build system such as PyTorch. Generated sources now use the same round-to-nearest-even float-to-bf16 conversion as the CMake build: for backward on all architectures and for forward on gfx11 and gfx12.
* Fixed fp32-to-bf16 conversion in CK Tile when `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT` selects the inline-assembly round-to-nearest-even mode, where the compiler could overwrite the rounding bias and truncate results.
* Fixed intermittent wrong token counts and expert ranges from the CK Tile MoE sorting multi-phase kernel on gfx11, where the workgroup barriers around the expert-mesh workspace did not order the global stores between wavefronts.
* Fixed a performance regression in XDL grouped convolution forward and other operations built on `GridwiseGemmMultipleD_xdl_cshuffle` without D tensors, by restoring the LDS-shuffled output epilogue.
* Fixed CK build failures with Clang when glibc source fortification (`_FORTIFY_SOURCE`) is enabled, and with newer compilers that declare the AMDGPU builtins with `ext_vector_type` signatures, reject `__fp16` vector types, or report the experimental lifetime-safety diagnostics as errors.
* Fixed compilation errors on gfx1250 for CK Tile GEMMs that select the fp32 16x16x4 or the fp16 and bf16 16x16x32 WMMA instructions with fp32 accumulation, such as the bf16 grouped GEMM that PyTorch builds.
* Fixed build failures of the CK Tile FMHA backward instances for gfx1250 with `--offload-new-driver` by shortening the generated instance names, for example `_srv` instead of `_storerandval` and `_tl` instead of `_trload`.
* Fixed the CK Tile dispatcher heuristics data pipeline refusing `--capture_hw` on gfx950 and gfx1250 nodes and misparsing kernel names whose pipeline name contains an underscore, such as `comp_tdm_v2`.
* Fixed the `ck-template-driver` executable not being installed when CK is built with `CK_USE_CODEGEN=ON`.


## Composable Kernel 1.3.0 for ROCm 10.2.0

### Added

* Added a CI path-length check that rejects newly added or renamed files whose repository-relative path exceeds 200 characters, keeping the absolute path under the Windows `MAX_PATH` limit of 260.
* Added bf16, fp32, fp8, and bf8 data types and the rrr, crr, and ccr layouts to the CK Tile Tile Engine batched GEMM and its dispatcher bridge.

### Optimized

* Improved FMHA backward performance on gfx1250 by moving the operand transfers to TDM and the transposed reads to ds_load_tr, keeping the dK and dV accumulators in registers, pairing mirrored tiles under a causal mask, and coalescing the dQ atomic onto a single cache line.

## Composable Kernel 1.3.0 for ROCm 10.1.0

### Added

* Added support for building Composable Kernel for the target-agnostic SPIR-V target (`amdgcnspirv`), which is compiled to native code at run time.
* Added grouped convolution forward, backward data, and backward weight instances for gfx1250.
* Added a wavelet GEMM pipeline for convolution forward that specializes waves into separate load and math roles to reduce VALU contention.
* Added a batched contraction kernel with multiple ABD support to CK Tile.
* Added CK Tile dispatcher support for more GEMM operators, including weight preshuffle GEMM, batched GEMM, batched contraction, grouped A-quantized and AB-quantized GEMM, grouped row-column and tensor quantized GEMM, and block scale quantized GEMM.
* Added gfx1250 support to the CK Tile dispatcher GEMM operators, covering universal, grouped, multiple D, multiple ABD, batched, batched contraction, and grouped quantized GEMM.
* Added gfx1250 WMMA instance support to the CK backend for PyTorch Inductor, covering universal GEMM, batched GEMM, and grouped convolution forward.
* Added the tanh approximation of GELU to the XDL two-stage MoE GEMM epilogue.
* Added double-precision buffer atomic add support on gfx1250.

### Optimized

* Improved FMHA forward performance for head dimension 128 on gfx11 and gfx12 targets by retuning tile selection.
* Improved FMHA forward performance on gfx1250 for bf16 and fp16 head dimension 128 by padding the LDS layout in the qr_tdm pipeline.
* Improved memory coalescing of microscaling (MX) scale loads on gfx1250 by unifying the scale16 layout with scale32.

### Changed

* Disabled the large tensor XDL grouped convolution backward weight instances on gfx1250, where they could produce intermittent memory access faults.
* Changed the ck4inductor instance enumerators to emit a deduplicated, deterministically ordered instance list.

### Fixed

* Fixed a 32-bit integer overflow in the tensor descriptor element space size that caused undersized workspace allocation and out-of-bounds writes in grouped convolution backward weight for large tensors.
* Fixed grouped convolution forward rejecting valid problem sizes because the implicit GEMM view was subject to the 2 GB logical GEMM size limit.
* Fixed incorrect results in grouped convolution backward data and XDL GEMM kernels caused by an invalid `__restrict__` qualifier on LDS pointers.
* Fixed incorrect accumulation in atomic and split-K kernels on gfx1250 by issuing buffer atomic adds with device scope coherence.
* Fixed memory faults in FMHA batch prefill with a paged KV cache when the page size is a single token.
* Fixed the softmax sink gradient shape in the FMHA backward kernel and corrected sliding window progression when the attention sink is enabled.
* Fixed incorrect results in microscaling (MX) XDL GEMM and B preshuffle kernels when the B operand is more densely packed than A, such as FP8 by FP4.
* Fixed intermittent incorrect results in microscaling (MX) GEMM on gfx1250 caused by missing LDS read ordering in the double-buffered pipeline.
* Fixed incorrect results in FP8 block scale weight preshuffle GEMM and MoE GEMM on gfx1250 caused by an invalid preshuffle layout and a hardcoded wave size.
* Fixed incorrect results in FP4 weight preshuffle quantized GEMM caused by mismatched packed element granularity between the host B preshuffle and the device B tile distribution.
* Fixed incorrect results in the B preshuffle dequantized GEMM pipeline caused by the C transpose setting being ignored.
* Fixed incorrect results from the CK Tile compute v3 intrawave GEMM pipeline with eight-warp block arrangements on gfx1250.
* Fixed out-of-bounds asynchronous loads in the CK Tile compute async WMMA GEMM pipeline caused by a missing element validity check.
* Fixed accuracy loss in the bf16 fast GELU element-wise operation on gfx1250 by using the non-native implementation.
* Fixed a build failure caused by the compiler change from vector to ext_vector types.
* Fixed a CMake failure when building instance libraries for multiple offload targets.
* Fixed an unspecified minimum blocks per compute unit value being passed to the compiler.
* Fixed the CK Tile dispatcher code generator and its ctypes bindings failing to build.

## Composable Kernel 1.2.0 for ROCm 10.0

### Added

* Added multiple D (bias) and large tensor support to the CK Tile quantized GEMM kernel for row-column quantization.

### Changed

* Improved performance of row-column quantized a8w8 GEMM through better instruction scheduling in the eight-waves pipeline, wider epilogue stores, and nontemporal C/D memory access.

## Composable Kernel 1.2.0 for ROCm 7.13

### Added

* Added overload of load_tile_transpose that takes reference to output tensor as output parameter
* Use data type from LDS tensor view when determining tile distribution for transpose in the GEMM pipeline
* Added eightwarps support for abquant mode in blockscale GEMM.
* Added preshuffleB support for abquant mode in blockscale GEMM.
* Added support for explicit GEMM in CK_TILE grouped convolution forward and backward weight.
* Added TF32 convolution support on gfx942 and gfx950 in CK. It could be enabled/disabled via `DTYPES` of "tf32".
* Added streamingllm sink support for FMHA FWD, include qr_ks_vs, qr_async and splitkv pipelines.
* Added support for microscaling (MX) FP8/FP4 mixed data types to Flatmm pipeline.
* Added support for fp8 dynamic tensor-wise quantization of fp8 fmha fwd kernel.
* Added FP8 KV cache support for FMHA batch prefill.
* Added support for gfx1153 target.
* Added FMHA batch prefill kernel support for several KV cache layouts, flexible page sizes, and different lookup table configurations.
* Added gpt-oss sink support for FMHA FWD, include qr_ks_vs, qr_async, qr_async_trload and splitkv pipelines.
* Added persistent async input scheduler for CK Tile universal GEMM kernels to support asynchronous input streaming.
* Added FP8 block scale quantization for FMHA forward kernel.
* Added gfx11 support for FMHA.
* Added microscaling (MX) FP8/FP4 support on gfx950 for FMHA forward kernel ("qr" pipeline only).
* Added FP8 per-tensor quantization support for FMHA forward V3 pipeline on gfx950.

### Changed

### Upcoming changes

## Composable Kernel 1.2.0 for ROCm 7.2.0

### Added
* Added tests for f8 x bf8 on CompV3, and f8 x bf8 with K_BlockSize 32 on CompV4
* Added CK-Tile dispatcher - a unified kernel dispatch, code generation and architecture-based kernel filtering system with with C++ and Python frontends starting with GEMM support.
* Added support for bf16 data type to grouped_gemm and grouped_gemm_preshuffle.
* Added Col-Col-Row-Col layout support for aquant mode in blockscale GEMM.
* Added support for mixed precision fp8 x bf8 universal GEMM and weight preshuffle GEMM.
* Added a compute async pipeline in the CK Tile universal GEMM on gfx950.
* Added support for B Tensor type `pk_int4_t` in the CK Tile weight preshuffle GEMM.
* Added the new api to load different memory sizes to SGPR.
* Added support for B Tensor preshuffle in CK Tile grouped GEMM.
* Added a basic copy kernel example and supporting documentation for new CK Tile developers.
* Added support for grouped GEMM kernels to perform Multi D elementwise operation.
* Added support for multiple ABD GEMM.
* Added benchmarking support for tile engine GEMM Multi D.
* Added block scaling support in CK Tile GEMM, allowing flexible use of quantization matrices from either A or B operands.
* Added the row-wise column-wise quantization for CK Tile GEMM and CK Tile grouped GEMM.
* Added support for f32 to FMHA (forward and backward).
* Added tensor-wise quantization for CK Tile GEMM.
* Added support for batched contraction kernel.
* Added WMMA (gfx12) support for FMHA.
* Added pooling kernel in CK_TILE
* Added top-k sigmoid kernel in CK_TILE
* Added the blockscale 2D support for CK_TILE GEMM.
* Added Flatmm pipeline for microscaling (MX) FP8/FP4 data types
* Added reduce and multi reduction kernels

### Changed

* Removed `BlockSize` in `make_kernel` and `CShuffleEpilogueProblem` to support Wave32 in CK Tile (#2594)
* Added an optional template parameter `Arch` (`gfx9_t`, `gfx12_t` etc.) to `make_kernel` to support linking multiple object files that have the same kernel compiled for different architectures.
* FMHA examples and tests can be built for multiple architectures (gfx9, gfx950, gfx12) at the same time.

### Upcoming changes

* Composable Kernel will be adopting C++20 features in an upcoming ROCm release, updating the minimum compiler requirement to C++20. Ensure that your development environment complies with this requirement to facilitate a seamless transition.

## Composable Kernel 1.1.0 for ROCm 7.1.1

### Upcoming changes

* Composable Kernel will be adopting C++20 features in an upcoming ROCm release, updating the minimum compiler requirement to C++20. Ensure that your development environment complies with this requirement to facilitate a seamless transition.

## Composable Kernel 1.1.0 for ROCm 7.1.0

### Added

* Added support for hdim as a multiple of 32 for FMHA (fwd/fwd_splitkv/bwd)
* Added support for elementwise kernel.

### Upcoming changes

* Non-grouped convolutions are deprecated. Their functionality is supported by grouped convolution.

## Composable Kernel 1.1.0 for ROCm 7.0.0

### Added

* Added support for bf16, f32, and f16 for 2D and 3D NGCHW grouped convolution backward data
* Added a fully asynchronous HOST (CPU) arguments copy flow for CK grouped GEMM kernels.
* Added support GKCYX layout for grouped convolution forward (NGCHW/GKCYX/NGKHW, number of instances in instance factory for NGCHW/GKYXC/NGKHW has been reduced).
* Added support for GKCYX layout for grouped convolution forward (NGCHW/GKCYX/NGKHW).
* Added support for GKCYX layout for grouped convolution backward weight (NGCHW/GKCYX/NGKHW).
* Added support for GKCYX layout for grouped convolution backward data (NGCHW/GKCYX/NGKHW).
* Added support for Stream-K version of mixed fp8/bf16 GEMM
* Added support for Multiple D GEMM
* Added GEMM pipeline for microscaling (MX) FP8/FP6/FP4 data types
* Added support for FP16 2:4 structured sparsity to universal GEMM.
* Added support for Split K for grouped convolution backward data.
* Added logit soft-capping support for fMHA forward kernels.
* Added support for hdim as a multiple of 32 for FMHA (fwd/fwd_splitkv)
* Added benchmarking support for tile engine GEMM.
* Added Ping-pong scheduler support for GEMM operation along the K dimension.
* Added rotating buffer feature for CK_Tile GEMM.
* Added int8 support for CK_TILE GEMM.
* Added CK Tile Epilogue Chainer framework for composable epilogue sequences in GEMM operations

### Optimized

* Optimize the gemm multiply multiply preshuffle & lds bypass with Pack of KGroup and better instruction layout.
* Added Vectorize Transpose optimization for CK Tile
* Added the asynchronous copy for gfx950

### Changed

* Removed support for gfx940 and gfx941 targets (#1944)
* Replaced the raw buffer load/store intrinsics with Clang20 built-ins (#1876)
* DL and DPP kernels are now enabled by default.
* Number of instances in instance factory for grouped convolution forward NGCHW/GKYXC/NGKHW has been reduced.
* Number of instances in instance factory for grouped convolution backward weight NGCHW/GKYXC/NGKHW has been reduced.
* Number of instances in instance factory for grouped convolution backward data NGCHW/GKYXC/NGKHW has been reduced.

## Composable Kernel 1.1.0 for ROCm 6.1.0

### Additions

* Added generic instances for GEMM XDL operations (#1161)
* Added gamma and beta parameters for the layernorm and groupnorm bwd operations (#1133)
* Introduced wrapper sublibrary (limited functionality). (#1071, #1098, #1108, #1126)
* Added an option to vary the number of warm-up cycles and iterations for ckProfiler (#1124)

### Optimizations

* New performance optimizations for GEMM operations on MI200 and MI300 architectures (#1135)

### Fixes

* Reduced the build time for most GPU architectures (#1084)
* Fixed some conversion issues for fp8 data type (#1099)

### Changes

None

### Known issues

None

## Composable Kernel 1.1.0 for ROCm 6.0.0

### Fixes

* Fixed a hazard associated with inline v_dot (#808)
* Fixed two bugs in grouped convolution backward data without K padding (#848 #876)

### Optimizations

None

### Additions

* Added an image to a column kernel (#867)
* Added a column to an image kernel (#930)
* Support for 3D grouped convolution on RDNA 3 GPUs (#935, #950, #985)
* Grouped convolution support for small K and C (#822 #879 #897)
* Support for NHWGC (2D and 3D) grouped convolution backward weight (#769 #804)
* Support for bf16/f32/f16 and NHWGC (2D and 3D) grouped convolution backward data (#757 #799)
* Support for Batched GEMM DL (#732)

### Changes

* Changed the grouped convolution API to maintain consistency with other convolution kernels (#817)

## Composable Kernel 0.2.0 for ROCm 5.7.0

### Fixes

* Fixed a bug in 6-dimensional kernels (#555)
* Fixed a test case failure with grouped convolution backward weight (#524)

### Optimizations

* Improved the performance of the normalization kernel

### Additions

* New CMake flags:
  * "DL_KERNELS"-* Must be set to "ON" in order to build the GEMM DL and batched_gemm_multi_d_dl instances
  * "DTYPES" -- Can be set to any subset of "fp64;fp32;fp16;fp8;bf16;int8" to build an instance of the specified data types
  * "INSTANCES_ONLY" -- Only builds CK library and instances without tests, examples, or profiler
* New feature: if GPU_TARGETS is not set in the CMake command line, CK will be built for all targets supported by the compiler
* Support for MI300A/MI300X
* Support for AMD RDNA 3
* New user tutorial (#563)
* Additional instances for irregular GEMM sizes (#560)
* New inter-wave consumer-producer programming model for GEMM kernels (#310)
* GEMM with support multiple elementwise fusions (multi-D) (#534)
* Multi-embeddings support (#542)
* AMD RDNA 3 blockwise GEMM and real GEMM support (#541)
* AMD RDNA grouped convolution backward weight support (#505)
* MaxPool and AvgPool forward (#815); MaxPool backward (#750)

### Changes

None
