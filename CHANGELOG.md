# Changelog for Composable Kernel

Documentation for Composable Kernel available at [https://rocm.docs.amd.com/projects/composable_kernel/en/latest/](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/).

## (Unreleased) Composable Kernel for ROCm

### Added 

* Added a compute async pipeline in the CK TILE universal GEMM on gfx950
* Added support for B Tensor type pk_int4_t in the CK TILE weight preshuffle GEMM.
* Added the new api to load different memory sizes to SGPR.
* Added support for B Tensor Preshuffle in CK TILE Grouped GEMM.
* Added a basic copy kernel example and supporting documentation for new CK Tile developers.
* Added support for grouped_gemm kernels to perform multi_d elementwise operation.
* Added support for Multiple ABD GEMM
* Added benchmarking support for tile engine GEMM Multi D.
* Added block scaling support in CK_TILE GEMM, allowing flexible use of quantization matrices from either A or B operands.
* Added the row-wise column-wise quantization for CK_TILE GEMM & CK_TILE Grouped GEMM.
* Added support for f32 to FMHA (fwd/bwd).
* Added tensor-wise quantization for CK_TILE GEMM.
* Added support for batched contraction kernel.
* Added pooling kernel in CK_TILE

### Changed

* Removed `BlockSize` in `make_kernel` and `CShuffleEpilogueProblem` to support Wave32 in CK_TILE (#2594)

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
