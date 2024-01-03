# Changelog for Composable Kernel

Full documentation for Composable Kernel is not yet available.

## (Unreleased) CK for ROCm 6.0.0

### Fixes
 - Fixed a hazard associated with inline v_dot (#808)
 - Fixed two bugs in grouped convolution backward data without K padding (#848 #876)

### Optimizations
None

### Additions
- Added an image to a column kernel (#867)
- Added a column to an image kernel (#930)
- Support for 3D grouped convolution on RDNA 3 GPUs (#935, #950, #985)
- Grouped convolution support for small K and C (#822 #879 #897)
- Support for NHWGC (2D and 3D) grouped convolution backward weight (#769 #804)
- Support for bf16/f32/f16 and NHWGC (2D and 3D) grouped convolution backward data (#757 #799)
- Support for Batched Gemm DL (#732)
- Introduce wrapper sublibrary (limited functionality). (#1071, #1098, #1108)

### Changes
 - Changed the grouped convolution API to maintain consistency with other convolution kernels (#817)

## CK 0.2.0 for ROCm 5.7.0

### Fixes
- Fixed a bug in 6-dimensional kernels (#555)
- Fixed a test case failure with grouped convolution backward weight (#524)

### Optimizations
- Improved the performance of the normalization kernel

### Additions
- New CMake flags:
  - "DL_KERNELS"-- Must be set to "ON" in order to build the gemm_dl and batched_gemm_multi_d_dl instances
  - "DTYPES" -- Can be set to any subset of "fp64;fp32;fp16;fp8;bf16;int8" to build an instance of the specified data types
  - "INSTANCES_ONLY" -- Only builds CK library and instances without tests, examples, or profiler
- New feature: if GPU_TARGETS is not set in the CMake command line, CK will be built for all targets supported by the compiler
- Support for MI300A/MI300X
- Support for AMD RDNA 3
- New user tutorial (#563)
- Additional instances for irregular GEMM sizes (#560)
- New inter-wave consumer-producer programming model for GEMM kernels (#310)
- GEMM with support multiple elementwise fusions (multi-D) (#534)
- Multi-embeddings support (#542)
- AMD RDNA 3 blockwise GEMM and real GEMM support (#541)
- AMD RDNA grouped convolution backward weight support (#505)
- MaxPool and AvgPool forward (#815); MaxPool backward (#750)

### Changes
None
