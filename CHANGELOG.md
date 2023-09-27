# Changelog for Composable Kernel

Full documentation for Composable Kernel is not yet available.

## (Unreleased) CK for ROCm 6.0.0

### Fixes
 - Fixed a hazard associated with inline v_dot (#808).
 - Fixed a bug in grouped convolution backward data without K padding (#848 #876).

### Optimizations
None

### Additions
- Added an image to a column kernel (#867).
- Added a column to an image kernel (#930).
- Added support for 3D grouped convolution forward on RDNA 3 GPUs (#935).
- Added grouped convolution support for small K and C (#822 #879 #897).
- Added support for NHWGC (2D and 3D) grouped convolution backward weight (#769 #804).
- Added support for bf16/f32/f16 and NHWGC (2D and 3d) grouped convolution backward data (#757 #799).
- Added support for Batched Gemm DL (#732).

### Changes
 - Changed the grouped convolution API to maintain consistency with other convolution kernels (#817).

## CK 0.2.0 for ROCm 5.7.0

### Fixes
- Fixed a bug in 6-dimensional kernels (#555).
- Fixed a test case failure with grouped ConvBwdWeight (#524).

### Optimizations
- Improved the performance of the normalization kernel.

### Additions
- Added new CMake flags:
  - "DL_KERNELS"-- Must be set to "ON" in order to build the gemm_dl and batched_gemm_multi_d_dl instances
  - "DTYPES" -- Can be set to any subset of "fp64;fp32;fp16;fp8;bf16;int8" to build an instance of the specified data types
  - "INSTANCES_ONLY" -- Only builds CK library and instances without tests, examples, or profiler
- Added a new feature: if GPU_TARGETS is not set in the CMake command line, CK will be built for all targets supported by the compiler.
- Added support for MI300A/MI300X.
- Added support for AMD RDNA 3.
- Added a user tutorial (#563).
- Added additional instances for irregular GEMM sizes (#560).
- Added an inter-wave consumer-producer programming model for GEMM kernels (#310).
- Added multi-D GEMM client APIs (#534).
- Added multi-embeddings support (#542).
- Added AMD RDNA 3 blockwise GEMM and real GEMM support (#541).
- Added AMD RDNA-grouped ConvBwdWeight support (#505).
- Added MaxPool, AvgPool forward (#815).
- Added MaxPool backward (#750).

### Changes
None
