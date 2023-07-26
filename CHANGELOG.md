# Change Log for Composable Kernel

Full documentation for Composable Kernel is not yet available.

## CK 0.2.0 for ROCm 5.5.0

### Fixed
- Fixed a bug in 6-dimensional kernels (#555).
- Fixed grouped ConvBwdWeight test case failure (#524).

### Optimizations
- Improve proformance of normalization kernel

### Added
- Added new cmake flag "DL_KERNELS" must be set to "ON" in order to build the gemm_dl and batched_gemm_multi_d_dl instances.
- Added new cmake flag "DTYPES" which could be set to any subset of "fp64;fp32;fp16;fp8;bf16;int8" to build instance of select data types.
- Added new cmake flag "INSTANCES_ONLY" which will only build CK library and instances without the tests, examples, or profiler.
- Added new feature: if GPU_TARGETS is not set on cmake command line, CK will be built for all targets supported by compiler.
- Added support on MI300A/MI300X.
- Added support on NAVI3x.
- Added user tutorial (#563).
- Added more instances for irregular GEMM sizes (#560).
- Added inter-wave consumer-producer programming model for GEMM kernels (#310).
- Added multi-D GEMM client APIs (#534).
- Added multi-embeddings support (#542).
- Added Navi3x blockwise GEMM and real GEMM support (#541).
- Added Navi grouped ConvBwdWeight support (#505).
- Added pool3d forward (#697).
- Added maxpool backward (#750).

### Changed
- Changed ...
