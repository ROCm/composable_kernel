# Change Log for Composable Kernel

Full documentation for Composable Kernel is not yet available.

## CK 0.2.0 for ROCm 5.5.0

### Fixed
- Fixed a bug in 6-dimensional kernels (#555).
- Fixed grouped ConvBwdWeight test case failure (#524).

### Optimizations
- Improve proformance of normalization kernel

### Added
- Added support on NAVI3x.
- Added user tutorial (#563).
- Added more instances for irregular GEMM sizes (#560).
- Added inter-wave consumer-producer programming model for GEMM kernels (#310).
- Added multi-D GEMM client APIs (#534).
- Added multi-embeddings support (#542).
- Added Navi3x blockwise GEMM and real GEMM support (#541).
- Added Navi grouped ConvBwdWeight support (#505).

### Changed
- Changed ...
