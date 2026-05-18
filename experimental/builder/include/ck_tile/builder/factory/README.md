# Convolution Builder Factory Directory

This directory implements compile-time dispatch from high-level signature and algorithm descriptors to our existing specialized convolution kernel implementations.

See the [main builder documentation](../README.md) for an overview.

## Design Overview

The factory system operates in two phases:

1. **Algorithm Classification**: Predicate concepts in `conv_dispatcher.hpp` inspect the algorithm descriptor to determine which kernel variant it satisfies. The predicates are evaluated in a specific order using `if constexpr`:

   - **Cross-direction** (checked first, supports all convolution directions):
     - `ReferenceAlgorithm` — simple reference implementation for validation
     - `TileAlgorithm` — CK Tile backend, dispatches via `ConvTileFactory`

   - **Forward direction** (old CK):
     - `FwdXdlV3Algorithm` — newer XDL pipeline using block GEMM structure
     - `FwdXdlAlgorithm` — standard XDL using AMD XDLops instructions
     - `FwdWmmaAlgorithm` — WMMA variant for gfx11/gfx12 hardware
     - `FwdDlAlgorithm` — vectorized dot-product kernel (non-XDLops)
     - `LargeTensorAlgorithm` — XDL with extended tensor support

   - **Backward weight direction** (old CK):
     - `BwdXdlAlgorithm`, `BwdXdlV3Algorithm`, `BwdTwoStageXdlAlgorithm`, `BwdDlAlgorithm`, `BwdMultiDXdlAlgorithm`, `BwdWmmaV3Algorithm`, `BwdTwoStageWmmaV3Algorithm`, `BwdWmmaAlgorithm`, `BwdMultiDWmmaV3Algorithm`

   - **Backward data direction**: Currently supports only Reference and Tile algorithms. Optimized old CK kernels are not yet implemented.

2. **Factory Instantiation**: Each factory transforms builder descriptors into backend-specific template parameters and instantiates the corresponding kernel.

## Key Files

- **`conv_dispatcher.hpp`**: Entry point with `make_conv_instance()` function. Contains dispatch logic and algorithm classification predicates. **Start here** to understand the overall flow.

- **Forward factories** (old CK):
  `conv_fwd_v3_factory.hpp`, `conv_fwd_xdl_factory.hpp`, `conv_fwd_wmma_factory.hpp`, `conv_fwd_dl_factory.hpp`, `conv_fwd_large_tensor_factory.hpp`

- **Backward weight factories** (old CK):
  `conv_bwd_weight_xdl_factory.hpp`, `conv_bwd_weight_xdl_v3_factory.hpp`, `conv_bwd_weight_two_stage_xdl_factory.hpp`, `conv_bwd_weight_dl_factory.hpp`, `conv_bwd_weight_multi_d_xdl_factory.hpp`, `conv_bwd_weight_wmma_v3_factory.hpp`, `conv_bwd_weight_two_stage_wmma_v3_factory.hpp`, `conv_bwd_weight_wmma_factory.hpp`, `conv_bwd_weight_multi_d_wmma_v3_factory.hpp`

- **Cross-direction factories**:
  `reference_factory.hpp` (reference implementation), `conv_tile_factory.hpp` (CK Tile backend)

- **`helpers/`**: Transformation utilities that map builder types to backend-specific parameters. Organized into `helpers/ck/` (old CK mappings) and `helpers/ck_tile/` (CK Tile mappings).

## Usage

```cpp
#include "ck_tile/builder/factory/conv_dispatcher.hpp"

// Uses latest version by default (currently "0.1.0")
auto kernel = make_conv_instance<SIGNATURE, ALGORITHM>();

// Or pin to a specific version
auto kernel_v0 = make_conv_instance<SIGNATURE, ALGORITHM, "0.0.0">();
```

The dispatcher automatically selects the appropriate factory at compile time.

## Factory Architecture and the Unification Gap

Each factory is a self-contained facade: it accepts builder descriptors and produces a kernel instance, but it does so with its own algorithm descriptor shape and its own parameter mapping logic. The 16+ factories share no common infrastructure for parameter transformation.

**Old CK factories** (e.g., `ConvFwdXdlV3Factory`) flatten all algorithm parameters into a single device operation template instantiation with approximately 49 template arguments. The factory's primary job is mapping builder enum values (layouts, data types, elementwise ops) to CK's internal types. Within old CK, the XDL and WMMA factories duplicate much of this mapping logic despite sharing the same underlying parameter concepts.

**The CK Tile factory** (`ConvTileFactory`) composes modern objects — a traits type, a tile partitioner, a GEMM pipeline, and an epilogue pipeline — each with its own configuration. This results in approximately 31 parameters distributed across four composed types rather than one flat template.

Both factory paths produce a kernel `Instance` type that satisfies the same usage interface (construction, argument setup, invocation). The dispatcher abstracts this difference from the caller. However, the algorithm descriptor accepted by each factory is different — the unification burden currently falls on the caller (MIOpen), not the dispatcher. Collapsing these per-variant descriptors into a single algorithm format that the dispatcher decomposes internally is the key step toward making the builder a true unified facade.
