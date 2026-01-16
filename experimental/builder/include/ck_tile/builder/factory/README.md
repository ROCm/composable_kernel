# Convolution Builder Factory Directory

This directory implements compile-time dispatch from high-level signature algorithm descriptors to our exisitng specialized convolution kernel implementations.

See the [main builder documentation](../README.md) for an overview.

## Design Overview

The factory system operates in two phases:

1. **Algorithm Classification**: The function `make_conv_instance` in  `conv_dispatcher.hpp` inspects the signature and algorithm descriptors to determine which kernel variant they satisfy (XDL V3, XDL, WMMA, DL, or Large Tensor)

2. **Factory Instantiation**: Each factory (`conv_fwd_*_factory.hpp`) transforms builder descriptors into CK device operation template parameters and instantiates the corresponding kernel device operation.

## Key Files

- **`conv_dispatcher.hpp`**: Entry point with `make_conv_instance()` function. Contains dispatch logic and algorithm classification predicates. **Start here** to understand the overall flow.

- **`conv_fwd_*_factory.hpp`**: Individual factories for each kernel variant. Each extracts configuration from descriptors, validates parameters, and instantiates the underlying CK device operation.

- **`helpers/`**: Transformation utilities that map builder types to CK device operation parameters (layouts, data types, elementwise ops, block configurations, etc.)

## Usage

```cpp
#include "ck_tile/builder/factory/conv_dispatcher.hpp"

using Factory = decltype(make_conv_instance<signature, algorithm, "v1">());
```

The dispatcher automatically selects the appropriate factory following explicit logic.

# Convolution Algorithm Hierarchy

This section illustrates the hierarchy of convolution algorithm concepts defined in `conv_algorithms.hpp`.

## Overview

The convolution algorithms are organized into three main categories:

1. **XDL Algorithms** - GPU matrix multiplication using XDL (matrix core instructions)
2. **WMMA Algorithms** - GPU matrix multiplication using WMMA (Wave Matrix Multiply-Accumulate)
3. **DL Algorithms** - Special vectorized dot-product kernels optimized for specific data layouts with separate implementation.

XDL and WMMA algorithms share a common base, while DL algorithms have their own independent base.

## Common Base Hierarchy (XDL & WMMA)

Both XDL and WMMA algorithms share the following foundational concepts:

```
ConvWarpGemmAlgorithm (Base Concept)
│
│  Requirements:
│  • ConvAlgorithmDescriptor
│  • SpecifiesThreadBlock
│  • SpecifiesTileTransferParameters (ThreadClusters, LdsTransfer, AccessOrders)
│  • SpecifiesWarpGemm
│
├─── FwdAlgorithm (Forward Convolution)
│    │
│    │  Additional: SpecifiesFwdConvSpecialization
│    │
│    └─── FwdAlgorithmV3
│         │
│         │  Additional: SpecifiesPipelineV3 + SpecifiesGemmPipeline
│         │
│
└─── BwdAlgorithm (Backward Weight Convolution)
     │
     │  Additional: SpecifiesBwdWeightConvSpecialization
     │
     └─── BwdAlgorithmV3
          │
          │  Additional: SpecifiesPipelineV3 + SpecifiesGemmPipeline
          │
```

---

## XDL Algorithm Hierarchy

### Forward XDL Algorithms

```
FwdAlgorithm + SpecifiesXdl
│
├─── FwdXdlAlgorithmBase
     │
     ├─── FwdXdlAlgorithm
     │    │
     │    └─ Requirements: Base + SpecifiesGenericInstance
     │
     ├─── LargeTensorAlgorithm
     │    │
     │    └─ Requirements: Base + SpecifiesLargeTensorSupport
     │
     └─── FwdXdlV3Algorithm
          │
          └─ Based on: FwdAlgorithmV3 + SpecifiesXdl
```

### Backward XDL Algorithms

```
BwdAlgorithm + SpecifiesXdl
│
├─── BwdXdlAlgorithmBase (ThreadClusterRank=4)
│    │
│    ├─── BwdXdlAlgorithm
│    │    │
│    │    └─ Requirements: Base + SpecifiesTransposeTransfer + SpecifiesGenericInstance
│    │
│    └─── BwdMultiDXdlAlgorithm
│         │
│         └─ Requirements: Base + SpecifiesMultipleDSupport
│
└─── BwdXdlV3AlgorithmBase
     │
     ├─── BwdXdlV3Algorithm
     │    │
     │    └─ Requirements: Base
     │
     └─── BwdTwoStageXdlAlgorithm
          │
          └─ Requirements: Base + SpecifiesTransposeTransfer + SpecifiesTwoStageSupport
```

**Valid XDL Algorithms:**
- FwdXdlAlgorithm
- FwdXdlV3Algorithm
- LargeTensorAlgorithm
- BwdXdlAlgorithm
- BwdXdlV3Algorithm
- BwdTwoStageXdlAlgorithm
- BwdMultiDXdlAlgorithm

---

## WMMA Algorithm Hierarchy

### Forward WMMA Algorithms

```
FwdAlgorithm + SpecifiesWmma
│
└─── FwdWmmaAlgorithm
     │
     └─ Requirements: Base  + SpecifiesWmma
```

### Backward WMMA Algorithms

```
BwdAlgorithm + SpecifiesWmma
│
├─── BwdWmmaAlgorithmBase (ThreadClusterRank=3)
│    │
│    └─── BwdWmmaAlgorithm
│         │
│         └─ Requirements: Base + SpecifiesGemmPipeline + SpecifiesGenericInstance
│
└─── BwdWmmaV3AlgorithmBase (Based on BwdAlgorithmV3)
     │
     ├─── BwdMultiDWmmaV3Algorithm
     │    │
     │    └─ Requirements: Base + SpecifiesMultipleDSupport
     │
     ├─── BwdWmmaV3Algorithm
     │    │
     │    └─ Requirements: Base + SpecifiesTransposeTransfer
     │
     └─── BwdTwoStageWmmaV3Algorithm
          │
          └─ Requirements: Base + SpecifiesTransposeTransfer + SpecifiesTwoStageSupport
```

**Valid WMMA Algorithms:**
- FwdWmmaAlgorithm
- BwdWmmaAlgorithm
- BwdWmmaV3Algorithm
- BwdTwoStageWmmaV3Algorithm
- BwdMultiDWmmaV3Algorithm

---

## DL Algorithm Hierarchy

DL algorithms have a separate base and do not share the common hierarchy with XDL/WMMA algorithms.

```
DlAlgorithm
│
│  Requirements:
│  • ConvAlgorithmDescriptor
│  • SpecifiesThreadBlock
│  • SpecifiesDlThreadConfig
│  • SpecifiesDlThreadCluster
│  • SpecifiesDlEpilogue
│
├─── FwdDlAlgorithmBase
│    │
│    │  Requirements: Base + SpecifiesFwdConvSpecialization + SpecifiesDlFwdBlockTransfer + SpecifiesGemmSpecialization
│    │
│    └─── FwdDlAlgorithm
│
└─── BwdDlAlgorithm
     │
     └─ Requirements: Base + SpecifiesBwdWeightConvSpecialization + SpecifiesDlBwdBlockTransfer
```

**Valid DL Algorithms:**
- FwdDlAlgorithm
- BwdDlAlgorithm

---


## Reference Algorithms

```
ReferenceAlgorithm
│
└─ Requirements: ConvAlgorithmDescriptor
                + SpecifiesReferenceAlgorithm
```

Used for reference implementations and testing.

## CK Tile Algorithms

```
TileAlgorithm
│
└─ Requirements: ConvAlgorithmDescriptor
                + SpecifiesTileThreadBlock
                + SpecifiesTileTransfer
                + SpecifiesTileConvSpecialization
                + SpecifiesTileBlockGemm
                + SpecifiesTileOptimizations
```

The CK Tile algorithms are applicable to foward convolution as well as backwards convolution (weight and data).

---

## Summary for XDL/WMMA/DL algorithms

| Category | Algorithm Type | Forward Variants | Backward Variants |
|----------|---------------|------------------|-------------------|
| **XDL** | Base | FwdXdlAlgorithmBase | BwdXdlAlgorithmBase, BwdXdlV3AlgorithmBase |
| | Concrete | • FwdXdlAlgorithm<br>• FwdXdlV3Algorithm<br>• LargeTensorAlgorithm | • BwdXdlAlgorithm<br>• BwdXdlV3Algorithm<br>• BwdTwoStageXdlAlgorithm<br>• BwdMultiDXdlAlgorithm |
| **WMMA** | Base | FwdAlgorithm | BwdWmmaAlgorithmBase, BwdWmmaV3AlgorithmBase |
| | Concrete | • FwdWmmaAlgorithm | • BwdWmmaAlgorithm<br>• BwdWmmaV3Algorithm<br>• BwdTwoStageWmmaV3Algorithm<br>• BwdMultiDWmmaV3Algorithm |
| **DL** | Base | FwdDlAlgorithmBase | DlAlgorithm |
| | Concrete | • FwdDlAlgorithm | • BwdDlAlgorithm |

---
