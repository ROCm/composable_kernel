# Grouped Convolution Backward Algorithms Summary

This document provides a comprehensive overview of the backward convolution algorithms exposed by the Composable Kernel library for **grouped convolutions**.

## Overview

The library provides optimized GPU kernels for two types of backward convolution operations:
1. **Backward Data** (gradient with respect to input)
2. **Backward Weight** (gradient with respect to weights)

All algorithms are part of the static library and have pre-compiled instances.

## 1. Backward Data Convolution Algorithms

### 1.1 DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp`

**Description:** XDL (Matrix Core) based backward data convolution with CShuffle optimization.

**Key Features:**
- Uses AMD Matrix Core Instructions (XDL/MFMA)
- CShuffle for efficient data movement
- Supports multiple spatial dimensions (1D, 2D, 3D)
- Multiple data types: FP16, BF16, FP32, TF32, FP8, BF8
- Two specializations:
  - `ConvBwdDataDefault`: General convolution
  - `ConvBwdDataFilter1x1Stride1Pad0`: Optimized for 1x1 filters with stride 1 and no padding

**Instance Files:**
- `device_grouped_conv_bwd_data_xdl_instance.hpp` - Main XDL instances (FP16, BF16, FP32, TF32)
- `device_grouped_conv_bwd_data_xdl_bilinear_instance.hpp` - Bilinear variants
- `device_grouped_conv_bwd_data_xdl_scale_instance.hpp` - Scale variants
- `device_grouped_conv_bwd_data_transpose_xdl_instance.hpp` - Transpose variants

**Instantiation Sources:**
- `grouped_conv2d_bwd_data/xdl/*.cpp` - 2D convolution instances
- `grouped_conv3d_bwd_data/xdl/*.cpp` - 3D convolution instances

### 1.2 DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle.hpp`

**Description:** WMMA (Wave Matrix Multiply Accumulate) based backward data convolution.

**Key Features:**
- Uses WMMA instructions (16x16 matrix operations)
- More flexible for different block sizes
- Supports FP16 and INT8 data types
- Optimized for specific GPU architectures

**Instance Files:**
- `device_grouped_conv_bwd_data_wmma_f16_instance.hpp` - FP16 WMMA instances
- `device_grouped_conv_bwd_data_wmma_i8_instance.hpp` - INT8 WMMA instances

**Instantiation Sources:**
- `grouped_conv2d_bwd_data/wmma/*.cpp` - 2D convolution instances
- `grouped_conv3d_bwd_data/wmma/*.cpp` - 3D convolution instances

## 2. Backward Weight Convolution Algorithms

### 2.1 DeviceGroupedConvBwdWeight_Xdl_CShuffleV3

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp`

**Description:** Latest XDL-based backward weight convolution, version 3 with advanced optimizations.

**Status:** ✅ **Active - Part of static library** (Recommended for new code)

**Key Features:**
- Latest XDL implementation with CShuffle
- Support for split-K optimization with auto-deduction
- Multiple pipeline versions (v1, v2, v3, v4)
- Block GEMM pipeline schedulers (Intrawave, Interwave)
- Dual LDS buffer support (v4 pipeline)
- Data types: FP16, BF16, FP32, TF32
- Two specializations:
  - `ConvBwdWeightDefault`: General convolution
  - `ConvBwdWeightFilter1x1Stride1Pad0`: Optimized for 1x1 filters

**Instance Files:**
- `device_grouped_conv_bwd_weight_v3_xdl_instance.hpp`

**Instantiation Sources (2D):**
- `grouped_conv2d_bwd_weight/xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_v3_xdl_gnhwc_gkyxc_gnhwk_*.cpp`
- `grouped_conv2d_bwd_weight/xdl/nhwgc_gkyxc_nhwgk/device_grouped_conv2d_bwd_weight_v3_xdl_nhwgc_gkyxc_nhwgk_*.cpp`

### 2.2 DeviceGroupedConvBwdWeight_Wmma_CShuffleV3

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp`

**Description:** Latest WMMA-based backward weight convolution, version 3.

**Status:** ✅ **Active - Part of static library** (Recommended for WMMA)

**Key Features:**
- WMMA 16x16 matrix operations
- CShuffle optimization
- Pipeline schedulers and versions
- Data types: FP16, BF16

**Instance Files:**
- `device_grouped_conv_bwd_weight_v3_wmma_instance.hpp`

**Instantiation Sources (2D):**
- `grouped_conv2d_bwd_weight/wmma/nhwgc_gkyxc_nhwgk/device_grouped_conv2d_bwd_weight_wmma_nhwgc_gkyxc_nhwgk_*.cpp`

### 2.3 DeviceGroupedConvBwdWeight_Xdl_CShuffle

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp`

**Description:** Original XDL-based backward weight convolution (version 1).

**Status:** ✅ **Active - Part of static library** (Legacy, but still maintained)

**Key Features:**
- XDL/MFMA matrix core operations
- CShuffle optimization
- Data types: FP16, BF16, FP32, TF32, FP8/BF8
- Supports transpose operations for NCHW layouts
- Two specializations (Default, Filter1x1Stride1Pad0)

**Instance Files:**
- `device_grouped_conv_bwd_weight_xdl_instance.hpp`

**Instantiation Sources (2D):**
- `grouped_conv2d_bwd_weight/xdl/gnhwc_gkyxc_gnhwk/device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_*.cpp`
- `grouped_conv2d_bwd_weight/xdl/nhwgc_gkyxc_nhwgk/device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_*.cpp`
- `grouped_conv2d_bwd_weight/xdl/ngchw_gkcyx_ngkhw/device_grouped_conv2d_bwd_weight_xdl_ngchw_gkcyx_ngkhw_*.cpp`
- `grouped_conv2d_bwd_weight/xdl/ngchw_gkyxc_ngkhw/device_grouped_conv2d_bwd_weight_xdl_ngchw_gkyxc_ngkhw_*.cpp`

**Instantiation Sources (1D & 3D):**
- `grouped_conv1d_bwd_weight/xdl/*.cpp`
- `grouped_conv3d_bwd_weight/xdl/*.cpp`

### 2.4 DeviceGroupedConvBwdWeight_Wmma_CShuffle

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle.hpp`

**Description:** Original WMMA-based backward weight convolution for 3D convolutions only.

**Status:** ✅ **Active - Part of static library** (3D-specific)

**Key Features:**
- WMMA 16x16 matrix operations
- CShuffle optimization
- **Specialized for 3D convolutions only**
- Data types: FP16, BF16, FP32
- Two specializations (Default, Filter1x1Stride1Pad0)

**Supported Layouts (3D only):**
- NDHWGC/GKZYXC/NDHWGK
- GNDHWC/GKZYXC/GNDHWK

**Note:** This algorithm is specific to 3D convolutions and uses different template parameter structure than other WMMA variants.

### 2.5 DeviceGroupedConvBwdWeight_TwoStage_Xdl_CShuffle

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`

**Description:** Two-stage XDL-based backward weight for large convolutions with memory constraints.

**Status:** ✅ **Active - Part of static library**

**Key Features:**
- Two-stage computation with intermediate workspace
- Better memory efficiency for large problems
- XDL matrix core operations
- Multiple pipeline versions (v1, v2, v5)
- Group merging optimization (NumGroupsToMerge parameter)
- Data types: FP16, BF16
- Supports irregular MPerBlock/NPerBlock configurations

**Instance Files:**
- `device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp`

**Instantiation Sources (2D):**
- `grouped_conv2d_bwd_weight/xdl/nhwgc_gkyxc_nhwgk/device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_*_pipev*.cpp`
- `grouped_conv2d_bwd_weight/xdl/ngchw_gkcyx_ngkhw/device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkcyx_ngkhw_*_pipev*.cpp`
- `grouped_conv2d_bwd_weight/xdl/ngchw_gkyxc_ngkhw/device_grouped_conv2d_bwd_weight_two_stage_xdl_ngchw_gkyxc_ngkhw_*_pipev*.cpp`

**Instantiation Sources (3D):**
- `grouped_conv3d_bwd_weight/xdl/*.cpp`

### 2.6 DeviceGroupedConvBwdWeight_TwoStage_Wmma_CShuffleV3

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`

**Description:** Two-stage WMMA-based backward weight convolution.

**Status:** ✅ **Active - Part of static library**

**Key Features:**
- Two-stage computation
- WMMA 16x16 matrix operations
- Pipeline versions
- Group merging optimization
- Data types: FP16, BF16

**Instance Files:**
- `device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp`

**Instantiation Sources (2D):**
- `grouped_conv2d_bwd_weight/wmma/nhwgc_gkyxc_nhwgk/device_grouped_conv2d_bwd_weight_two_stage_wmma_nhwgc_gkyxc_nhwgk_*_pipev*.cpp`

**Instantiation Sources (3D):**
- `grouped_conv3d_bwd_weight/wmma/*.cpp`

### 2.7 DeviceGroupedConvBwdWeight_DL

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp`

**Description:** Direct Load variant using different memory access pattern.

**Status:** ✅ **Active - Part of static library**

**Key Features:**
- Direct load memory access pattern (no shared memory for A/B)
- Suitable for specific problem sizes
- Supports 1D, 2D, and 3D convolutions
- Data types: FP16, BF16/FP32 (mixed precision), FP32
- Two specializations (Default, Filter1x1Stride1Pad0)

**Instance Files:**
- `device_grouped_conv_bwd_weight_dl_instance.hpp`

**Instantiation Sources:**
- `grouped_conv1d_bwd_weight/dl/device_grouped_conv1d_bwd_weight_dl_*.cpp`
- `grouped_conv2d_bwd_weight/dl/device_grouped_conv2d_bwd_weight_dl_*.cpp`
- `grouped_conv3d_bwd_weight/dl/device_grouped_conv3d_bwd_weight_dl_*.cpp`

### 2.8 DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp`

**Description:** XDL-based backward weight with Multiple D tensor support for fused operations.

**Status:** ✅ **Active - Part of static library**

**Key Features:**
- Supports additional input tensors (D tensors) for fused operations
- Fused element-wise operations (Bilinear, Scale)
- XDL matrix core operations
- Data types: FP16, BF16, FP32, TF32, FP8/BF8
- Two specializations (Default, Filter1x1Stride1Pad0)

**Instance Files:**
- `device_grouped_conv_bwd_weight_xdl_bilinear_instance.hpp` - Bilinear fusion
- `device_grouped_conv_bwd_weight_xdl_scale_instance.hpp` - Scale fusion

**Instantiation Sources:**
- `grouped_conv3d_bwd_weight_bilinear/*.cpp` - 3D Bilinear variants
- `grouped_conv3d_bwd_weight_scale/*.cpp` - 3D Scale variants

### 2.9 DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3

**Implementation File:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_multiple_d_wmma_cshuffle_v3.hpp`

**Description:** WMMA-based backward weight with Multiple D tensor support for fused operations.

**Status:** ✅ **Active - Part of static library**

**Key Features:**
- WMMA 16x16 matrix operations
- Supports fused Scale operation
- Data types: FP16, BF16

**Instance Files:**
- `device_grouped_conv_bwd_weight_wmma_scale_instance.hpp`

**Instantiation Sources:**
- `grouped_conv3d_bwd_weight_scale/*.cpp` - 3D Scale variants

## 3. Algorithm Comparison

| Algorithm | Version | Instruction Set | Data Types | Fused Ops | Split-K | Dimensions | Status |
|-----------|---------|----------------|------------|-----------|---------|------------|--------|
| **Backward Data** | | | | | | | |
| DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1 | v1 | XDL/MFMA | FP16, BF16, FP32, TF32, FP8, BF8 | Yes | No | 1D, 2D, 3D | ✅ Active |
| DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle | - | WMMA | FP16, INT8 | No | No | 1D, 2D, 3D | ✅ Active |
| **Backward Weight** | | | | | | | |
| DeviceGroupedConvBwdWeight_Xdl_CShuffleV3 | v3 | XDL/MFMA | FP16, BF16, FP32, TF32 | No | Yes (auto) | 1D, 2D, 3D | ✅ Active (Recommended) |
| DeviceGroupedConvBwdWeight_Wmma_CShuffleV3 | v3 | WMMA | FP16, BF16 | No | No | 1D, 2D, 3D | ✅ Active (Recommended) |
| DeviceGroupedConvBwdWeight_Xdl_CShuffle | v1 | XDL/MFMA | FP16, BF16, FP32, TF32, FP8, BF8 | No | No | 1D, 2D, 3D | ✅ Active (Legacy) |
| DeviceGroupedConvBwdWeight_Wmma_CShuffle | v1 | WMMA | FP16, BF16, FP32 | No | No | 3D only | ✅ Active (3D-specific) |
| DeviceGroupedConvBwdWeight_TwoStage_Xdl_CShuffle | Two-stage | XDL/MFMA | FP16, BF16 | No | No | 1D, 2D, 3D | ✅ Active |
| DeviceGroupedConvBwdWeight_TwoStage_Wmma_CShuffleV3 | Two-stage | WMMA | FP16, BF16 | No | No | 1D, 2D, 3D | ✅ Active |
| DeviceGroupedConvBwdWeight_DL | - | Direct Load | FP16, BF16, FP32 | No | No | 1D, 2D, 3D | ✅ Active |
| DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle | - | XDL/MFMA | FP16, BF16, FP32, TF32, FP8, BF8 | Yes | No | 1D, 2D, 3D | ✅ Active |
| DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3 | v3 | WMMA | FP16, BF16 | Yes (Scale) | No | 3D | ✅ Active |

## 4. Supported Configurations

### Data Types
- **FP16** (half_t): 16-bit floating point
- **BF16** (bhalf_t): Brain float 16
- **FP32** (float): 32-bit floating point
- **TF32** (tf32_t): TensorFloat-32 (compute type)
- **INT8** (int8_t): 8-bit integer (WMMA bwd data only)
- **FP8/BF8**: 8-bit floating point (compute type for newer GPUs)

### Tensor Layouts

**Backward Data:**
- Input: GNHWC, NHWGC, NGCHW (2D), GNDHWC, NDHWGC (3D), GNWC (1D)
- Weight: GKYXC, GKCYX (2D), GKZYXC (3D), GKXC (1D)
- Output: GNHWK, NHWGK, NGKHW (2D), GNDHWK, NDHWGK (3D), GNWK (1D)

**Backward Weight:**
- Input: GNHWC, NHWGC, NGCHW (2D), GNDHWC, NDHWGC (3D), GNWC (1D)
- Weight: GKYXC, GKCYX (2D), GKZYXC (3D), GKXC (1D)
- Output: GNHWK, NHWGK, NGKHW (2D), GNDHWK, NDHWGK (3D), GNWK (1D)

### Spatial Dimensions
- **1D Convolution**: NDimSpatial = 1
- **2D Convolution**: NDimSpatial = 2
- **3D Convolution**: NDimSpatial = 3

### Specializations
1. **Default**: General purpose convolution
2. **Filter1x1Stride1Pad0**: Optimized for 1x1 convolution with stride 1 and no padding

## 5. Instance Organization

Instances are organized by:
- **Spatial dimension** (1D, 2D, 3D)
- **Data type** (FP16, BF16, FP32, INT8, etc.)
- **Tensor layout** combination
- **Specialization** (Default, 1x1S1P0)
- **Element-wise operations** (PassThrough, Bilinear, Scale)

Each instance specifies detailed template parameters including:
- Block sizes (BlockSize, MPerBlock, NPerBlock, K0PerBlock)
- Wave/Warp configurations (MXdlPerWave, NXdlPerWave or MRepeat, NRepeat)
- Thread cluster arrangements
- Vector access patterns
- LDS (Local Data Share) optimizations
- CShuffle parameters

## 6. Key Template Parameters

### Common Parameters (XDL variants)
- **BlockSize**: Total threads per block (64, 128, 256)
- **MPerBlock, NPerBlock**: GEMM tile sizes per block
- **K0PerBlock**: K dimension blocking
- **K1**: Vector width in K dimension (typically 8 for FP16, 4 for FP32)
- **MPerXDL, NPerXDL**: Matrix dimensions per XDL instruction (16x16 or 32x32)
- **MXdlPerWave, NXdlPerWave**: Number of XDL tiles per wave
- **Pipeline Version**: v1, v2, v3, v4, v5 (different prefetch strategies)
- **Pipeline Scheduler**: Intrawave vs Interwave

### Common Parameters (WMMA variants)
- **BlockSize**: Total threads per block (32, 64, 96, 128, 256)
- **MPerBlock, NPerBlock**: GEMM tile sizes
- **K0PerBlock**: K dimension blocking  
- **K1**: Vector width (typically 8 for FP16, 16 for INT8)
- **MPerWmma, NPerWmma**: WMMA tile size (16x16)
- **MRepeat, NRepeat**: Repetition factors

### Two-Stage Specific Parameters
- **NumGroupsToMerge**: Number of groups to merge for better performance

## 7. Performance Considerations

### Choosing the Right Algorithm

**For Backward Data:**
1. **XDL variant**: Best for modern AMD GPUs with Matrix Core support (MI100, MI200, MI300 series)
2. **WMMA variant**: Good for varied problem sizes and broader compatibility
3. **Use FP16/BF16** for best performance on modern hardware

**For Backward Weight:**
1. **V3 variants (XDL or WMMA)**: Recommended for new code, latest optimizations
2. **Two-Stage variants**: Best for very large convolutions with memory constraints
3. **V1 XDL**: Good alternative with broader layout support (including NCHW)
4. **DL variant**: Specific use cases, no shared memory overhead
5. **MultipleD variants**: When you need fused operations (Bilinear, Scale)

### Optimization Features
- **Split-K**: Parallelizes the reduction dimension for better occupancy (V3 XDL only, auto-deduced)
- **CShuffle**: Optimized cross-lane shuffle for data redistribution
- **Pipeline Versions**: Different prefetch strategies to hide memory latency
  - v1: Basic pipeline
  - v2: Enhanced prefetching with tail number support (1-7)
  - v3: Further optimizations
  - v4: Dual LDS buffer support
  - v5: Advanced prefetching
- **LDS Padding**: Avoid bank conflicts in shared memory
- **Two-Stage**: Reduces memory footprint for large problems

## 8. Library Structure

```
library/
├── include/ck/library/tensor_operation_instance/gpu/
│   ├── grouped_conv_bwd_data/
│   │   ├── device_grouped_conv_bwd_data_xdl_instance.hpp
│   │   ├── device_grouped_conv_bwd_data_wmma_f16_instance.hpp
│   │   ├── device_grouped_conv_bwd_data_wmma_i8_instance.hpp
│   │   └── ... (other variants)
│   └── grouped_conv_bwd_weight/
│       ├── device_grouped_conv_bwd_weight_v3_xdl_instance.hpp
│       ├── device_grouped_conv_bwd_weight_v3_wmma_instance.hpp
│       ├── device_grouped_conv_bwd_weight_xdl_instance.hpp
│       ├── device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp
│       ├── device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp
│       ├── device_grouped_conv_bwd_weight_dl_instance.hpp
│       ├── device_grouped_conv_bwd_weight_xdl_bilinear_instance.hpp
│       ├── device_grouped_conv_bwd_weight_xdl_scale_instance.hpp
│       ├── device_grouped_conv_bwd_weight_wmma_scale_instance.hpp
│       └── device_grouped_conv_bwd_weight_wmma_bilinear_instance.hpp
└── src/tensor_operation_instance/gpu/
    ├── grouped_conv1d_bwd_weight/
    ├── grouped_conv2d_bwd_data/
    │   ├── wmma/ (WMMA instances for 2D)
    │   └── xdl/  (XDL instances for 2D)
    ├── grouped_conv2d_bwd_weight/
    │   ├── dl/   (Direct load instances)
    │   ├── wmma/ (WMMA instances for 2D)
    │   └── xdl/  (XDL instances for 2D - multiple layout subdirs)
    ├── grouped_conv3d_bwd_data/
    │   ├── wmma/ (WMMA instances for 3D)
    │   └── xdl/  (XDL instances for 3D)
    └── grouped_conv3d_bwd_weight/
        ├── dl/                              (Direct load)
        ├── wmma/                            (WMMA)
        ├── xdl/                             (XDL)
        ├── grouped_conv3d_bwd_weight_bilinear/
        └── grouped_conv3d_bwd_weight_scale/
```

## Summary

The Composable Kernel library provides a comprehensive set of optimized grouped convolution backward kernels:

### Backward Data Algorithms: 2
- **XDL variant**: ~200+ instances across all data types and layouts
- **WMMA variant**: ~30 instances for FP16 and INT8

### Backward Weight Algorithms: 9 (all part of static library)
1. **DeviceGroupedConvBwdWeight_Xdl_CShuffleV3** - Latest XDL (recommended)
2. **DeviceGroupedConvBwdWeight_Wmma_CShuffleV3** - Latest WMMA (recommended)
3. **DeviceGroupedConvBwdWeight_Xdl_CShuffle** - Original XDL (legacy but maintained)
4. **DeviceGroupedConvBwdWeight_Wmma_CShuffle** - Original WMMA (3D only)
5. **DeviceGroupedConvBwdWeight_TwoStage_Xdl_CShuffle** - Two-stage XDL
6. **DeviceGroupedConvBwdWeight_TwoStage_Wmma_CShuffleV3** - Two-stage WMMA
7. **DeviceGroupedConvBwdWeight_DL** - Direct load variant
8. **DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle** - XDL with fused ops
9. **DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3** - WMMA with fused ops

**Total estimated instances:** 300-400+ across all algorithms, data types, layouts, and specializations

**Key differentiators:**
- V3 variants: Latest optimizations, recommended for new code
- Two-stage variants: Better for very large convolutions
- MultipleD variants: Support fused element-wise operations
- DL variant: No shared memory overhead
- Wide range of data types (FP16, BF16, FP32, INT8, FP8/BF8)
- Various tensor layout combinations
- Advanced optimizations (CShuffle, Split-K, Pipeline tuning)
