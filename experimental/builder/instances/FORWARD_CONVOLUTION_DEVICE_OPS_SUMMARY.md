# Forward Convolution Device Operations - Comprehensive Summary

**Generated:** October 29, 2025  
**Namespace:** `ck::tensor_operation::device`  
**Location:** `/library/include` and `/library/src` directories

---

## Overview

This document provides a comprehensive list of all device operations used for forward convolutions in the Composable Kernel library, along with their template instantiations.

### Total Statistics
- **Unique Device Operation Types:** 8
- **Total Template Instantiations:** 1,000
- **Files Analyzed:** 420
- **Header Files:** Located in `/library/include/ck/library/tensor_operation_instance/gpu/`
- **Source Files:** Located in `/library/src/tensor_operation_instance/gpu/`

---

## Device Operations Table

### Grouped Convolution Device Operations

| # | Device Operation Name | Instantiations | Files | Description |
|---|----------------------|----------------|-------|-------------|
| 1 | `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle` | 570 | 9 | XDL-based with multiple ABD inputs, CShuffle optimization |
| 2 | `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3` | 141 | 2 | Version 3 with advanced pipeline scheduling |
| 3 | `DeviceGroupedConvFwdMultipleD_Wmma_CShuffle` | 42 | 1 | WMMA-based (Wave Matrix Multiply Accumulate) |
| 4 | `DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor` | 12 | 1 | Optimized for large tensor dimensions |
| 5 | `DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK` | 6 | 1 | Direct Load implementation |
| **Total** | **Grouped Convolution Operations** | **771** | **14** | |

### Non-Grouped Convolution Device Operations

| # | Device Operation Name | Instantiations | Files | Description |
|---|----------------------|----------------|-------|-------------|
| 1 | `DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K` | 156 | 4 | Standard 2D convolution, explicit layout |
| 2 | `DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K` | 61 | 1 | 2D convolution with C-Shuffle optimization |
| 3 | `DeviceConvFwd` | 12 | 6 | Generic interface type |
| **Total** | **Non-Grouped Operations** | **229** | **11** | |

### Grand Total: 1,000 Template Instantiations across 8 Device Operation Types

---

## Device Operations List

### 1. DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
**Most commonly used device operation for forward convolutions**
- **Template Instantiations:** 570
- **Files:** 9 header files
- **Primary Header:** `grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp`
- **Description:** XDL-based grouped convolution with multiple auxiliary inputs/outputs and CShuffle optimization
- **Variants:**
  - Standard instances (BF16, F16, F32, INT8, F8, BF8)
  - 16x16 MFMA instances
  - NCHW layout instances
  - Generic instances
  - Compute-friendly instances with FP8

**Key Files:**
- `device_grouped_conv_fwd_xdl_instance.hpp` - Main instantiations (193 instances)
- `device_grouped_conv_fwd_xdl_bilinear_instance.hpp` - With bilinear fusion (80 instances)
- `device_grouped_conv_fwd_xdl_scale_instance.hpp` - With scale operation (80 instances)
- `device_grouped_conv_fwd_xdl_dynamic_op_instance.hpp` - With dynamic operations (64 instances)
- `device_grouped_conv_fwd_xdl_outelementop_instance.hpp` - With custom output ops (80 instances)
- `device_grouped_conv_fwd_xdl_merged_groups_instance.hpp` - Merged groups optimization (21 instances)
- `device_grouped_conv_fwd_xdl_scaleadd_ab_instance.hpp` - ScaleAdd operations (20 instances)
- `device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_instance.hpp` - Fused operations (16 instances)
- `device_grouped_conv_fwd_xdl_binary_outelementop_instance.hpp` - Binary output ops (16 instances)

---

### 2. DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K
**Standard 2D convolution forward operation**
- **Template Instantiations:** 156
- **Files:** 4 source files
- **Description:** XDL-based 2D convolution with explicit NHWC layout

**Instantiation Files:**
- `conv2d_fwd/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instance.cpp` (39 instances)
- `conv2d_fwd/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instance.cpp` (39 instances)
- `conv2d_fwd/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instance.cpp` (39 instances)
- `conv2d_fwd/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instance.cpp` (39 instances)

---

### 3. DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3
**Version 3 of XDL CShuffle implementation**
- **Template Instantiations:** 141
- **Files:** 2 header files
- **Description:** Advanced XDL implementation with compute-friendly scheduling and memory optimizations

**Key Files:**
- `device_grouped_conv_fwd_xdl_comp_instance.hpp` (49 instances) - Compute-optimized variants
- `device_grouped_conv_fwd_xdl_mem_instance.hpp` (92 instances) - Memory-optimized variants

**Features:**
- BlockGemmPipelineScheduler variants (Interwave, Intrawave)
- Multiple pipeline versions (v1, v3, v4, v5)
- Support for BF16, F16, F32, TF32, INT8

---

### 4. DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K
**C-Shuffle variant for 2D convolution**
- **Template Instantiations:** 61
- **Files:** 1 source file
- **File:** `conv2d_fwd/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instance.cpp`
- **Data Type:** F16 only
- **Description:** CShuffle optimization for channel dimension

---

### 5. DeviceGroupedConvFwdMultipleD_Wmma_CShuffle
**WMMA-based grouped convolution**
- **Template Instantiations:** 42
- **Files:** 1 header file
- **File:** `grouped_conv_fwd/device_grouped_conv_fwd_wmma_instance.hpp`
- **Description:** Uses Wave Matrix Multiply Accumulate (WMMA) instructions
- **Data Types:** F16, INT8
- **Block Sizes:** 32, 64, 128, 256

---

### 6. DeviceConvFwd
**Generic convolution forward interface**
- **Template Instantiations:** 12
- **Files:** 6 files (headers and sources)
- **Description:** High-level interface type for convolution forward operations
- **Usage:** Factory pattern and API definitions

---

### 7. DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor
**Large tensor optimization**
- **Template Instantiations:** 12
- **Files:** 1 header file
- **File:** `grouped_conv_fwd/device_grouped_conv_fwd_xdl_large_tensor_instance.hpp`
- **Description:** Optimized for large tensor dimensions
- **Data Types:** BF16, F16, F32, TF32, INT8

---

### 8. DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK
**Direct Load (DL) implementation**
- **Template Instantiations:** 6
- **Files:** 1 header file
- **File:** `grouped_conv_fwd/device_grouped_conv_fwd_dl_instance.hpp`
- **Description:** Direct load convolution implementation for specific layouts
- **Data Types:** F16, F32

---

## File Organization

### Header Files (`/library/include/ck/library/tensor_operation_instance/gpu/`)

#### Grouped Convolution Forward (`grouped_conv_fwd/`)
1. `device_grouped_conv_fwd_dl_instance.hpp` - Direct Load implementations
2. `device_grouped_conv_fwd_wmma_instance.hpp` - WMMA implementations
3. `device_grouped_conv_fwd_xdl_instance.hpp` - **Main XDL implementations**
4. `device_grouped_conv_fwd_xdl_bilinear_instance.hpp` - With bilinear fusion
5. `device_grouped_conv_fwd_xdl_binary_outelementop_instance.hpp` - Binary output ops
6. `device_grouped_conv_fwd_xdl_comp_instance.hpp` - Compute-optimized
7. `device_grouped_conv_fwd_xdl_dynamic_op_instance.hpp` - Dynamic operations
8. `device_grouped_conv_fwd_xdl_large_tensor_instance.hpp` - Large tensor optimization
9. `device_grouped_conv_fwd_xdl_mem_instance.hpp` - Memory-optimized
10. `device_grouped_conv_fwd_xdl_merged_groups_instance.hpp` - Merged groups
11. `device_grouped_conv_fwd_xdl_outelementop_instance.hpp` - Custom output operations
12. `device_grouped_conv_fwd_xdl_scale_instance.hpp` - With scale operation
13. `device_grouped_conv_fwd_xdl_scaleadd_ab_instance.hpp` - ScaleAdd on inputs
14. `device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_instance.hpp` - Fused operations

### Source Files (`/library/src/tensor_operation_instance/gpu/`)

#### Conv2D Forward (`conv2d_fwd/`)
- Standard XDL instances for BF16, F16, F32, INT8
- C-Shuffle variant for F16

#### Grouped Conv1D/2D/3D Forward
Organized by:
- **Algorithm:** `dl/`, `wmma/`, `xdl/`
- **Optimization:** `comp/`, `mem/`, `large_tensor/`, `merged_groups/`
- **Data Type:** Per file (bf16, f16, f32, int8, fp8, bf8)
- **Layout:** Encoded in filename (nhwgc, ngchw, etc.)

---

## Template Instantiation Patterns

### Common Parameters
Template instantiations typically include:
- **Spatial Dimensions:** 1D, 2D, 3D (NDimSpatial)
- **Layouts:** NHWC, NCHW, and grouped variants (NHWGC, NGCHW, etc.)
- **Data Types:** BF16, F16, F32, TF32, INT8, F8, BF8
- **Accumulator Type:** F32, INT32
- **Block Sizes:** 32, 64, 128, 256
- **Thread Tile Sizes:** MPerBlock, NPerBlock, KPerBlock
- **MFMA Sizes:** 16x16, 32x32
- **Pipeline Stages:** 1-2 stages
- **Element-wise Operations:** PassThrough, Scale, ScaleAdd, Bilinear, ReLU, Clamp, etc.

### Example Instantiation (from DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle)
```cpp
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    NDimSpatial,    // 2 or 3
    ALayout,        // NHWGC, NGCHW, etc.
    BLayout,        // GKYXC, GKCYX, etc.
    DsLayout,       // Additional inputs layout
    ELayout,        // Output layout
    BF16,           // Input data type
    BF16,           // Weight data type
    F32,            // Accumulator type
    BF16,           // CShuffle data type
    DsDataTypes,    // Additional input types
    BF16,           // Output data type
    PassThrough,    // Input element-wise op
    PassThrough,    // Weight element-wise op
    OutElementOp,   // Output element-wise op
    ConvSpec,       // Convolution specialization
    GemmMNKPadding, // GEMM specialization
    1,              // NumGemmKPrefetchStage
    256,            // BlockSize
    128,            // MPerBlock
    128,            // NPerBlock
    32,             // KPerBlock
    8,              // AK1
    8,              // BK1
    32,             // MPerXdl
    32,             // NPerXdl
    2,              // MXdlPerWave
    2,              // NXdlPerWave
    // ... additional block transfer parameters
>
```

---

## Key Findings

1. **DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle is the primary device operation** with 570 instantiations across 9 different header files for various use cases and fusion patterns.

2. **Eight distinct device operation types** are used for forward convolutions, each optimized for different scenarios:
   - XDL (Matrix Core) based operations
   - WMMA (Wave Matrix Multiply) based operations  
   - Direct Load (DL) based operations
   - Specialized variants for large tensors, merged groups, and memory optimization

3. **Multiple optimization strategies:**
   - Compute-friendly scheduling (Interwave, Intrawave)
   - Memory access patterns (Inter-wave, Intra-wave)
   - Pipeline versions (v1, v3, v4, v5)
   - CShuffle for efficient data movement

4. **Comprehensive data type support:** BF16, F16, F32, TF32, INT8, F8, BF8

5. **Extensive fusion support:** Operations can be fused with Scale, ScaleAdd, Bilinear, ReLU, Clamp, and combinations thereof.

---

## Output Files

### Complete Instantiation Files (Recommended)

1. **`forward_conv_all_instantiations.txt`** (593KB, 4,271 lines)
   - **COMPLETE listing of ALL 1,000 template instantiations**
   - Human-readable format with line numbers
   - Every instantiation shown in full detail
   - Organized by device operation, then by file

2. **`forward_conv_all_instantiations.json`** (1.4MB)
   - **Structured JSON for programmatic instantiation generation**
   - Complete instantiation text for each template
   - Includes parsed parameters (data types, block sizes)
   - Hierarchical organization: device_operation → file → instantiations[]
   - Ready for automated code generation tools

### Summary and Quick Reference Files

3. **`FORWARD_CONVOLUTION_DEVICE_OPS_SUMMARY.md`** (This file)
   - Executive summary with tables
   - Device operation descriptions
   - File organization reference

4. **`forward_convolution_device_ops_report.txt`** (33KB)
   - High-level summary of all device operations
   - Lists all files containing each device operation
   - Quick reference guide

5. **`forward_conv_device_ops_detailed_report.txt`** (43KB)
   - Detailed report with sample template instantiations
   - Shows first 3 examples per file
   - Includes line numbers and file locations

6. **`forward_convolution_device_ops_data.json`** (34KB)
   - Basic JSON format with summary data
   - File paths and instantiation counts

---

## Usage Examples

### Example 1: DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle with BF16
Found in: `device_grouped_conv_fwd_xdl_instance.hpp`

This is the most versatile device operation with:
- Generic instances for all block sizes
- Optimized instances for small conv.K and conv.C
- NCHW layout support
- 16x16 MFMA support
- FP8 compute support

### Example 2: DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK
Found in: `device_grouped_conv_fwd_dl_instance.hpp`

Direct Load implementation with:
- F16 and F32 support
- Specific block configurations
- NHWC layout only

### Example 3: DeviceGroupedConvFwdMultipleD_Wmma_CShuffle  
Found in: `device_grouped_conv_fwd_wmma_instance.hpp`

WMMA-based implementation with:
- F16 and INT8 support
- Multiple block sizes (32, 64, 128, 256)
- 16x16 WMMA instruction usage

---

## File Locations Reference

### Main Device Operation Headers
```
library/include/ck/library/tensor_operation_instance/gpu/
├── grouped_conv_fwd/
│   ├── device_grouped_conv_fwd_xdl_instance.hpp          ← PRIMARY FILE
│   ├── device_grouped_conv_fwd_dl_instance.hpp
│   ├── device_grouped_conv_fwd_wmma_instance.hpp
│   ├── device_grouped_conv_fwd_xdl_comp_instance.hpp
│   ├── device_grouped_conv_fwd_xdl_mem_instance.hpp
│   ├── device_grouped_conv_fwd_xdl_large_tensor_instance.hpp
│   └── ... (other variants)
└── convolution_forward.hpp                               ← INTERFACE DEFINITIONS
```

### Source Instantiations
```
library/src/tensor_operation_instance/gpu/
├── conv2d_fwd/                    ← Non-grouped 2D convolutions
├── grouped_conv1d_fwd/            ← 1D grouped convolutions
├── grouped_conv2d_fwd/            ← 2D grouped convolutions
│   ├── dl/                        ← Direct Load variants
│   ├── wmma/                      ← WMMA variants
│   └── xdl/                       ← XDL variants
│       ├── comp/                  ← Compute-optimized
│       ├── mem/                   ← Memory-optimized
│       ├── large_tensor/          ← Large tensor optimized
│       └── merged_groups/         ← Merged groups optimized
└── grouped_conv3d_fwd/            ← 3D grouped convolutions
```

---

## Additional Resources

For detailed analysis and full template instantiations, refer to the generated reports:
- `forward_convolution_device_ops_report.txt` - Quick summary
- `forward_conv_device_ops_detailed_report.txt` - Full details with examples
- `forward_convolution_device_ops_data.json` - Machine-readable format

The analysis scripts used to generate this information:
- `extract_conv_fwd_device_ops.py` - Initial device operation extraction
- `extract_detailed_instantiations.py` - Detailed instantiation analysis
