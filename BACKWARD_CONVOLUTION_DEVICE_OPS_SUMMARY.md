# Backward Convolution Device Operations - Comprehensive Summary

**Generated:** October 29, 2025  
**Namespace:** `ck::tensor_operation::device`  
**Location:** `/library/include` and `/library/src` directories

---

## Overview

This document provides a comprehensive list of all device operations used for backward convolutions (both weight gradients and data gradients) in the Composable Kernel library.

### Total Statistics
- **Unique Device Operation Types:** 15
- **Backward Weight Operations:** 9 types (885 instantiations)
- **Backward Data Operations:** 7 types (1046 instantiations)
- **Total Template Instantiations:** 1827

---

## Backward Weight Device Operations

| # | Device Operation Name | Instantiations | Files | Primary Location |
|---|----------------------|----------------|-------|------------------|
| 1 | `DeviceGroupedConvBwdWeight` | 342 | 157 | `grouped_convolution_backward_weight.hpp` |
| 2 | `DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle` | 161 | 2 | `device_grouped_conv_bwd_weight_xdl_bilinear_instance.hpp` |
| 3 | `DeviceGroupedConvBwdWeight_Xdl_CShuffle` | 108 | 1 | `device_grouped_conv_bwd_weight_xdl_instance.hpp` |
| 4 | `DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K` | 104 | 4 | `device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instance.cpp` |
| 5 | `DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle` | 96 | 1 | `device_grouped_conv_bwd_weight_two_stage_xdl_instance.hpp` |
| 6 | `DeviceGroupedConvBwdWeight_Wmma_CShuffle` | 41 | 1 | `device_grouped_conv_bwd_weight_wmma_instance.hpp` |
| 7 | `DeviceGroupedConvBwdWeight_Xdl_CShuffleV3` | 16 | 1 | `device_grouped_conv_bwd_weight_v3_xdl_instance.hpp` |
| 8 | `DeviceGroupedConvBwdWeightMultipleD` | 14 | 2 | `grouped_convolution_backward_weight_bilinear.hpp` |
| 9 | `DeviceGroupedConvBwdWeight_Dl` | 3 | 1 | `device_grouped_conv_bwd_weight_dl_instance.hpp` |
| **Total** | **Backward Weight Operations** | **885** | **170** | |

---

## Backward Data Device Operations

| # | Device Operation Name | Instantiations | Files | Primary Location |
|---|----------------------|----------------|-------|------------------|
| 1 | `DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1` | 370 | 4 | `device_grouped_conv_bwd_data_transpose_xdl_instance.hpp` |
| 2 | `DeviceConvNdBwdDataNwcKxcNwk_Xdl` | 312 | 12 | `device_conv1d_bwd_data_xdl_nwc_kxc_nwk_bf16_instance.cpp` |
| 3 | `DeviceGroupedConvBwdDataMultipleD` | 156 | 76 | `grouped_convolution_backward_data.hpp` |
| 4 | `DeviceConv2dBwdDataXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K` | 104 | 4 | `device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instance.cpp` |
| 5 | `DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle` | 66 | 2 | `device_grouped_conv_bwd_data_wmma_f16_instance.hpp` |
| 6 | `DeviceConvBwdData` | 32 | 16 | `convolution_backward_data.hpp` |
| 7 | `DeviceConvNdBwdDataNwcKxcNwk_Dl` | 6 | 3 | `device_conv2d_bwd_data_dl_nhwc_kyxc_nhwk_f16_instance.cpp` |
| **Total** | **Backward Data Operations** | **1046** | **117** | |

### Grand Total: 1827 Template Instantiations across 15 Device Operation Types

---

## Output Files

### Complete Instantiation Files

1. **`backward_conv_all_instantiations.txt`**
   - COMPLETE listing of ALL template instantiations
   - Human-readable format with line numbers
   - Organized by operation type, then by file

2. **`backward_conv_all_instantiations.json`**
   - Structured JSON for programmatic instantiation generation
   - Separated into `backward_weight_operations` and `backward_data_operations`
   - Each instantiation includes full text and parsed parameters
   - Ready for automated code generation

3. **`BACKWARD_CONVOLUTION_DEVICE_OPS_SUMMARY.md`** (This file)
   - Executive summary with tables
   - Quick reference for all backward operations
