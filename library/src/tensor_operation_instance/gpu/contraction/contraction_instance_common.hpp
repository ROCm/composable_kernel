// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// This (ifndef) is a hack to use customized behavior for buffer load rather than using default
// setting Don't use this hack unless absolutely necessary!
// FIXME: make the behavior of buffer load a configurable (template) parameter of each device op
#define CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 1

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/contraction/device_contraction_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

// Macro to generate a contraction device operation instance definition and its
// registration function. Each invocation produces one using-alias and one
// add_device_* function inside ck::tensor_operation::device::instance.
//
// Parameters:
//   INST_TPL     — instance template (e.g. device_contraction_kk_instance,
//                  device_contraction_f64_kk_instance)
//   OP_NAME      — lowercase operation name for identifier construction
//                  (bilinear or scale)
//   CDE_OP       — C++ element-wise operation type for template argument
//                  (Bilinear or Scale)
//   NDIM_VAL     — number of dimensions (2 or 6)
//   NAME_SUFFIX  — data-type and layout suffix for the generated names
//                  (e.g. f32_f32_f32_f32_kknn, bf16_bf16_bf16_bf16_compute_f32_knnn)
//   ADATA        — ADataType
//   BDATA        — BDataType
//   ACC          — AccDataType
//   CSHUFFLE     — CShuffleDataType
//   DS_TUPLE     — DsDataType (e.g. F32_Tuple, Empty_Tuple)
//   EDATA        — EDataType
//   COMPUTE      — ComputeDataType
//
// Example — bilinear, F32, kk layout, 2D:
//
//   CK_CONTRACTION_INSTANCE(device_contraction_kk_instance,
//       bilinear, Bilinear, 2, f32_f32_f32_f32_kknn,
//       F32, F32, F32, F32, F32_Tuple, F32, F32)
//
// Expands to:
//   using device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance = ...;
//   void add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance(...)
//   { ... }
//
// clang-format off
#define CK_CONTRACTION_INSTANCE(INST_TPL, OP_NAME, CDE_OP, NDIM_VAL,                             \
    NAME_SUFFIX, ADATA, BDATA, ACC, CSHUFFLE, DS_TUPLE, EDATA, COMPUTE)                           \
                                                                                                   \
namespace ck {                                                                                     \
namespace tensor_operation {                                                                       \
namespace device {                                                                                 \
namespace instance {                                                                               \
                                                                                                   \
using device_contraction_##OP_NAME##_m##NDIM_VAL##_n##NDIM_VAL##_k##NDIM_VAL##_xdl_c_shuffle_##NAME_SUFFIX##_instance = \
    INST_TPL<ADATA, BDATA, ACC, CSHUFFLE, DS_TUPLE, EDATA, COMPUTE,                               \
             PassThrough, PassThrough, CDE_OP, NDIM_VAL>;                                         \
                                                                                                   \
void add_device_contraction_##OP_NAME##_m##NDIM_VAL##_n##NDIM_VAL##_k##NDIM_VAL##_xdl_c_shuffle_##NAME_SUFFIX##_instance( \
    std::vector<std::unique_ptr<DeviceContractionMultipleD<NDIM_VAL, NDIM_VAL, NDIM_VAL,          \
        ADATA, BDATA, DS_TUPLE, EDATA, PassThrough, PassThrough, CDE_OP, COMPUTE>>>& instances)   \
{                                                                                                  \
    add_device_operation_instances(instances,                                                       \
        device_contraction_##OP_NAME##_m##NDIM_VAL##_n##NDIM_VAL##_k##NDIM_VAL##_xdl_c_shuffle_##NAME_SUFFIX##_instance{}); \
}                                                                                                  \
                                                                                                   \
} /* namespace instance */                                                                         \
} /* namespace device */                                                                           \
} /* namespace tensor_operation */                                                                 \
} /* namespace ck */
// clang-format on
