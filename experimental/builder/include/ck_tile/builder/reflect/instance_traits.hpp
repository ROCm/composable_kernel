// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// Compile-time reflection for CK device kernel instances.
//
// - This is the Lowest-level reflection primitive for higher-level semantic abstractions (e.g.,
//   ConvTraits).
// - Extracts raw template parameters (block sizes, data types, layouts, tuning params) from kernel
//   specializations.
// - Provides uniform interface to query kernel configuration without implementation knowledge
// - Other details about the device kernels can be manually added to template specializations.
// - Currently supports:
//   - DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3

#pragma once

#include <array>
#include <string>
#include <sstream>
#include <type_traits>
#include <ck/utility/data_type.hpp>
#include <ck/utility/sequence.hpp>
#include <ck/utility/blkgemmpipe_scheduler.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include "instance_traits_util.hpp"

// Forward declaration to avoid circular dependency.
// This file will be included by the device implementation header, so we cannot include
// the implementation header here. We only need the template signature to pattern-match
// on template parameters - we don't need any implementation details.
namespace ck::tensor_operation::device {

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          ck::index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          ck::index_t BBlockLdsExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AComputeDataType,
          typename BComputeDataType>
struct DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3;

} // namespace ck::tensor_operation::device

namespace ck_tile::reflect {

// Primary template for InstanceTraits - extracts compile-time information directly from
// device kernel instances (e.g., DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3)
//
// This is an unspecialized template declaration. Actual specializations for specific
// device kernels are provided in separate header files (e.g.,
// instance_traits_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp).
template <typename Instance>
struct InstanceTraits;

// Free function that delegates to InstanceTraits static member function.
// Each InstanceTraits specialization provides its own instance_string() implementation.
template <typename T>
inline std::string instance_string()
{
    return InstanceTraits<T>::instance_string();
}

} // namespace ck_tile::reflect
