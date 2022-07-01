// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <tuple>

#include "ck/ck.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/device/device_convnd_fwd_xdl_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

using DeviceConvFwdNoOpPtr = DeviceConvFwdPtr<element_wise::PassThrough,
                                              element_wise::PassThrough,
                                              element_wise::PassThrough>;
namespace instance {

void add_device_convnd_2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_convnd_2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_convnd_2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_convnd_2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace test {
namespace conv {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

using DeviceConvFwdNoOpPtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<InElementOp, WeiElementOp, OutElementOp>;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

template <ck::index_t SpatialDims,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::
    DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        // clang-format off
        InDataType,         // 
        WeiDataType,        //
        OutDataType,        //
        AccDataType,        // Accumulator data type.
        InElementOp,        // Input Elementwise Operation
        WeiElementOp,       // Weights Elementwise Operation
        OutElementOp,       // Output Elementwise Operation
        ConvFwdDefault,     // ConvForwardSpecialization
        SpatialDims,        // SptialDims
        256,                // BlockSize
        128,                // MPerBlock
        256,                // NPerBlock
        4,                  // K0PerBlock
        8,                  // K1
        32,                 // MPerXdl
        32,                 // NPerXdl
        2,                  // MXdlPerWave
        4,                  // NXdlPerWave
        S<4, 64, 1>,        // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,         // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // ABlockTransferSrcAccessOrder
        2,                  // ABlockTransferSrcVectorDim
        8,                  // ABlockTransferSrcScalarPerVector
        8,                  // ABlockTransferDstScalarPerVector_K1
        true,               // ABlockLdsAddExtraM
        S<4, 64, 1>,        // BBlockTransferThreadClusterLengths_K0_N_K1
        S<1, 0, 2>,         // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // BBlockTransferSrcAccessOrder
        2,                  // BBlockTransferSrcVectorDim
        8,                  // BBlockTransferSrcScalarPerVector
        8,                  // BBlockTransferDstScalarPerVector_K1
        true,               // BBlockLdsAddExtraN
        7,                  // CThreadTransferSrcDstVectorDim
        1>;                // CThreadTransferDstScalarPerVector
// clang-format on

template <ck::index_t NDim,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType>
void get_test_convolution_fwd_instance(std::vector<DeviceConvFwdNoOpPtr>& instances)
{
    using ConvInstanceT =
        DeviceConvNDFwdInstance<NDim, InDataType, WeiDataType, OutDataType, AccDataType>;
    instances.emplace_back(std::make_unique<ConvInstanceT>());
}

// TODO (aosewski)
// Temporary solution to get all DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K
// instances. When switched over to DeviceConvNDFwdXdl for 2D remove ConvolutionNDFwdInstances
// structures.
template <typename InDataType, typename WeiDataType, typename OutDataType>
struct ConvolutionNDFwdInstances;

template <>
struct ConvolutionNDFwdInstances<float, float, float>
{
    static std::vector<DeviceConvFwdNoOpPtr> Get(std::size_t num_dim_spatial)
    {
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
        if(num_dim_spatial == 2)
        {
            ck::tensor_operation::device::instance::
                add_device_convnd_2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(conv_ptrs);
        }
        return conv_ptrs;
    }
};

template <>
struct ConvolutionNDFwdInstances<ck::half_t, ck::half_t, ck::half_t>
{
    static std::vector<DeviceConvFwdNoOpPtr> Get(std::size_t num_dim_spatial)
    {
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
        if(num_dim_spatial == 2)
        {
            ck::tensor_operation::device::instance::
                add_device_convnd_2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
        }
        return conv_ptrs;
    }
};

template <>
struct ConvolutionNDFwdInstances<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>
{
    static std::vector<DeviceConvFwdNoOpPtr> Get(std::size_t num_dim_spatial)
    {
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
        if(num_dim_spatial == 2)
        {
            ck::tensor_operation::device::instance::
                add_device_convnd_2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(conv_ptrs);
        }
        return conv_ptrs;
    }
};

template <>
struct ConvolutionNDFwdInstances<int8_t, int8_t, int8_t>
{
    static std::vector<DeviceConvFwdNoOpPtr> Get(std::size_t num_dim_spatial)
    {
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
        if(num_dim_spatial == 2)
        {
            ck::tensor_operation::device::instance::
                add_device_convnd_2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(conv_ptrs);
        }
        return conv_ptrs;
    }
};

} // namespace conv
} // namespace test
