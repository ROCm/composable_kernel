// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

constexpr ck::index_t NDimSpatial = 3;
using InDataType                  = ck::half_t;
using WeiDataType                 = ck::half_t;
using AccDataType                 = float;
using CShuffleDataType            = ck::half_t;
using OutDataType                 = ck::half_t;
using AComputeDataType            = ck::half_t;
using BComputeDataType            = ck::half_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// Use correct tensor layouts for WMMA (matching working tests)
using InLayout  = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::NDHWGK;

using InElementOp      = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp     = ck::tensor_operation::element_wise::PassThrough;
using DynamicElementOp = ck::tensor_operation::element_wise::DynamicUnaryOp;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceGroupedConvNDActivInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<
        NDimSpatial,      // NDimSpatial
        InLayout,         // ALayout
        WeiLayout,        // BLayout
        ck::Tuple<>,      // DsLayout
        OutLayout,        // ELayout
        InDataType,       // ADataType
        WeiDataType,      // BDataType
        AccDataType,      // AccDataType
        CShuffleDataType, // CShuffleDataType
        ck::Tuple<>,      // DsDataType
        OutDataType,      // EDataType
        InElementOp,      // AElementwiseOperation
        WeiElementOp,     // BElementwiseOperation
        DynamicElementOp, // CDEElementwiseOperation
        ConvSpec,         // ConvForwardSpecialization
        GemmSpec,         // GemmSpecialization
        64,               // BlockSize
        64,               // MPerBlock
        64,               // NPerBlock
        32,               // KPerBlock
        8,                // AK1
        8,                // BK1
        16,               // MPerWmma
        16,               // NPerWmma
        4,                // MRepeat
        2,                // NRepeat
        S<4, 16, 1>,      // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
        2,                // ABlockTransferSrcVectorDim
        1,                // ABlockTransferSrcScalarPerVector
        8,                // ABlockTransferDstScalarPerVector_AK1
        1,                // ABlockLdsExtraM
        S<4, 16, 1>,      // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
        2,                // BBlockTransferSrcVectorDim
        1,                // BBlockTransferSrcScalarPerVector
        8,                // BBlockTransferDstScalarPerVector_BK1
        1,                // BBlockLdsExtraN
        1,                // CShuffleMRepeatPerShuffle
        1,                // CShuffleNRepeatPerShuffle
        S<1, 16, 1, 4>,   // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        1,                // CDEBlockTransferScalarPerVector_NPerBlock
        ck::BlockGemmPipelineScheduler::Intrawave, // BlkGemmPipeSched
        ck::BlockGemmPipelineVersion::v1,          // BlkGemmPipelineVer
        true,                                      // UseThreadTileTransfer
        AComputeDataType,                          // AComputeDataType
        BComputeDataType,                          // BComputeDataType
        1>;                                        // NumGroupsToMerge

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvNDFwdInstance>
bool run_grouped_conv(bool do_verification,
                      int init_method,
                      bool time_kernel,
                      const ck::utils::conv::ConvParam& conv_param,
                      const ck::HostTensorDescriptor& in_g_n_c_wis_desc,
                      const ck::HostTensorDescriptor& wei_g_k_c_xs_desc,
                      const ck::HostTensorDescriptor& out_g_n_k_wos_desc,
                      const InElementOp& in_element_op,
                      const WeiElementOp& wei_element_op,
                      const OutElementOp& out_element_op)
{
    ck::Tensor<InDataType> in(in_g_n_c_wis_desc);
    ck::Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    ck::Tensor<OutDataType> out_host(out_g_n_k_wos_desc);
    ck::Tensor<OutDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-2, 2});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-2, 2});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.05, 0.05});
    }

    ck::DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    ck::DeviceMem out_device_buf(sizeof(OutDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // do Conv
    auto conv     = DeviceConvNDFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_device_buf.GetDeviceBuffer(),
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{{}},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{{}},
                                      e_g_n_k_wos_lengths,
                                      e_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      in_element_op,
                                      wei_element_op,
                                      out_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error("The device op with the specified compilation parameters does "
                                 "not support this convolution problem.");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     OutElementOp>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  out_host,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);

        out_device_buf.FromDevice(out_device.mData.data());

        return ck::utils::check_err(out_device, out_host, "Error: incorrect results!", 1e-3, 0.1);
    }

    return true;
}
