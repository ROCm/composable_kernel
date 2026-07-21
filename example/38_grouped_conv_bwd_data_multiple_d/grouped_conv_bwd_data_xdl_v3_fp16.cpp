// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Reproducer for ASAN heap-buffer-overflow in grouped_conv_bwd_data_xdl_cshuffle_v3
// Issue: Multi-Arch CI ASAN #3383 / #4385
// Failing shape: G:1 N:1 C:1 K:1 H:28 W:28 y:3 x:3 pad.y:1 pad.x:1 stride.y:1 stride.x:1
//
// Directly instantiates only the crashing instance (no factory, no other instances):
//   DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3<
//     2, NHWGK, GKYXC, Tuple<>, NHWGC, F16, F16, F32, F16, Tuple<>, F16,
//     PassThrough, PassThrough, PassThrough, Default, true, true,
//     256, 128, 64, 32, 8, 8, 32, 32, 2, 1,
//     S<4,64,1>, S<1,0,2>, S<1,0,2>, 2, 1, 8, 0,
//     S<4,64,1>, S<0,2,1>, S<0,2,1>, 1, 1, 8, 0,
//     1, 1, S<1,64,1,4>, 1,
//     Intrawave, v1, F16, F16, false, true>

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "common.hpp"

using F16         = ck::half_t;
using F32         = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using OutLayout = ck::tensor_layout::convolution::NHWGK;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using InLayout  = ck::tensor_layout::convolution::NHWGC;
using DsLayout  = ck::Tuple<>;

// The exact instance from the CI backtrace.
using CrashingOp = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3<
    2,           // NDimSpatial
    OutLayout,   // ALayout  (output / dy)
    WeiLayout,   // BLayout  (weight)
    DsLayout,    // DsLayout
    InLayout,    // ELayout  (input / dx)
    F16,         // ADataType
    F16,         // BDataType
    F32,         // AccDataType
    F16,         // CShuffleDataType
    ck::Tuple<>, // DsDataType
    F16,         // EDataType
    PassThrough, // AElementwiseOperation
    PassThrough, // BElementwiseOperation
    PassThrough, // CDEElementwiseOperation
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default,
    true,        // DoPadGemmM
    true,        // DoPadGemmN
    256,         // BlockSize
    128,         // MPerBlock
    64,          // NPerBlock
    32,          // KPerBlock
    8,           // AK1
    8,           // BK1
    32,          // MPerXdl
    32,          // NPerXdl
    2,           // MXdlPerWave
    1,           // NXdlPerWave
    S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
    S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
    2,           // ABlockTransferSrcVectorDim
    1,           // ABlockTransferSrcScalarPerVector  (= ScalarPerVector=1)
    8,           // ABlockTransferDstScalarPerVector_AK1
    0,           // ABlockLdsExtraM
    S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
    S<0, 2, 1>,  // BBlockTransferThreadClusterArrangeOrder
    S<0, 2, 1>,  // BBlockTransferSrcAccessOrder
    1,           // BBlockTransferSrcVectorDim
    1,           // BBlockTransferSrcScalarPerVector  (= ScalarPerVector=1)
    8,           // BBlockTransferDstScalarPerVector_BK1
    0,           // BBlockLdsExtraN
    1,           // CShuffleMXdlPerWavePerShuffle
    1,           // CShuffleNXdlPerWavePerShuffle
    S<1, 64, 1, 4>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    1,           // CDEBlockTransferScalarPerVector_NPerBlock
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    F16,         // AComputeDataType
    F16,         // BComputeDataType
    false,       // AEnableLds (DirectLoad = false)
    true>;       // large-tensor (int64 index)

int main(int argc, char* argv[])
{
    // Bug-triggering shape from ASAN #3383 / #4385
    ck::utils::conv::ConvParam conv_params{
        NDimSpatial, 1, 1, 1, 1, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};

    if(argc >= 5)
    {
        const int ndim = std::stoi(argv[4]);
        conv_params    = ck::utils::conv::parse_conv_param(ndim, 5, argv);
    }

    std::cout << "Conv params:"
              << " G:" << conv_params.G_ << " N:" << conv_params.N_
              << " C:" << conv_params.C_ << " K:" << conv_params.K_
              << " H:" << conv_params.input_spatial_lengths_[0]
              << " W:" << conv_params.input_spatial_lengths_[1]
              << " y:" << conv_params.filter_spatial_lengths_[0]
              << " x:" << conv_params.filter_spatial_lengths_[1]
              << " pad.y:" << conv_params.input_left_pads_[0]
              << " pad.x:" << conv_params.input_left_pads_[1]
              << " stride.y:" << conv_params.conv_filter_strides_[0]
              << " stride.x" << conv_params.conv_filter_strides_[1]
              << " dilation.y:" << conv_params.conv_filter_dilations_[0]
              << " dilation.x" << conv_params.conv_filter_dilations_[1]
              << std::endl;

    const auto out_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
            conv_params);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
            conv_params);
    const auto in_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_params);

    Tensor<F16> out(out_desc);
    Tensor<F16> wei(wei_desc);

    out.GenerateTensorValue(GeneratorTensor_2<F16>{-5, 5});
    wei.GenerateTensorValue(GeneratorTensor_2<F16>{-5, 5});

    DeviceMem out_buf(sizeof(F16) * out.mDesc.GetElementSpaceSize());
    DeviceMem wei_buf(sizeof(F16) * wei.mDesc.GetElementSpaceSize());
    DeviceMem in_buf(sizeof(F16) * in_desc.GetElementSpaceSize());
    out_buf.ToDevice(out.mData.data());
    wei_buf.ToDevice(wei.mData.data());
    in_buf.SetZero();

    std::array<ck::index_t, NDimSpatial + 3> a_lengths{}, a_strides{}, b_lengths{}, b_strides{},
        e_lengths{}, e_strides{};
    std::array<ck::index_t, NDimSpatial> filter_strides{}, filter_dilations{}, left_pads{},
        right_pads{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };
    copy(out_desc.GetLengths(), a_lengths);
    copy(out_desc.GetStrides(), a_strides);
    copy(wei_desc.GetLengths(), b_lengths);
    copy(wei_desc.GetStrides(), b_strides);
    copy(in_desc.GetLengths(), e_lengths);
    copy(in_desc.GetStrides(), e_strides);
    copy(conv_params.conv_filter_strides_, filter_strides);
    copy(conv_params.conv_filter_dilations_, filter_dilations);
    copy(conv_params.input_left_pads_, left_pads);
    copy(conv_params.input_right_pads_, right_pads);

    CrashingOp op{};
    std::cout << "Instance: " << op.GetTypeString() << std::endl;

    auto arg = op.MakeArgumentPointer(out_buf.GetDeviceBuffer(),
                                      wei_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      in_buf.GetDeviceBuffer(),
                                      a_lengths, a_strides,
                                      b_lengths, b_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      e_lengths, e_strides,
                                      filter_strides, filter_dilations,
                                      left_pads, right_pads,
                                      PassThrough{}, PassThrough{}, PassThrough{});

    std::cout << "IsSupportedArgument: " << op.IsSupportedArgument(arg.get()) << std::endl;

    auto invoker = op.MakeInvokerPointer();
    invoker->Run(arg.get(), StreamConfig{nullptr, false});

    return 0;
}
