// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_convnd_fwd_nwc_kxc_nwk_xdl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

#include "parse_conv_parameter.hpp"

using InDataType  = ck::bhalf_t;
using WeiDataType = ck::bhalf_t;
using OutDataType = ck::bhalf_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InLayout  = ck::tensor_layout::convolution::NHWC;
using WeiLayout = ck::tensor_layout::convolution::KYXC;
using OutLayout = ck::tensor_layout::convolution::NHWK;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

template <ck::index_t NumDimSpatial>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::DeviceConvNdFwdNwcKxcNwk_Xdl<
    InDataType,     //
    WeiDataType,    //
    OutDataType,    //
    AccDataType,    //
    InElementOp,    // Input Elementwise Operation
    WeiElementOp,   // Weights Elementwise Operation
    OutElementOp,   // Output Elementwise Operation
    ConvFwdDefault, // ConvForwardSpecialization
    NumDimSpatial,  // NumDimSpatial
    256,            // BlockSize
    128,            // MPerBlock
    256,            // NPerBlock
    4,              // K0PerBlock
    8,              // K1
    32,             // MPerXdl
    32,             // NPerXdl
    2,              // MXdlPerWave
    4,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_K1
    true,           // ABlockLdsAddExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    8,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_K1
    true,           // BBlockLdsAddExtraN
    7,              // CThreadTransferSrcDstVectorDim
    1>;             // CThreadTransferDstScalarPerVector

template <ck::index_t NumDimSpatial>
using ReferenceConvNDFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                                WeiDataType,
                                                                                OutDataType,
                                                                                InElementOp,
                                                                                WeiElementOp,
                                                                                OutElementOp,
                                                                                NumDimSpatial>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;
    int num_dim_spatial  = 2;

    ck::tensor_operation::device::ConvParams params;

    if(argc >= 5)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        num_dim_spatial = std::stoi(argv[4]);
    }

    if(argc >= 6)
    {
        params = parse_conv_params(num_dim_spatial, argc, argv);
    }

    auto f_nchw_host_tensor_descriptor =
        [](ck::index_t n, ck::index_t c, std::vector<ck::index_t> spatial_lengths) {
            std::vector<std::size_t> nhwc_lengths{static_cast<std::size_t>(n),
                                                  static_cast<std::size_t>(c)};
            nhwc_lengths.insert(
                nhwc_lengths.begin() + 1, spatial_lengths.begin(), spatial_lengths.end());

            return transpose_host_tensor_descriptor_given_new2old(
                HostTensorDescriptor(nhwc_lengths), std::vector<std::size_t>({0, 3, 1, 2}));
        };

    Tensor<InDataType> input(
        f_nchw_host_tensor_descriptor(params.N_, params.C_, params.input_spatial_lengths_));
    Tensor<InDataType> weights(
        f_nchw_host_tensor_descriptor(params.K_, params.C_, params.filter_spatial_lengths_));
    Tensor<InDataType> host_output(
        f_nchw_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths()));
    Tensor<InDataType> device_output(
        f_nchw_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths()));

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weights: " << weights.mDesc << std::endl;
    std::cout << "output: " << host_output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        weights.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        weights.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weights.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * device_output.mDesc.GetElementSpace());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weights.mData.data());

    // do Conv
    auto conv     = DeviceConvNDFwdInstance<2>{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      params.N_,
                                      params.K_,
                                      params.C_,
                                      params.input_spatial_lengths_,
                                      params.filter_spatial_lengths_,
                                      params.GetOutputSpatialLengths(),
                                      params.conv_filter_strides_,
                                      params.conv_filter_dilations_,
                                      params.input_left_pads_,
                                      params.input_right_pads_,
                                      InElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = params.GetFlops();
    std::size_t num_btype = params.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        auto verify_f = [&input, &weights, &host_output, &params, &out_device_buf, &device_output](
                            const auto& ref_conv) {
            auto ref_invoker  = ref_conv.MakeInvoker();
            auto ref_argument = ref_conv.MakeArgument(input,
                                                      weights,
                                                      host_output,
                                                      params.conv_filter_strides_,
                                                      params.conv_filter_dilations_,
                                                      params.input_left_pads_,
                                                      params.input_right_pads_,
                                                      InElementOp{},
                                                      WeiElementOp{},
                                                      OutElementOp{});

            ref_invoker.Run(ref_argument);
            out_device_buf.FromDevice(device_output.mData.data());
            return ck::utils::check_err(host_output.mData,
                                        device_output.mData,
                                        "Error: incorrect results!",
                                        1e-5f,
                                        1e-4f)
                       ? 0
                       : 1;
        };

        switch(num_dim_spatial)
        {
        case 1: {
            auto ref_conv = ReferenceConvNDFwdInstance<1>();
            return verify_f(ref_conv);
        }
        case 2: {
            auto ref_conv = ReferenceConvNDFwdInstance<2>();
            return verify_f(ref_conv);
        }
        case 3: {
            auto ref_conv = ReferenceConvNDFwdInstance<3>();
            return verify_f(ref_conv);
        }
        default: {
            throw std::runtime_error("Unsupported number of spatial dimensions provided!");
        }
        }
    }
    return 0;
}
