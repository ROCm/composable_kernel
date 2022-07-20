// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv2d_fwd_xdl_c_shuffle_bias_activation_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

namespace {

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InLayout  = ck::tensor_layout::convolution::NHWC;
using WeiLayout = ck::tensor_layout::convolution::KYXC;
using OutLayout = ck::tensor_layout::convolution::NHWK;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddRelu;

static constexpr auto MemorySet = ck::InMemoryDataOperationEnum::Set;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

// clang-format off
using DeviceConvFwdInstance = ck::tensor_operation::device::
    DeviceConv2dFwdXdl_C_Shuffle_Bias_Activation_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        InDataType,                   // InDataType
        WeiDataType,                  // WeiDataType
        OutDataType,                  // OutDataType
        AccDataType,                  // AccDataType
        InElementOp,                  // InElementwiseOperation
        WeiElementOp,                 // WeiElementwiseOperation
        OutElementOp,                 // OutElementwiseOperation
        MemorySet,                    // OutGlobalMemoryDataOperation
        ConvFwdDefault,               // ConvForwardSpecialization
        256,                          // BlockSize
        128,                          // MPerBlock
        256,                          // NPerBlock
        4,                            // K0PerBlock
        8,                            // K1
        32,                           // MPerXdl
        32,                           // NPerXdl
        2,                            // MXdlPerWave
        4,                            // NXdlPerWave
        S<4, 64, 1>,                  // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,                   // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,                   // ABlockTransferSrcAccessOrder
        2,                            // ABlockTransferSrcVectorDim
        8,                            // ABlockTransferSrcScalarPerVector
        8,                            // ABlockTransferDstScalarPerVector_K1
        true,                         // ABlockLdsAddExtraM
        S<4, 64, 1>,                  // BBlockTransferThreadClusterLengths_K0_N_K1
        S<1, 0, 2>,                   // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,                   // BBlockTransferSrcAccessOrder
        2,                            // BBlockTransferSrcVectorDim
        8,                            // BBlockTransferSrcScalarPerVector
        8,                            // BBlockTransferDstScalarPerVector_K1
        true,                         // BBlockLdsAddExtraN
        1,                            // CShuffleMXdlPerWavePerShuffle
        1,                            // CShuffleNXdlPerWavePerShuffle
        S<1, 1, 32, 1, 1, 8>,         // CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
        8>;                           // CBlockTransferScalarPerVector_NWaveNPerXdl
// clang-format on

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << "arg4: N spatial dimensions (default 2)\n"
              << "Following arguments (depending on number of spatial dims):\n"
              << " N, K, C, \n"
              << " <filter spatial dimensions>, (ie Y, X for 2D)\n"
              << " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
              << " <strides>, (ie Sy, Sx for 2D)\n"
              << " <dilations>, (ie Dy, Dx for 2D)\n"
              << " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
              << " <right padding>, (ie RightPy, RightPx for 2D)\n"
              << std::endl;
}

ck::utils::conv::ConvParam parse_conv_params(int num_dim_spatial, int arg_idx, char* const argv[])
{
    const ck::index_t N = std::stoi(argv[arg_idx++]);
    const ck::index_t K = std::stoi(argv[arg_idx++]);
    const ck::index_t C = std::stoi(argv[arg_idx++]);

    std::vector<ck::index_t> filter_spatial_lengths(num_dim_spatial);
    std::vector<ck::index_t> input_spatial_lengths(num_dim_spatial);
    std::vector<ck::index_t> conv_filter_strides(num_dim_spatial);
    std::vector<ck::index_t> conv_filter_dilations(num_dim_spatial);
    std::vector<ck::index_t> input_left_pads(num_dim_spatial);
    std::vector<ck::index_t> input_right_pads(num_dim_spatial);

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        filter_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_strides[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_dilations[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_left_pads[i] = std::stoi(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_right_pads[i] = std::stoi(argv[arg_idx++]);
    }

    return ck::utils::conv::ConvParam{num_dim_spatial,
                                      N,
                                      K,
                                      C,
                                      filter_spatial_lengths,
                                      input_spatial_lengths,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads};
}

} // namespace

int main(int argc, char* argv[])
{
    print_helper_msg();

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int num_dim_spatial  = 2;

    ck::utils::conv::ConvParam params{
        2, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        num_dim_spatial = std::stoi(argv[4]);

        params = parse_conv_params(num_dim_spatial, 5, argv);
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    auto f_nhwc_host_tensor_descriptor =
        [](ck::index_t n, ck::index_t c, std::vector<ck::index_t> spatial_lengths) {
            std::vector<std::size_t> nhwc_lengths{static_cast<std::size_t>(n),
                                                  static_cast<std::size_t>(c)};
            nhwc_lengths.insert(
                nhwc_lengths.begin() + 1, spatial_lengths.begin(), spatial_lengths.end());

            return HostTensorDescriptor(nhwc_lengths);
        };

    Tensor<InDataType> in_n_hi_wi_c(
        f_nhwc_host_tensor_descriptor(params.N_, params.C_, params.input_spatial_lengths_));
    Tensor<WeiDataType> wei_k_y_x_c(
        f_nhwc_host_tensor_descriptor(params.K_, params.C_, params.filter_spatial_lengths_));
    // bias: assume contiguous 1d vector
    Tensor<OutDataType> bias_k(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(params.K_)})));
    Tensor<OutDataType> out_n_ho_wo_k_host(
        f_nhwc_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths()));
    Tensor<OutDataType> out_n_ho_wo_k_device(
        f_nhwc_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths()));

    std::cout << "in_n_hi_wi_c: " << in_n_hi_wi_c.mDesc << std::endl;
    std::cout << "wei_k_y_x_c: " << wei_k_y_x_c.mDesc << std::endl;
    std::cout << "bias_k: " << bias_k.mDesc << std::endl;
    std::cout << "output: " << out_n_ho_wo_k_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in_n_hi_wi_c.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei_k_y_x_c.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        bias_k.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in_n_hi_wi_c.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei_k_y_x_c.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias_k.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem bias_device_buf(sizeof(OutDataType) * bias_k.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_ho_wo_k_device.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_hi_wi_c.mData.data());
    wei_device_buf.ToDevice(wei_k_y_x_c.mData.data());
    bias_device_buf.ToDevice(bias_k.mData.data());

    // do GEMM
    auto conv    = DeviceConvFwdInstance{};
    auto invoker = conv.MakeInvoker();
    auto argument =
        conv.MakeArgument(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                          static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                          static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                          static_cast<const OutDataType*>(bias_device_buf.GetDeviceBuffer()),
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
                          in_element_op,
                          wei_element_op,
                          out_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = params.GetFlops();
    std::size_t num_btype = params.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        // use OutDataType for intermediate data
        Tensor<OutDataType> tmp_n_ho_wo_k_host(
            f_nhwc_host_tensor_descriptor(params.N_, params.K_, params.GetOutputSpatialLengths()));

        auto ref_conv =
            ck::tensor_operation::host::ReferenceConvFwd<2,
                                                         ck::tensor_layout::convolution::NHWC,
                                                         ck::tensor_layout::convolution::KYXC,
                                                         ck::tensor_layout::convolution::NHWK,
                                                         InDataType,
                                                         WeiDataType,
                                                         OutDataType,
                                                         InElementOp,
                                                         WeiElementOp,
                                                         PassThrough>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in_n_hi_wi_c,
                                                  wei_k_y_x_c,
                                                  tmp_n_ho_wo_k_host,
                                                  params.conv_filter_strides_,
                                                  params.conv_filter_dilations_,
                                                  params.input_left_pads_,
                                                  params.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        // FIXME: implement reference pointwise operation
        for(int n = 0; n < params.N_; n++)
        {
            for(int ho = 0; ho < params.output_spatial_lengths_[0]; ho++)
            {
                for(int wo = 0; wo < params.output_spatial_lengths_[1]; wo++)
                {
                    for(int k = 0; k < params.K_; k++)
                    {
                        out_element_op(out_n_ho_wo_k_host(n, ho, wo, k),
                                       tmp_n_ho_wo_k_host(n, ho, wo, k),
                                       bias_k(k));
                    }
                }
            }
        }

        out_device_buf.FromDevice(out_n_ho_wo_k_device.mData.data());

        return ck::utils::check_err(out_n_ho_wo_k_host.mData,
                                    out_n_ho_wo_k_device.mData,
                                    "Error: incorrect results!",
                                    1e-5f,
                                    1e-4f)
                   ? 0
                   : 1;
    }

    return 0;
}
