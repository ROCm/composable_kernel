#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "element_wise_operation.hpp"
#include "device_conv2d_fwd_xdl_c_shuffle_bias_activation_nhwc_kyxc_nhwk.hpp"
#include "reference_conv_fwd_bias_activation.hpp"
#include "convolution_utility.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InLayout  = ck::tensor_layout::convolution::NHWC;
using WeiLayout = ck::tensor_layout::convolution::KYXC;
using OutLayout = ck::tensor_layout::convolution::NHWK;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddRelu;

static constexpr auto MemorySet = ck::InMemoryDataOperationEnum_t::Set;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization_t::Default;

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

using ReferenceConvFwdInstance =
    ck::tensor_operation::host::ReferenceConvFwd_Bias_Activation<InDataType,
                                                                 WeiDataType,
                                                                 OutDataType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 OutElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;

    // Conv shape
    ck::index_t N               = 128;
    ck::index_t K               = 256;
    ck::index_t C               = 192;
    ck::index_t Y               = 3;
    ck::index_t X               = 3;
    ck::index_t Hi              = 71;
    ck::index_t Wi              = 71;
    ck::index_t conv_stride_h   = 2;
    ck::index_t conv_stride_w   = 2;
    ck::index_t conv_dilation_h = 1;
    ck::index_t conv_dilation_w = 1;
    ck::index_t in_left_pad_h   = 1;
    ck::index_t in_left_pad_w   = 1;
    ck::index_t in_right_pad_h  = 1;
    ck::index_t in_right_pad_w  = 1;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
    }
    else if(argc == 19)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);

        N               = std::stoi(argv[4]);
        K               = std::stoi(argv[5]);
        C               = std::stoi(argv[6]);
        Y               = std::stoi(argv[7]);
        X               = std::stoi(argv[8]);
        Hi              = std::stoi(argv[9]);
        Wi              = std::stoi(argv[10]);
        conv_stride_h   = std::stoi(argv[11]);
        conv_stride_w   = std::stoi(argv[12]);
        conv_dilation_h = std::stoi(argv[13]);
        conv_dilation_w = std::stoi(argv[14]);
        in_left_pad_h   = std::stoi(argv[15]);
        in_left_pad_w   = std::stoi(argv[16]);
        in_right_pad_h  = std::stoi(argv[17]);
        in_right_pad_w  = std::stoi(argv[18]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4 to 18: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(0);
    }

    const std::vector<ck::index_t> conv_filter_strides{conv_stride_h, conv_stride_w};
    const std::vector<ck::index_t> conv_filter_dilations{conv_dilation_h, conv_dilation_w};
    const std::vector<ck::index_t> input_left_pads{in_left_pad_h, in_left_pad_w};
    const std::vector<ck::index_t> input_right_pads{in_right_pad_h, in_right_pad_w};
    const auto output_spatial_lengths =
        ck::tensor_operation::ConvolutionUtility::ComputeOutputSpatialLengths({Hi, Wi},
                                                                              {Y, X},
                                                                              conv_filter_strides,
                                                                              conv_filter_dilations,
                                                                              input_left_pads,
                                                                              input_right_pads);

    const ck::index_t Ho = output_spatial_lengths[0];
    const ck::index_t Wo = output_spatial_lengths[1];

    // tensor layout
    auto f_host_tensor_descriptor = [](std::size_t N_,
                                       std::size_t C_,
                                       std::size_t H,
                                       std::size_t W,
                                       auto layout) {
        if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NCHW>::value ||
                     ck::is_same<decltype(layout), ck::tensor_layout::convolution::KCYX>::value ||
                     ck::is_same<decltype(layout), ck::tensor_layout::convolution::NKHW>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                        std::vector<std::size_t>({C_ * H * W, H * W, W, 1}));
        }
        else if constexpr(ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::NHWC>::value ||
                          ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::KYXC>::value ||
                          ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::NHWK>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                        std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
        }
    };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<WeiDataType> wei_k_c_y_x(f_host_tensor_descriptor(K, C, Y, X, WeiLayout{}));
    Tensor<OutDataType> out_n_k_ho_wo_host_result(
        f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));
    Tensor<OutDataType> out_n_k_ho_wo_device_result(
        f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));

    // bias: assume contiguous 1d vector
    Tensor<OutDataType> bias_k(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(K)})));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo_host_result.mDesc << std::endl;
    std::cout << "bias_k: " << bias_k.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        bias_k.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias_k.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_k_ho_wo_device_result.mDesc.GetElementSpace());
    DeviceMem bias_device_buf(sizeof(OutDataType) * bias_k.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    bias_device_buf.ToDevice(bias_k.mData.data());

    auto conv    = DeviceConvFwdInstance{};
    auto invoker = conv.MakeInvoker();
    auto argument =
        conv.MakeArgument(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                          static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                          static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                          static_cast<const OutDataType*>(bias_device_buf.GetDeviceBuffer()),
                          N,
                          K,
                          C,
                          std::vector<ck::index_t>{Hi, Wi},
                          std::vector<ck::index_t>{Y, X},
                          std::vector<ck::index_t>{Ho, Wo},
                          conv_filter_strides,
                          conv_filter_dilations,
                          input_left_pads,
                          input_right_pads,
                          InElementOp{},
                          WeiElementOp{},
                          OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device operator with the specified compilation parameters does "
            "not support this problem");
    }

    float ave_time = invoker.Run(argument, nrepeat);

    std::size_t flop = std::size_t(2) * N * K * Ho * Wo * C * Y * X;

    std::size_t num_btype = sizeof(InDataType) * (N * C * Hi * Wi) +
                            sizeof(WeiDataType) * (K * C * Y * X) +
                            sizeof(OutDataType) * (N * K * Ho * Wo) + sizeof(OutDataType) * (K);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        auto ref_conv    = ReferenceConvFwdInstance{};
        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi,
                                                  wei_k_c_y_x,
                                                  out_n_k_ho_wo_host_result,
                                                  bias_k,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});
        ref_invoker.Run(ref_argument);

        out_device_buf.FromDevice(out_n_k_ho_wo_device_result.mData.data());

        check_error(out_n_k_ho_wo_host_result, out_n_k_ho_wo_device_result);
    }
}
