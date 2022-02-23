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
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp"
#include "device_conv3d_fwd_naive_ndhwc_kzyxc_ndhwk.hpp"
#include "convolution_utility.hpp"

// convolution data type
using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

using F16 = ck::half_t;
using F32 = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InLayout  = ck::tensor_layout::convolution::NDHWC;
using WeiLayout = ck::tensor_layout::convolution::KZYXC;
using OutLayout = ck::tensor_layout::convolution::NDHWK;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization_t::Default;

using DeviceConv3dFwdInstance = ck::tensor_operation::device::
    DeviceConv3dFwdXdl_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K<
        InDataType,     // InData
        WeiDataType,    // WeiData
        OutDataType,    // OutData
        AccDataType,    // AccData
        InElementOp,    // InElementwise Operation
        WeiElementOp,   // WeiElementwise Operation
        OutElementOp,   // OutElementwise Operation
        ConvFwdDefault, // ConvForwardSpecialization
        256,            // BlockSize
        128,            // MPerBlock
        256,            // NPerBlock
        4,              // K0PerBlock
        8,              // K1. K0PerBlock * K1 = KPerBlock
        32,             // MPerXDL
        32,             // NPerXDL. Each XDL computes a matrix of size (MPerXDL, NPerBlock)
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

int main(int argc, char* argv[])
{
    bool do_verification = false;
    int init_method      = 0;
    int nrepeat          = 5;

    // convolution shape
    ck::index_t N                                   = 4;
    ck::index_t K                                   = 256;
    ck::index_t C                                   = 192;
    std::vector<ck::index_t> in_spatial_lengths     = {71, 71, 71};
    std::vector<ck::index_t> filter_spatial_lengths = {3, 3, 3};
    std::vector<ck::index_t> conv_filter_strides    = {2, 2, 2};
    std::vector<ck::index_t> conv_filter_dilations  = {1, 1, 1};
    std::vector<ck::index_t> in_left_pads           = {1, 1, 1};
    std::vector<ck::index_t> in_right_pads          = {1, 1, 1};

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
    }
    else if(argc == 25)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);

        N                         = std::stoi(argv[4]);
        K                         = std::stoi(argv[5]);
        C                         = std::stoi(argv[6]);
        filter_spatial_lengths[0] = std::stoi(argv[7]);
        filter_spatial_lengths[1] = std::stoi(argv[8]);
        filter_spatial_lengths[2] = std::stoi(argv[9]);
        in_spatial_lengths[0]     = std::stoi(argv[10]);
        in_spatial_lengths[1]     = std::stoi(argv[11]);
        in_spatial_lengths[2]     = std::stoi(argv[12]);
        conv_filter_strides[0]    = std::stoi(argv[13]);
        conv_filter_strides[1]    = std::stoi(argv[14]);
        conv_filter_strides[2]    = std::stoi(argv[15]);
        conv_filter_dilations[0]  = std::stoi(argv[16]);
        conv_filter_dilations[1]  = std::stoi(argv[17]);
        conv_filter_dilations[2]  = std::stoi(argv[18]);
        in_left_pads[0]           = std::stoi(argv[19]);
        in_left_pads[1]           = std::stoi(argv[20]);
        in_left_pads[2]           = std::stoi(argv[21]);
        in_right_pads[0]          = std::stoi(argv[22]);
        in_right_pads[1]          = std::stoi(argv[23]);
        in_right_pads[2]          = std::stoi(argv[24]);
    }
    else
    {
        printf("Usage: 3 or 24 input arguments\n");
        printf(" arg1: verification (0=no, 1=yes)\n");
        printf(" arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf(" arg3: run kernel # of times (>1)\n");
        printf(" arg4 to 24: N, K, C, Z, Y, X, Di, Hi, Wi, Sz, Sy, Sz, Dz, Dy, Dx, LeftPz, LeftPy, "
               "LeftPz, RightPz, RightPy, RightPx\n");
        exit(0);
    }

    auto conv3d = DeviceConv3dFwdInstance{};

    const auto out_spatial_lengths =
        ck::tensor_operation::ConvolutionUtility::ComputeOutputSpatialLengths(
            in_spatial_lengths,
            filter_spatial_lengths,
            conv_filter_strides,
            conv_filter_dilations,
            in_left_pads,
            in_right_pads);
    Tensor<InDataType> in(
        {N, in_spatial_lengths[0], in_spatial_lengths[1], in_spatial_lengths[2], C});
    Tensor<WeiDataType> wei(
        {K, filter_spatial_lengths[0], filter_spatial_lengths[1], filter_spatial_lengths[2], C});
    Tensor<OutDataType> out(
        {N, out_spatial_lengths[0], out_spatial_lengths[1], out_spatial_lengths[2], K});

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    // do Convolution
    auto invoker  = conv3d.MakeInvoker();
    auto argument = conv3d.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                        static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                        static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                        N,
                                        K,
                                        C,
                                        in_spatial_lengths,
                                        filter_spatial_lengths,
                                        out_spatial_lengths,
                                        conv_filter_strides,
                                        conv_filter_dilations,
                                        in_left_pads,
                                        in_right_pads,
                                        InElementOp{},
                                        WeiElementOp{},
                                        OutElementOp{});

    if(!conv3d.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv3d with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, nrepeat);

    const auto Di = in_spatial_lengths[0];
    const auto Hi = in_spatial_lengths[1];
    const auto Wi = in_spatial_lengths[2];
    const auto Do = out_spatial_lengths[0];
    const auto Ho = out_spatial_lengths[1];
    const auto Wo = out_spatial_lengths[2];
    const auto Z  = filter_spatial_lengths[0];
    const auto Y  = filter_spatial_lengths[1];
    const auto X  = filter_spatial_lengths[2];

    std::size_t flop      = std::size_t(2) * N * K * Do * Ho * Wo * C * Z * Y * X;
    std::size_t num_btype = sizeof(InDataType) * N * Di * Hi * Wi * C +
                            sizeof(WeiDataType) * K * Z * Y * X * C +
                            sizeof(OutDataType) * N * Do * Ho * Wo * K;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    out_device_buf.FromDevice(out.mData.data());

    if(do_verification)
    {
        DeviceMem out_ref_device_buf(sizeof(OutDataType) * N * Do * Ho * Wo * K);

        using DeviceConv3dFwdNaive = ck::tensor_operation::device::
            DeviceConv3dFwdNaive_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K<
                InDataType,
                WeiDataType,
                OutDataType,
                AccDataType,
                InElementOp,
                WeiElementOp,
                OutElementOp>;
        auto conv3d_naive   = DeviceConv3dFwdNaive{};
        auto invoker_naive  = conv3d_naive.MakeInvoker();
        auto argument_naive = conv3d_naive.MakeArgument(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_ref_device_buf.GetDeviceBuffer()),
            N,
            K,
            C,
            in_spatial_lengths,
            filter_spatial_lengths,
            out_spatial_lengths,
            conv_filter_strides,
            conv_filter_dilations,
            in_left_pads,
            in_right_pads,
            InElementOp{},
            WeiElementOp{},
            OutElementOp{});

        if(!conv3d_naive.IsSupportedArgument(argument_naive))
        {
            throw std::runtime_error(
                "wrong! device_conv3d_naive does NOT support the specified compilation parameters");
        }
        invoker_naive.Run(argument_naive);

        Tensor<OutDataType> out_ref(
            {N, out_spatial_lengths[0], out_spatial_lengths[1], out_spatial_lengths[2], K});

        out_ref_device_buf.FromDevice(out_ref.mData.data());

        check_error(out_ref, out);
    }

    return 0;
}
