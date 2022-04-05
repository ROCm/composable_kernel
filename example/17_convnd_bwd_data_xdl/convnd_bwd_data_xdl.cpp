#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "config.hpp"
#include "conv_fwd_util.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "element_wise_operation.hpp"
#include "device_convnd_bwd_data_xdl_ndhwc_kzyxc_ndhwk.hpp"
#include "reference_conv_bwd_data.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;
static constexpr auto ConvBwdDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

using DeviceConvBwdDataBasePtr =
    ck::tensor_operation::device::DeviceConvBwdDataPtr<InElementOp, WeiElementOp, OutElementOp>;

template <ck::index_t NumDimSpatial>
using DeviceConvNDBwdDataInstance = ck::tensor_operation::device::
    DeviceConvndBwdDataXdl_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K<
        InDataType,     // InDataType
        WeiDataType,    // WeiDataType
        OutDataType,    // OutDataType
        AccDataType,    // AccDataType
        InElementOp,    // InElementwiseOperation
        WeiElementOp,   // WeiElementwiseOperation
        OutElementOp,   // OutElementwiseOperation
        ConvBwdDefault, // ConvolutionBackwardDataSpecialization
        NumDimSpatial,  // NumDimSpatial
        256,            // BlockSize
        128,            // MPerBlock
        128,            // NPerBlock
        4,              // K0PerBlock
        8,              // K1
        32,             // MPerXdl
        32,             // NPerXdl
        2,              // MXdlPerWave
        2,              // NXdlPerWave
        S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
        2,              // ABlockTransferSrcVectorDim
        8,              // ABlockTransferSrcScalarPerVector
        8,              // ABlockTransferDstScalarPerVector_K1
        true,           // ABlockLdsAddExtraM
        S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
        S<2, 0, 1>,     // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,     // BBlockTransferSrcAccessOrder
        1,              // BBlockTransferSrcVectorDim
        2,              // BBlockTransferSrcScalarPerVector
        8,              // BBlockTransferDstScalarPerVector_K1
        true,           // BBlockLdsAddExtraN
        7,
        1>; // GemmCThreadTransferDstScalarPerVector

template <ck::index_t NumDimSpatial>
using ReferenceConvBwdDataInstance =
    ck::tensor_operation::host::ReferenceConvBwdData<InDataType,
                                                     WeiDataType,
                                                     OutDataType,
                                                     AccDataType,
                                                     InElementOp,
                                                     WeiElementOp,
                                                     OutElementOp,
                                                     NumDimSpatial>;

void print_use_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=random value, 2= init to 1 )\n"
              << "arg3: run kernel # of times (>1)\n"
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
ck::utils::conv::ConvParams parse_conv_params(int num_dim_spatial, char* argv[])
{
    // (N, K, C) + num_dim_spatial * 6 (filter, input, strides, dilations, pad left, pad right)
    ck::utils::conv::ConvParams params;
    int arg_idx = 5;

    params.num_dim_spatial = num_dim_spatial;
    params.N               = std::stoi(argv[arg_idx++]);
    params.K               = std::stoi(argv[arg_idx++]);
    params.C               = std::stoi(argv[arg_idx++]);

    params.filter_spatial_lengths.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.filter_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_spatial_lengths.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_strides.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_strides[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_dilations.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_dilations[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_left_pads.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_left_pads[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_right_pads.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_right_pads[i] = std::stoi(argv[arg_idx++]);
    }

    return params;
}

DeviceConvBwdDataBasePtr get_conv_instance(int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 3: {
        return std::make_unique<DeviceConvNDBwdDataInstance<3>>();
    }
    case 2: {
        return std::make_unique<DeviceConvNDBwdDataInstance<2>>();
    }
    case 1: {
        return std::make_unique<DeviceConvNDBwdDataInstance<1>>();
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;
    int num_dim_spatial  = 2;

    ck::utils::conv::ConvParams params;
    params.C = 128;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
    }
    else if(argc > 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
        num_dim_spatial = std::stoi(argv[4]);
        // check args number
        int conv_args     = 3 + num_dim_spatial * 6;
        int cmdline_nargs = conv_args + 5;
        if(cmdline_nargs != argc)
        {
            print_use_msg();
            exit(1);
        }

        params = parse_conv_params(num_dim_spatial, argv);
    }
    else if(argc != 1)
    {
        print_use_msg();
        exit(1);
    }

    std::vector<std::size_t> input_dims{static_cast<std::size_t>(params.N),
                                        static_cast<std::size_t>(params.C)};
    input_dims.insert(std::end(input_dims),
                      std::begin(params.input_spatial_lengths),
                      std::end(params.input_spatial_lengths));

    std::vector<std::size_t> filter_dims{static_cast<std::size_t>(params.K),
                                         static_cast<std::size_t>(params.C)};
    filter_dims.insert(std::end(filter_dims),
                       std::begin(params.filter_spatial_lengths),
                       std::end(params.filter_spatial_lengths));

    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();
    std::vector<std::size_t> output_dims{static_cast<std::size_t>(params.N),
                                         static_cast<std::size_t>(params.K)};
    output_dims.insert(std::end(output_dims),
                       std::begin(output_spatial_lengths),
                       std::end(output_spatial_lengths));

    Tensor<InDataType> in_n_c_hi_wi_host_result(
        ck::utils::conv::get_input_host_tensor_descriptor(input_dims, num_dim_spatial));
    Tensor<InDataType> in_n_c_hi_wi_device_result(
        ck::utils::conv::get_input_host_tensor_descriptor(input_dims, num_dim_spatial));
    Tensor<WeiDataType> wei_k_c_y_x(
        ck::utils::conv::get_filters_host_tensor_descriptor(filter_dims, num_dim_spatial));
    Tensor<OutDataType> out_n_k_ho_wo(
        ck::utils::conv::get_output_host_tensor_descriptor(output_dims, num_dim_spatial));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi_host_result.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.2, 0.2});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.2, 0.2});
        break;
    default:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{1});
    }

    DeviceMem in_device_buf(sizeof(InDataType) *
                            in_n_c_hi_wi_device_result.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_k_ho_wo.mDesc.GetElementSpace());

    out_device_buf.ToDevice(out_n_k_ho_wo.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    // reset input to zero
    in_device_buf.SetZero();

    // do GEMM
    auto conv    = get_conv_instance(num_dim_spatial);
    auto invoker = conv->MakeInvokerPointer();
    auto argument =
        conv->MakeArgumentPointer(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                  static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                  static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                  params.N,
                                  params.K,
                                  params.C,
                                  params.input_spatial_lengths,
                                  params.filter_spatial_lengths,
                                  output_spatial_lengths,
                                  params.conv_filter_strides,
                                  params.conv_filter_dilations,
                                  params.input_left_pads,
                                  params.input_right_pads,
                                  InElementOp{},
                                  WeiElementOp{},
                                  OutElementOp{});

    if(!conv->IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float ave_time = invoker->Run(argument.get(), nrepeat);

    std::size_t flop = ck::utils::conv::get_flops(
        params.N, params.C, params.K, params.filter_spatial_lengths, output_spatial_lengths);
    std::size_t num_btype = ck::utils::conv::get_btype<InDataType, WeiDataType, OutDataType>(
        params.N,
        params.C,
        params.K,
        params.input_spatial_lengths,
        params.filter_spatial_lengths,
        output_spatial_lengths);

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        auto verify_f = [&](const auto& ref_conv) {
            auto ref_invoker = ref_conv.MakeInvoker();

            auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi_host_result,
                                                      wei_k_c_y_x,
                                                      out_n_k_ho_wo,
                                                      params.conv_filter_strides,
                                                      params.conv_filter_dilations,
                                                      params.input_left_pads,
                                                      params.input_right_pads,
                                                      InElementOp{},
                                                      WeiElementOp{},
                                                      OutElementOp{});

            ref_invoker.Run(ref_argument);

            in_device_buf.FromDevice(in_n_c_hi_wi_device_result.mData.data());

            check_error(in_n_c_hi_wi_host_result, in_n_c_hi_wi_device_result);
        };

        switch(num_dim_spatial)
        {
        case 3: {
            auto ref_conv = ReferenceConvBwdDataInstance<3>();
            verify_f(ref_conv);
            break;
        }
        case 2: {
            auto ref_conv = ReferenceConvBwdDataInstance<2>();
            verify_f(ref_conv);
            break;
        }
        case 1: {
            auto ref_conv = ReferenceConvBwdDataInstance<1>();
            verify_f(ref_conv);
            break;
        }
        default: {
            throw std::runtime_error("Unsupported number of spatial dimensions provided!");
        }
        }
    }
}
