#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "config.hpp"
#include "conv_utils.hpp"
#include "device.hpp"
#include "device_tensor.hpp"
#include "device_convnd_fwd_xdl_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "reference_conv_fwd.hpp"
#include "tensor_layout.hpp"

using InDataType  = float;
using WeiDataType = float;
using OutDataType = float;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization_t::Default;

using DeviceConvFwdBasePtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<InElementOp, WeiElementOp, OutElementOp>;

template <ck::index_t NumDimSpatial>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::
    DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        // clang-format off
        InDataType,         // 
        WeiDataType,        //
        OutDataType,        //
        AccDataType,        // 
        InElementOp,        // Input Elementwise Operation
        WeiElementOp,       // Weights Elementwise Operation
        OutElementOp,       // Output Elementwise Operation
        ConvFwdDefault,     // ConvForwardSpecialization
        NumDimSpatial,      // NumDimSpatial
        256,                // BlockSize
        256,                // MPerBlock
        128,                // NPerBlock
        4,                  // K0PerBlock
        4,                  // K1
        32,                 // MPerXDL
        32,                 // NPerXDL
        4,                  // MXdlPerWave
        2,                  // NXdlPerWave
        S<4, 64, 1>,        // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,         // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // ABlockTransferSrcAccessOrder
        2,                  // ABlockTransferSrcVectorDim
        4,                  // ABlockTransferSrcScalarPerVector
        4,                  // ABlockTransferDstScalarPerVector_K1
        true,               // ABlockLdsAddExtraM
        S<4, 64, 1>,        // BBlockTransferThreadClusterLengths_K0_N_K1
        S<1, 0, 2>,         // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // BBlockTransferSrcAccessOrder
        2,                  // BBlockTransferSrcVectorDim
        4,                  // BBlockTransferSrcScalarPerVector
        4,                  // BBlockTransferDstScalarPerVector_K1
        true,               // BBlockTransferAddExtraN
        7,                  // CThreadTransferSrcDstVectorDim
        1>;                 // CThreadTransferDstScalarPerVector
// clang-format on

template <ck::index_t NumDimSpatial>
using ReferenceConvNDFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                                WeiDataType,
                                                                                OutDataType,
                                                                                InElementOp,
                                                                                WeiElementOp,
                                                                                OutElementOp,
                                                                                NumDimSpatial>;

DeviceConvFwdBasePtr GetConvInstance(int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 2: {
        return std::make_unique<DeviceConvNDFwdInstance<2>>();
    }
    case 1: {
        return std::make_unique<DeviceConvNDFwdInstance<1>>();
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

void PrintUseMsg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
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

ck::conv_util::ConvParams ParseConvParams(int num_dim_spatial, int argc, char* argv[])
{
    // (N, K, C) + num_dim_spatial * 6 (filter, input, strides, dilations, pad left, pad right)
    int conv_args     = 3 + num_dim_spatial * 6;
    int cmdline_nargs = conv_args + 5;
    if(cmdline_nargs != argc)
    {
        PrintUseMsg();
        exit(0);
    }

    ck::conv_util::ConvParams params;
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

HostTensorDescriptor GetOutputHostTensorDescriptor(const std::vector<std::size_t>& dims,
                                                   int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 2: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NHWK{});
    }
    case 1: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NWK{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

HostTensorDescriptor GetFiltersHostTensorDescriptor(const std::vector<std::size_t>& dims,
                                                    int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 2: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::KYXC{});
    }
    case 1: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::KXC{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

HostTensorDescriptor GetInputHostTensorDescriptor(const std::vector<std::size_t>& dims,
                                                  int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 2: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NHWC{});
    }
    case 1: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NWC{});
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

    ck::conv_util::ConvParams params;

    if(argc >= 5)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
        num_dim_spatial = std::stoi(argv[4]);
    }

    if(argc >= 6)
    {
        params = ParseConvParams(num_dim_spatial, argc, argv);
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

    Tensor<InDataType> input(GetInputHostTensorDescriptor(input_dims, num_dim_spatial));
    Tensor<WeiDataType> weights(GetFiltersHostTensorDescriptor(filter_dims, num_dim_spatial));
    Tensor<OutDataType> host_output(GetOutputHostTensorDescriptor(output_dims, num_dim_spatial));
    Tensor<OutDataType> device_output(GetOutputHostTensorDescriptor(output_dims, num_dim_spatial));

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

    // do GEMM
    auto conv    = GetConvInstance(num_dim_spatial);
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

    std::size_t flop = ck::conv_util::GetFlops(
        params.N, params.C, params.K, params.filter_spatial_lengths, output_spatial_lengths);
    std::size_t num_btype =
        ck::conv_util::GetBtype<InDataType, WeiDataType, OutDataType>(params.N,
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
        auto verify_f = [&input, &weights, &host_output, &params, &out_device_buf, &device_output](
                            const auto& ref_conv) {
            auto ref_invoker  = ref_conv.MakeInvoker();
            auto ref_argument = ref_conv.MakeArgument(input,
                                                      weights,
                                                      host_output,
                                                      params.conv_filter_strides,
                                                      params.conv_filter_dilations,
                                                      params.input_left_pads,
                                                      params.input_right_pads,
                                                      InElementOp{},
                                                      WeiElementOp{},
                                                      OutElementOp{});

            ref_invoker.Run(ref_argument);
            out_device_buf.FromDevice(device_output.mData.data());
            check_error(host_output, device_output);
        };

        switch(num_dim_spatial)
        {
        case 2: {
            auto ref_conv = ReferenceConvNDFwdInstance<2>();
            verify_f(ref_conv);
            break;
        }
        case 1: {
            auto ref_conv = ReferenceConvNDFwdInstance<1>();
            verify_f(ref_conv);
            break;
        }
        default: {
            throw std::runtime_error("Unsupported number of spatial dimensions provided!");
        }
        }
    }
}
