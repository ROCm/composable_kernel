#include <cstdlib>
#include <half.hpp>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <type_traits>

#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "element_wise_operation.hpp"
#include "device_convnd_fwd_xdl_nhwc_kyxc_nhwk.hpp"
#include "reference_conv_fwd.hpp"

using InDataType  = float;
using WeiDataType = float;
using OutDataType = float;
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
    ck::tensor_operation::device::ConvolutionForwardSpecialization_t::Default;

using DeviceConvFwdBasePtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<InElementOp, WeiElementOp, OutElementOp>;

template <ck::index_t SpatialDims>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::
    DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        // clang-format off
        InDataType,         // 
        WeiDataType,        //
        OutDataType,        //
        AccDataType,        // 
        InElementOp,       // Input Elementwise Operation
        WeiElementOp,      // Weights Elementwise Operation
        OutElementOp,      // Output Elementwise Operation
        ConvFwdDefault,     // ConvForwardSpecialization
        SpatialDims,        // SptialDims
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

template <ck::index_t SpatialDims>
using ReferenceConvNDFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                                WeiDataType,
                                                                                OutDataType,
                                                                                InElementOp,
                                                                                WeiElementOp,
                                                                                OutElementOp,
                                                                                SpatialDims>;

template <typename TensorLayout>
HostTensorDescriptor GetHostTensorDescriptor(const std::vector<std::size_t>& dims,
                                             const TensorLayout& layout)
{
    std::size_t N = dims[0];
    std::size_t C = dims[1];
    // 1D
    if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NCW>::value ||
                 std::is_same<TensorLayout, ck::tensor_layout::convolution::KCX>::value ||
                 std::is_same<TensorLayout, ck::tensor_layout::convolution::NKW>::value)
    {

        return HostTensorDescriptor(std::vector<std::size_t>({N, C, dims[2]}),
                                    std::vector<std::size_t>({C * dims[2], dims[2], 1}));
    }
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NWC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KXC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NWK>::value)
    {
        return HostTensorDescriptor(std::vector<std::size_t>({N, C, dims[2]}),
                                    std::vector<std::size_t>({C * dims[2], 1, C}));
    }
    // 2D
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NCHW>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KCYX>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NKHW>::value)
    {

        return HostTensorDescriptor(
            std::vector<std::size_t>({N, C, dims[2], dims[3]}),
            std::vector<std::size_t>({C * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1}));
    }
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NHWC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KYXC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NHWK>::value)
    {
        return HostTensorDescriptor(
            std::vector<std::size_t>({N, C, dims[2], dims[3]}),
            std::vector<std::size_t>({C * dims[2] * dims[3], 1, dims[3] * C, C}));
    }

    std::stringstream err_msg;
    err_msg << "Unsupported data layout provided: " << layout << "!";
    throw std::runtime_error(err_msg.str());
}

DeviceConvFwdBasePtr GetConvInstance(int spatial_dims)
{
    switch(spatial_dims)
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

std::size_t GetFlops(ck::index_t N,
                     ck::index_t C,
                     ck::index_t K,
                     const std::vector<ck::index_t>& filter_spatial_lengths,
                     const std::vector<ck::index_t>& output_spatial_lengths)
{
    // 2 * N * K * <output spatial lengths product> * C * <filter spatial lengths product>
    return ck::index_t(2) * N * K *
           std::accumulate(std::begin(output_spatial_lengths),
                           std::end(output_spatial_lengths),
                           1,
                           std::multiplies<ck::index_t>()) *
           C *
           std::accumulate(std::begin(filter_spatial_lengths),
                           std::end(filter_spatial_lengths),
                           1,
                           std::multiplies<ck::index_t>());
}

ck::index_t GetBtype(ck::index_t N,
                     ck::index_t C,
                     ck::index_t K,
                     const std::vector<ck::index_t>& input_spatial_lengths,
                     const std::vector<ck::index_t>& filter_spatial_lengths,
                     const std::vector<ck::index_t>& output_spatial_lengths)
{
    // sizeof(InDataType) * (N * C * <input spatial lengths product>) +
    // sizeof(WeiDataType) * (K * C * <filter spatial lengths product>) +
    // sizeof(OutDataType) * (N * K * <output spatial lengths product>);
    return sizeof(InDataType) * (N * C *
                                 std::accumulate(std::begin(input_spatial_lengths),
                                                 std::end(input_spatial_lengths),
                                                 1,
                                                 std::multiplies<ck::index_t>())) +
           sizeof(WeiDataType) * (K * C *
                                  std::accumulate(std::begin(filter_spatial_lengths),
                                                  std::end(filter_spatial_lengths),
                                                  1,
                                                  std::multiplies<ck::index_t>())) +
           sizeof(OutDataType) * (N * K *
                                  std::accumulate(std::begin(output_spatial_lengths),
                                                  std::end(output_spatial_lengths),
                                                  1,
                                                  std::multiplies<ck::index_t>()));
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

struct ConvParams
{
    ConvParams()
        : spatial_dims(2),
          N(128),
          K(256),
          C(192),
          filter_spatial_lengths(2, 3),
          input_spatial_lengths(2, 71),
          conv_filter_strides(2, 2),
          conv_filter_dilations(2, 1),
          input_left_pads(2, 1),
          input_right_pads(2, 1)
    {
    }

    ck::index_t spatial_dims;
    ck::index_t N;
    ck::index_t K;
    ck::index_t C;

    std::vector<ck::index_t> filter_spatial_lengths;
    std::vector<ck::index_t> input_spatial_lengths;

    std::vector<ck::index_t> conv_filter_strides;
    std::vector<ck::index_t> conv_filter_dilations;

    std::vector<ck::index_t> input_left_pads;
    std::vector<ck::index_t> input_right_pads;

    std::vector<ck::index_t> GetOutputSpatialLengths() const
    {
        std::vector<ck::index_t> out_spatial_len(spatial_dims, 0);
        for(ck::index_t i = 0; i < spatial_dims; ++i)
        {
            // XEff = (X - 1) * conv_dilation_w + 1;
            // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
            const ck::index_t idx_eff =
                (filter_spatial_lengths[i] - 1) * conv_filter_dilations[i] + 1;
            out_spatial_len[i] =
                (input_spatial_lengths[i] + input_left_pads[i] + input_right_pads[i] - idx_eff) /
                    conv_filter_strides[i] +
                1;
        }
        return out_spatial_len;
    }
};

ConvParams ParseConvParams(int spatial_dims, int argc, char* argv[])
{
    // (N, K, C) + spatial_dims * 6 (filter, input, strides, dilations, pad left, pad right)
    int conv_args     = 3 + spatial_dims * 6;
    int cmdline_nargs = conv_args + 5;
    if(cmdline_nargs != argc)
    {
        PrintUseMsg();
        exit(0);
    }

    ConvParams params;
    int arg_idx = 5;

    params.spatial_dims = spatial_dims;
    params.N            = std::stoi(argv[arg_idx++]);
    params.K            = std::stoi(argv[arg_idx++]);
    params.C            = std::stoi(argv[arg_idx++]);

    params.filter_spatial_lengths.resize(spatial_dims);
    for(int i = 0; i < spatial_dims; ++i)
    {
        params.filter_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_spatial_lengths.resize(spatial_dims);
    for(int i = 0; i < spatial_dims; ++i)
    {
        params.input_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_strides.resize(spatial_dims);
    for(int i = 0; i < spatial_dims; ++i)
    {
        params.conv_filter_strides[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_dilations.resize(spatial_dims);
    for(int i = 0; i < spatial_dims; ++i)
    {
        params.conv_filter_dilations[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_left_pads.resize(spatial_dims);
    for(int i = 0; i < spatial_dims; ++i)
    {
        params.input_left_pads[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_right_pads.resize(spatial_dims);
    for(int i = 0; i < spatial_dims; ++i)
    {
        params.input_right_pads[i] = std::stoi(argv[arg_idx++]);
    }

    return params;
}

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;
    int spatial_dims     = 2;

    ConvParams params;

    if(argc >= 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
    }

    if(argc >= 5)
    {
        spatial_dims = std::stoi(argv[4]);
        params       = ParseConvParams(spatial_dims, argc, argv);
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

    Tensor<InDataType> input(GetHostTensorDescriptor(input_dims, InLayout{}));
    Tensor<WeiDataType> weights(GetHostTensorDescriptor(filter_dims, WeiLayout{}));
    Tensor<OutDataType> host_output(GetHostTensorDescriptor(output_dims, OutLayout{}));
    Tensor<OutDataType> device_output(GetHostTensorDescriptor(output_dims, OutLayout{}));

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
    auto conv    = GetConvInstance(spatial_dims);
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

    ck::index_t flop = GetFlops(
        params.N, params.C, params.K, params.filter_spatial_lengths, output_spatial_lengths);
    ck::index_t num_btype = GetBtype(params.N,
                                     params.C,
                                     params.K,
                                     params.input_spatial_lengths,
                                     params.filter_spatial_lengths,
                                     output_spatial_lengths);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

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

        switch(spatial_dims)
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
