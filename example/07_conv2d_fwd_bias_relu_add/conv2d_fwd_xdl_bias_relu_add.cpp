#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "check_err.hpp"
#include "config.hpp"
#include "conv_util.hpp"
#include "device.hpp"
#include "device_conv2d_fwd_xdl_c_shuffle_bias_activation_add_nhwc_kyxc_nhwk.hpp"
#include "device_tensor.hpp"
#include "element_wise_operation.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "reference_conv_fwd_bias_activation_add.hpp"
#include "tensor_layout.hpp"

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

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddReluAdd;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

// clang-format off
using DeviceConvFwdInstance = ck::tensor_operation::device::
    DeviceConv2dFwdXdl_C_Shuffle_Bias_Activation_Add_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        InDataType,              // InDataType
        WeiDataType,             // WeiDataType
        OutDataType,             // OutDataType
        AccDataType,             // AccDataType
        InElementOp,             // InElementwiseOperation
        WeiElementOp,            // WeiElementwiseOperation
        OutElementOp,            // OutElementwiseOperation
        ConvFwdDefault,          // ConvForwardSpecialization
        256,                     // BlockSize
        128,                     // MPerBlock
        256,                     // NPerBlock
        4,                       // K0PerBlock
        8,                       // K1
        32,                      // MPerXdl
        32,                      // NPerXdl
        2,                       // MXdlPerWave
        4,                       // NXdlPerWave
        S<4, 64, 1>,             // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,              // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,              // ABlockTransferSrcAccessOrder
        2,                       // ABlockTransferSrcVectorDim
        8,                       // ABlockTransferSrcScalarPerVector
        8,                       // ABlockTransferDstScalarPerVector_K1
        true,                    // ABlockLdsAddExtraM
        S<4, 64, 1>,             // BBlockTransferThreadClusterLengths_K0_N_K1
        S<1, 0, 2>,              // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,              // BBlockTransferSrcAccessOrder
        2,                       // BBlockTransferSrcVectorDim
        8,                       // BBlockTransferSrcScalarPerVector
        8,                       // BBlockTransferDstScalarPerVector_K1
        true,                    // BBlockLdsAddExtraN
        1,                       // CShuffleMXdlPerWavePerShuffle
        1,                       // CShuffleNXdlPerWavePerShuffle
        S<1, 1, 32, 1, 1, 8>,    // CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
        8>;                      // CBlockTransferScalarPerVector_NWaveNPerXdl
// clang-format on

using ReferenceConvFwdInstance =
    ck::tensor_operation::host::ReferenceConvFwd_Bias_Activation_Add<InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     OutElementOp>;

void PrintUseMsg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=n0, 1=yes)\n"
              << "Following arguments:\n"
              << " N, K, C, \n"
              << " <filter spatial dimensions>, (ie Y, X for 2D)\n"
              << " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
              << " <strides>, (ie Sy, Sx for 2D)\n"
              << " <dilations>, (ie Dy, Dx for 2D)\n"
              << " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
              << " <right padding>, (ie RightPy, RightPx for 2D)\n"
              << std::endl;
}

ck::utils::conv::ConvParams ParseConvParams(int argc, char* argv[])
{
    // (N, K, C) + num_dim_spatial * 6 (filter, input, strides, dilations, pad left, pad right)
    int num_dim_spatial = 2;
    int conv_args       = 3 + num_dim_spatial * 6;
    int cmdline_nargs   = conv_args + 4;
    if(cmdline_nargs != argc)
    {
        PrintUseMsg();
        exit(0);
    }

    ck::utils::conv::ConvParams params;
    int arg_idx = 4;

    params.num_dim_spatial_ = num_dim_spatial;
    params.N_               = std::stoi(argv[arg_idx++]);
    params.K_               = std::stoi(argv[arg_idx++]);
    params.C_               = std::stoi(argv[arg_idx++]);

    params.filter_spatial_lengths_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.filter_spatial_lengths_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_spatial_lengths_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_spatial_lengths_[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_strides_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_strides_[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_dilations_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_dilations_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_left_pads_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_left_pads_[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_right_pads_.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_right_pads_[i] = std::stoi(argv[arg_idx++]);
    }

    return params;
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    using namespace ck::utils::conv;

    bool do_verification      = true;
    int init_method           = 1;
    bool time_kernel          = false;
    const int num_dim_spatial = 2;

    ck::utils::conv::ConvParams params;

    if(argc >= 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }

    if(argc >= 5)
    {
        params = ParseConvParams(argc, argv);
    }

    std::vector<std::size_t> input_dims{static_cast<std::size_t>(params.N_),
                                        static_cast<std::size_t>(params.C_)};
    input_dims.insert(std::end(input_dims),
                      std::begin(params.input_spatial_lengths_),
                      std::end(params.input_spatial_lengths_));

    std::vector<std::size_t> filter_dims{static_cast<std::size_t>(params.K_),
                                         static_cast<std::size_t>(params.C_)};
    filter_dims.insert(std::end(filter_dims),
                       std::begin(params.filter_spatial_lengths_),
                       std::end(params.filter_spatial_lengths_));

    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();
    std::vector<std::size_t> output_dims{static_cast<std::size_t>(params.N_),
                                         static_cast<std::size_t>(params.K_)};
    output_dims.insert(std::end(output_dims),
                       std::begin(output_spatial_lengths),
                       std::end(output_spatial_lengths));

    Tensor<InDataType> input(get_input_host_tensor_descriptor(input_dims, num_dim_spatial));
    Tensor<WeiDataType> weights(get_filters_host_tensor_descriptor(filter_dims, num_dim_spatial));
    Tensor<OutDataType> host_output(
        get_output_host_tensor_descriptor(output_dims, num_dim_spatial));
    Tensor<OutDataType> device_output(
        get_output_host_tensor_descriptor(output_dims, num_dim_spatial));

    // bias: assume contiguous 1d vector
    Tensor<OutDataType> bias(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(params.K_)})));

    // residual: assume same layout as output tensor
    Tensor<OutDataType> residual(get_output_host_tensor_descriptor(output_dims, num_dim_spatial));

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weights: " << weights.mDesc << std::endl;
    std::cout << "output: " << host_output.mDesc << std::endl;
    std::cout << "bias: " << bias.mDesc << std::endl;
    std::cout << "residual: " << residual.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-2, 2});
        weights.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-2, 2});
        bias.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-2, 2});
        residual.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-2, 2});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        weights.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        residual.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weights.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * device_output.mDesc.GetElementSpace());
    DeviceMem bias_device_buf(sizeof(OutDataType) * bias.mDesc.GetElementSpace());
    DeviceMem resi_device_buf(sizeof(OutDataType) * residual.mDesc.GetElementSpace());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weights.mData.data());
    bias_device_buf.ToDevice(bias.mData.data());
    resi_device_buf.ToDevice(residual.mData.data());

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    auto conv    = DeviceConvFwdInstance{};
    auto invoker = conv.MakeInvoker();
    auto argument =
        conv.MakeArgument(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                          static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                          static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                          static_cast<const OutDataType*>(bias_device_buf.GetDeviceBuffer()),
                          static_cast<const OutDataType*>(resi_device_buf.GetDeviceBuffer()),
                          params.N_,
                          params.K_,
                          params.C_,
                          params.input_spatial_lengths_,
                          params.filter_spatial_lengths_,
                          output_spatial_lengths,
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
            "wrong! device operator with the specified compilation parameters does "
            "not support this problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = get_flops(
        params.N_, params.C_, params.K_, params.filter_spatial_lengths_, output_spatial_lengths);
    std::size_t num_btype =
        get_btype<InDataType, WeiDataType, OutDataType>(params.N_,
                                                        params.C_,
                                                        params.K_,
                                                        params.input_spatial_lengths_,
                                                        params.filter_spatial_lengths_,
                                                        output_spatial_lengths) +
        sizeof(OutDataType) * (params.K_) +
        sizeof(OutDataType) *
            (params.N_ * params.K_ * output_spatial_lengths[0] * output_spatial_lengths[1]);

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        auto ref_conv    = ReferenceConvFwdInstance{};
        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weights,
                                                  host_output,
                                                  bias,
                                                  residual,
                                                  params.conv_filter_strides_,
                                                  params.conv_filter_dilations_,
                                                  params.input_left_pads_,
                                                  params.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);
        out_device_buf.FromDevice(device_output.mData.data());
        return ck::utils::check_err(device_output.mData, host_output.mData) ? 0 : 1;
    }

    return 0;
}
