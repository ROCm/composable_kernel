#ifndef TEST_CONV_UTIL_HPP
#define TEST_CONV_UTIL_HPP

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "config.hpp"
#include "conv_utils.hpp"
#include "device.hpp"
#include "device_tensor.hpp"
#include "device_convnd_fwd_xdl_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation.hpp"
#include "host_tensor.hpp"
#include "reference_conv_fwd.hpp"
#include "tensor_layout.hpp"
#include "test_util.hpp"

namespace {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization_t::Default;

template <ck::index_t SpatialDims, typename InDataType, typename WeiDataType, typename OutDataType>
using DeviceConvNDFwdInstance = ck::tensor_operation::device::
    DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<
        // clang-format off
        InDataType,         // 
        WeiDataType,        //
        OutDataType,        //
        InDataType,         // 
        InElementOp,        // Input Elementwise Operation
        WeiElementOp,       // Weights Elementwise Operation
        OutElementOp,       // Output Elementwise Operation
        ConvFwdDefault,     // ConvForwardSpecialization
        SpatialDims,        // SptialDims
        64,                 // BlockSize
        16,                 // MPerBlock
        16,                 // NPerBlock
        4,                  // K0PerBlock
        1,                  // K1                                           
        16,                 // MPerXDL
        16,                 // NPerXDL
        1,                  // MXdlPerWave
        1,                  // NXdlPerWave
        S<1, 16, 1>,        // ABlockTransferThreadClusterLengths_K0_M_K1
        S<1, 0, 2>,         // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // ABlockTransferSrcAccessOrder
        2,                  // ABlockTransferSrcVectorDim
        1,                  // ABlockTransferSrcScalarPerVector
        1,                  // ABlockTransferDstScalarPerVector_K1
        true,               // ABlockLdsAddExtraM
        S<1, 16, 1>,        // BBlockTransferThreadClusterLengths_K0_N_K1
        S<1, 0, 2>,         // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,         // BBlockTransferSrcAccessOrder
        2,                  // BBlockTransferSrcVectorDim
        1,                  // BBlockTransferSrcScalarPerVector
        1,                  // BBlockTransferDstScalarPerVector_K1
        true,               // BBlockTransferAddExtraN
        7,                  // CThreadTransferSrcDstVectorDim
        1>;                 // CThreadTransferDstScalarPerVector
// clang-format on

} // namespace

namespace test {
namespace conv {

using DeviceConvFwdNoOpPtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough>;

template <typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float,
          typename InLayout    = ck::tensor_layout::convolution::NHWC,
          typename WeiLayout   = ck::tensor_layout::convolution::KYXC,
          typename OutLayout   = ck::tensor_layout::convolution::NHWK>
auto GetHostTensors(const ck::conv_util::ConvParams& params)
{
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

    Tensor<InDataType> input(ck::conv_util::GetHostTensorDescriptor(input_dims, InLayout{}));
    Tensor<WeiDataType> weights(ck::conv_util::GetHostTensorDescriptor(filter_dims, WeiLayout{}));
    Tensor<OutDataType> host_output(
        ck::conv_util::GetHostTensorDescriptor(output_dims, OutLayout{}));
    Tensor<OutDataType> device_output(
        ck::conv_util::GetHostTensorDescriptor(output_dims, OutLayout{}));

    std::generate(input.begin(), input.end(), [n = 0]() mutable { return InDataType(n++ * 0.1f); });
    std::fill(weights.begin(), weights.end(), WeiDataType(0.5f));
    std::fill(host_output.begin(), host_output.end(), OutDataType(0.f));
    std::fill(device_output.begin(), device_output.end(), OutDataType(0.f));

    return std::make_tuple(input, weights, host_output, device_output);
}

template <ck::index_t NDim,
          typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float>
void RunReferenceConv(const ck::conv_util::ConvParams& params,
                      const Tensor<InDataType>& input,
                      const Tensor<WeiDataType>& weights,
                      Tensor<OutDataType>& output)
{
    auto ref_conv     = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                 WeiDataType,
                                                                 OutDataType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 OutElementOp,
                                                                 NDim>();
    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(input,
                                              weights,
                                              output,
                                              params.conv_filter_strides,
                                              params.conv_filter_dilations,
                                              params.input_left_pads,
                                              params.input_right_pads,
                                              InElementOp{},
                                              WeiElementOp{},
                                              OutElementOp{});

    ref_invoker.Run(ref_argument);
}

template <ck::index_t NDim,
          typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float>
void RunConv(const ck::conv_util::ConvParams& params,
             const Tensor<InDataType>& input,
             const Tensor<WeiDataType>& weights,
             Tensor<OutDataType>& output)
{
    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weights.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * output.mDesc.GetElementSpace());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weights.mData.data());
    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();

    auto conv     = DeviceConvNDFwdInstance<NDim, InDataType, WeiDataType, OutDataType>();
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
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

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "Error! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    invoker.Run(argument);
    out_device_buf.FromDevice(output.mData.data());
}

template <ck::index_t NDim,
          typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float>
bool RunConvInstances(const ck::conv_util::ConvParams& params,
                      const std::vector<DeviceConvFwdNoOpPtr>& conv_ptrs,
                      const Tensor<InDataType>& input,
                      const Tensor<WeiDataType>& weights,
                      Tensor<OutDataType>& output,
                      const Tensor<OutDataType>& host_output)
{
    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weights.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * output.mDesc.GetElementSpace());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weights.mData.data());
    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();

    bool res{true};
    for(auto& conv_ptr : conv_ptrs)
    {
        auto invoker  = conv_ptr->MakeInvokerPointer();
        auto argument = conv_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
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

        if(conv_ptr->IsSupportedArgument(argument.get()))
        {
            invoker->Run(argument.get());
            out_device_buf.FromDevice(output.mData.data());
            res = res &&
                  test::check_err(
                      output.mData, host_output.mData, "Error: incorrect results!", 1e-5f, 1e-4f);
            hipGetErrorString(
                hipMemset(out_device_buf.GetDeviceBuffer(), 0, out_device_buf.mMemSize));
        }
    }
    return res;
}

} // namespace conv
} // namespace test

#endif
