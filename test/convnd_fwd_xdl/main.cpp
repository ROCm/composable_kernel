#include <algorithm>
#include <cstdlib>
#include <half.hpp>
#include <iostream>
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

// Forward declarations for conv instances.

using DeviceConvFwdNoOpPtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv1d_fwd_instance {

void add_device_conv1d_fwd_xdl_nwc_kxc_nwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv1d_fwd_xdl_nwc_kxc_nwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv1d_fwd_xdl_nwc_kxc_nwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv1d_fwd_xdl_nwc_kxc_nwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace device_conv1d_fwd_instance
namespace device_conv2d_fwd_instance {

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace device_conv2d_fwd_instance
namespace device_conv3d_fwd_instance {

void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace device_conv3d_fwd_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace {

using bhalf_t = test_util::bhalf_t;

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
                  test_util::check_err(
                      output.mData, host_output.mData, "Error: incorrect results!", 1e-5f, 1e-4f);
            hipGetErrorString(
                hipMemset(out_device_buf.GetDeviceBuffer(), 0, out_device_buf.mMemSize));
        }
    }
    return res;
}

bool TestConv1DNWC()
{
    bool res{true};
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 1;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 4;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{16};
    params.conv_filter_strides    = std::vector<ck::index_t>{1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1};
    params.input_left_pads        = std::vector<ck::index_t>{1};
    params.input_right_pads       = std::vector<ck::index_t>{1};

    auto host_tensors            = GetHostTensors<float,
                                       float,
                                       float,
                                       ck::tensor_layout::convolution::NWC,
                                       ck::tensor_layout::convolution::KXC,
                                       ck::tensor_layout::convolution::NWK>(params);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& host_output   = std::get<2>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    RunReferenceConv<1>(params, input, weights, host_output);
    RunConv<1>(params, input, weights, device_output);
    res = res &&
          test_util::check_err(
              device_output.mData, host_output.mData, "Error: incorrect results!", 1e-5f, 1e-4f);

    return res;
}

bool TestConv2DNHWC()
{
    bool res{true};
    ck::conv_util::ConvParams params;
    params.N                     = 2;
    params.K                     = 16;
    params.C                     = 4;
    params.input_spatial_lengths = std::vector<ck::index_t>{16, 16};
    params.conv_filter_strides   = std::vector<ck::index_t>{1, 1};

    auto host_tensors            = GetHostTensors(params);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& host_output   = std::get<2>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    RunReferenceConv<2>(params, input, weights, host_output);
    RunConv<2>(params, input, weights, device_output);
    res = res &&
          test_util::check_err(
              device_output.mData, host_output.mData, "Error: incorrect results!", 1e-5f, 1e-4f);

    return res;
}

bool TestConv3DNDHWC()
{
    bool res{true};
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 3;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 4;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{16, 16, 16};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    auto host_tensors            = GetHostTensors<float,
                                       float,
                                       float,
                                       ck::tensor_layout::convolution::NDHWC,
                                       ck::tensor_layout::convolution::KZYXC,
                                       ck::tensor_layout::convolution::NDHWK>(params);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& host_output   = std::get<2>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    RunReferenceConv<3>(params, input, weights, host_output);
    RunConv<3>(params, input, weights, device_output);
    res = res &&
          test_util::check_err(
              device_output.mData, host_output.mData, "Error: incorrect results!", 1e-5f, 1e-4f);

    return res;
}

bool TestConv3DNDHWC2GBInput()
{
    // >2GB Input
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 3;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 32;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{32, 1000, 1000};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    auto host_tensors            = GetHostTensors<float,
                                       float,
                                       float,
                                       ck::tensor_layout::convolution::NDHWC,
                                       ck::tensor_layout::convolution::KZYXC,
                                       ck::tensor_layout::convolution::NDHWK>(params);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    try
    {
        RunConv<3>(params, input, weights, device_output);
    }
    catch(const std::runtime_error& err)
    {
        std::string err_msg{"Error! device_conv with the specified compilation parameters does "
                            "not support this Conv problem"};
        if(err.what() != err_msg)
        {
            return false;
        }
        return true;
    }
    std::cout << "Error: Failure checking oversized tensor!" << std::endl;
    return false;
}

bool TestConv3DNDHWC2GBFilters()
{
    // >2GB Filters
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 3;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 32;
    params.filter_spatial_lengths = std::vector<ck::index_t>{4, 1000, 1000};
    params.input_spatial_lengths  = std::vector<ck::index_t>{16, 16, 16};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    auto host_tensors            = GetHostTensors<float,
                                       float,
                                       float,
                                       ck::tensor_layout::convolution::NDHWC,
                                       ck::tensor_layout::convolution::KZYXC,
                                       ck::tensor_layout::convolution::NDHWK>(params);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    try
    {
        RunConv<3>(params, input, weights, device_output);
    }
    catch(const std::runtime_error& err)
    {
        std::string err_msg{"Error! device_conv with the specified compilation parameters does "
                            "not support this Conv problem"};
        if(err.what() != err_msg)
        {
            return false;
        }
        return true;
    }
    std::cout << "Error: Failure checking oversized tensor!" << std::endl;
    return false;
}

bool TestConv3DNDHWC2GBOutput()
{
    // >2GB Output
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 3;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 2;
    params.filter_spatial_lengths = std::vector<ck::index_t>{1, 1, 1};
    params.input_spatial_lengths  = std::vector<ck::index_t>{1000, 1000, 30};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{2, 2, 2};
    params.input_right_pads       = std::vector<ck::index_t>{2, 2, 2};

    auto host_tensors            = GetHostTensors<float,
                                       float,
                                       float,
                                       ck::tensor_layout::convolution::NDHWC,
                                       ck::tensor_layout::convolution::KZYXC,
                                       ck::tensor_layout::convolution::NDHWK>(params);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    try
    {
        RunConv<3>(params, input, weights, device_output);
    }
    catch(const std::runtime_error& err)
    {
        std::string err_msg{"Error! device_conv with the specified compilation parameters does "
                            "not support this Conv problem"};
        if(err.what() != err_msg)
        {
            return false;
        }
        return true;
    }
    std::cout << "Error: Failure checking oversized tensor!" << std::endl;
    return false;
}

template <typename T>
bool TestConv1DNWCInstances()
{
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 1;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{71};
    params.conv_filter_strides    = std::vector<ck::index_t>{2};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1};
    params.input_left_pads        = std::vector<ck::index_t>{1};
    params.input_right_pads       = std::vector<ck::index_t>{1};

    auto host_tensors        = GetHostTensors<T,
                                       T,
                                       T,
                                       ck::tensor_layout::convolution::NWC,
                                       ck::tensor_layout::convolution::KXC,
                                       ck::tensor_layout::convolution::NWK>(params);
    const Tensor<T>& input   = std::get<0>(host_tensors);
    const Tensor<T>& weights = std::get<1>(host_tensors);
    Tensor<T>& host_output   = std::get<2>(host_tensors);
    Tensor<T>& device_output = std::get<3>(host_tensors);

    RunReferenceConv<1>(params, input, weights, host_output);

    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv1d_fwd_instance::
        add_device_conv1d_fwd_xdl_nwc_kxc_nwk_bf16_instances(conv_ptrs);

    return RunConvInstances<1>(params, conv_ptrs, input, weights, device_output, host_output);
}
bool TestConv1DNWCBF16Instances() { return TestConv1DNWCInstances<bhalf_t>(); }

bool TestConv1DNWCF16Instances() { return TestConv1DNWCInstances<ck::half_t>(); }

bool TestConv1DNWCF32Instances() { return TestConv1DNWCInstances<float>(); }

bool TestConv1DNWCInt8Instances() { return TestConv1DNWCInstances<int8_t>(); }

template <typename T>
bool TestConv2DNHWCInstances()
{
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 2;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{71, 71};
    params.conv_filter_strides    = std::vector<ck::index_t>{2, 2};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1};

    auto host_tensors        = GetHostTensors<T,
                                       T,
                                       T,
                                       ck::tensor_layout::convolution::NHWC,
                                       ck::tensor_layout::convolution::KYXC,
                                       ck::tensor_layout::convolution::NHWK>(params);
    const Tensor<T>& input   = std::get<0>(host_tensors);
    const Tensor<T>& weights = std::get<1>(host_tensors);
    Tensor<T>& host_output   = std::get<2>(host_tensors);
    Tensor<T>& device_output = std::get<3>(host_tensors);

    RunReferenceConv<2>(params, input, weights, host_output);

    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv2d_fwd_instance::
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(conv_ptrs);

    return RunConvInstances<2>(params, conv_ptrs, input, weights, device_output, host_output);
}

bool TestConv2DNHWCBF16Instances() { return TestConv2DNHWCInstances<bhalf_t>(); }

bool TestConv2DNHWCF16Instances() { return TestConv2DNHWCInstances<ck::half_t>(); }

bool TestConv2DNHWCF32Instances() { return TestConv2DNHWCInstances<float>(); }

bool TestConv2DNHWCInt8Instances() { return TestConv2DNHWCInstances<int8_t>(); }

template <typename T>
bool TestConv3DNDHWCInstances()
{
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 3;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{71, 71, 71};
    params.conv_filter_strides    = std::vector<ck::index_t>{2, 2, 2};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    auto host_tensors        = GetHostTensors<T,
                                       T,
                                       T,
                                       ck::tensor_layout::convolution::NDHWC,
                                       ck::tensor_layout::convolution::KZYXC,
                                       ck::tensor_layout::convolution::NDHWK>(params);
    const Tensor<T>& input   = std::get<0>(host_tensors);
    const Tensor<T>& weights = std::get<1>(host_tensors);
    Tensor<T>& host_output   = std::get<2>(host_tensors);
    Tensor<T>& device_output = std::get<3>(host_tensors);

    RunReferenceConv<3>(params, input, weights, host_output);

    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv3d_fwd_instance::
        add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(conv_ptrs);

    return RunConvInstances<3>(params, conv_ptrs, input, weights, device_output, host_output);
}

bool TestConv3DNDHWCBF16Instances() { return TestConv3DNDHWCInstances<bhalf_t>(); }

bool TestConv3DNDHWCF16Instances() { return TestConv3DNDHWCInstances<ck::half_t>(); }

bool TestConv3DNDHWCF32Instances() { return TestConv3DNDHWCInstances<float>(); }

bool TestConv3DNDHWCInt8Instances() { return TestConv3DNDHWCInstances<int8_t>(); }

} // anonymous namespace

int main()
{
    bool res{true};
    res = TestConv1DNWC();
    std::cout << "TestConv1DNWC ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv2DNHWC();
    std::cout << "TestConv2DNHWC ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv3DNDHWC();
    std::cout << "TestConv3DNDHWC ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = TestConv3DNDHWC2GBInput();
    std::cout << "TestConv3DNDHWC2GBInput ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv3DNDHWC2GBFilters();
    std::cout << "TestConv3DNDHWC2GBFilters ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv3DNDHWC2GBOutput();
    std::cout << "TestConv3DNDHWC2GBOutput ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = TestConv1DNWCBF16Instances();
    std::cout << "TestConv1DNWCBF16Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv1DNWCF16Instances();
    std::cout << "TestConv1DNWCF16Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv1DNWCF32Instances();
    std::cout << "TestConv1DNWCF32Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv1DNWCInt8Instances();
    std::cout << "TestConv1DNWCInt8Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = TestConv2DNHWCBF16Instances();
    std::cout << "TestConv2DNHWCBF16Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv2DNHWCF16Instances();
    std::cout << "TestConv2DNHWCF16Instances ....." << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv2DNHWCF32Instances();
    std::cout << "TestConv2DNHWCF32Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv2DNHWCInt8Instances();
    std::cout << "TestConv2DNHWCInt8Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = TestConv3DNDHWCBF16Instances();
    std::cout << "TestConv3DNDHWCBF16Instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = TestConv3DNDHWCF16Instances();
    std::cout << "TestConv3DNDHWCF16Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv3DNDHWCF32Instances();
    std::cout << "TestConv3DNDHWCF32Instances ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv3DNDHWCInt8Instances();
    std::cout << "TestConv3DNDHWCInt8Instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
}
