// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

// CPU references
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"

// GPU references
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_data_gpu.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_weight_gpu.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "common_test_params.hpp"

namespace ck {
namespace test {

enum class ConvKernelType
{
    Forward,
    BackwardData,
    BackwardWeight
};

// Helper function to initialize and copy a tensor to device
template <typename DataType>
void initialize_and_copy_tensor(Tensor<DataType>& host_tensor, DeviceMem& device_mem)
{
    host_tensor.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
    device_mem.ToDevice(host_tensor.mData.data());
}

// Helper to get layout types based on NDimSpatial
template <index_t NDimSpatial>
struct ConvLayoutTypes
{
    using InLayout  = std::conditional_t<NDimSpatial == 3,
                                         tensor_layout::convolution::GNCDHW,
                                         std::conditional_t<NDimSpatial == 2,
                                                            tensor_layout::convolution::GNCHW,
                                                            tensor_layout::convolution::GNCW>>;
    using WeiLayout = std::conditional_t<NDimSpatial == 3,
                                         tensor_layout::convolution::GKCZYX,
                                         std::conditional_t<NDimSpatial == 2,
                                                            tensor_layout::convolution::GKCYX,
                                                            tensor_layout::convolution::GKCX>>;
    using OutLayout = std::conditional_t<NDimSpatial == 3,
                                         tensor_layout::convolution::GNKDHW,
                                         std::conditional_t<NDimSpatial == 2,
                                                            tensor_layout::convolution::GNKHW,
                                                            tensor_layout::convolution::GNKW>>;
};

// Forward convolution implementation
template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_fwd_impl(const ck::utils::conv::ConvParam& params,
                        const Tensor<InDataType>& input_cpu,
                        const Tensor<WeiDataType>& weight_cpu,
                        const std::vector<index_t>& out_lengths_cpu,
                        DeviceMem& input_dev,
                        DeviceMem& weight_dev,
                        DeviceMem& output_dev)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;
    using Layouts      = ConvLayoutTypes<NDimSpatial>;

    // Call GPU reference with ConvParam directly
    ref::naive_conv_fwd<typename Layouts::InLayout,
                        typename Layouts::WeiLayout,
                        typename Layouts::OutLayout,
                        InDataType,
                        WeiDataType,
                        OutDataType,
                        InElementOp,
                        WeiElementOp,
                        OutElementOp>(
        reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
        reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
        reinterpret_cast<OutDataType*>(output_dev.GetDeviceBuffer()),
        params,
        nullptr);

    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.conv_filter_strides_.begin(),
                                           params.conv_filter_strides_.end());
    std::vector<long_index_t> dilations_long(params.conv_filter_dilations_.begin(),
                                             params.conv_filter_dilations_.end());
    std::vector<long_index_t> pads_long(params.input_left_pads_.begin(),
                                        params.input_left_pads_.end());

    Tensor<InDataType> input_ref   = input_cpu;
    Tensor<WeiDataType> weight_ref = weight_cpu;
    Tensor<OutDataType> output_ref(out_lengths_cpu);

    auto ref_conv    = tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                InDataType,
                                                                WeiDataType,
                                                                OutDataType,
                                                                InElementOp,
                                                                WeiElementOp,
                                                                OutElementOp>();
    auto ref_invoker = ref_conv.MakeInvoker();
    auto ref_arg     = ref_conv.MakeArgument(input_ref,
                                         weight_ref,
                                         output_ref,
                                         strides_long,
                                         dilations_long,
                                         pads_long,
                                         pads_long,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
    ref_invoker.Run(ref_arg);

    // Copy result from device and compare
    Tensor<OutDataType> output_gpu(out_lengths_cpu);
    output_dev.FromDevice(output_gpu.mData.data());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Compare results
    return ck::utils::check_err(output_gpu, output_ref);
}

// Backward data convolution implementation
template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_bwd_data_impl(const ck::utils::conv::ConvParam& params,
                             const Tensor<WeiDataType>& weight_cpu,
                             const Tensor<OutDataType>& output_cpu,
                             const std::vector<index_t>& in_lengths_cpu,
                             DeviceMem& weight_dev,
                             DeviceMem& output_dev,
                             DeviceMem& input_dev)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;
    using Layouts      = ConvLayoutTypes<NDimSpatial>;

    // Call GPU reference with ConvParam directly
    ref::naive_conv_bwd_data<typename Layouts::InLayout,
                             typename Layouts::WeiLayout,
                             typename Layouts::OutLayout,
                             InDataType,
                             WeiDataType,
                             OutDataType,
                             InElementOp,
                             WeiElementOp,
                             OutElementOp>(
        reinterpret_cast<InDataType*>(input_dev.GetDeviceBuffer()),
        reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
        reinterpret_cast<const OutDataType*>(output_dev.GetDeviceBuffer()),
        params,
        nullptr);

    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.conv_filter_strides_.begin(),
                                           params.conv_filter_strides_.end());
    std::vector<long_index_t> dilations_long(params.conv_filter_dilations_.begin(),
                                             params.conv_filter_dilations_.end());
    std::vector<long_index_t> pads_long(params.input_left_pads_.begin(),
                                        params.input_left_pads_.end());

    Tensor<InDataType> input_ref(in_lengths_cpu);
    Tensor<WeiDataType> weight_ref = weight_cpu;
    Tensor<OutDataType> output_ref = output_cpu;

    auto ref_conv    = tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                    InDataType,
                                                                    WeiDataType,
                                                                    OutDataType,
                                                                    InElementOp,
                                                                    WeiElementOp,
                                                                    OutElementOp>();
    auto ref_invoker = ref_conv.MakeInvoker();
    auto ref_arg     = ref_conv.MakeArgument(input_ref,
                                         weight_ref,
                                         output_ref,
                                         strides_long,
                                         dilations_long,
                                         pads_long,
                                         pads_long,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
    ref_invoker.Run(ref_arg);

    // Copy result from device and compare
    Tensor<InDataType> input_gpu(in_lengths_cpu);
    input_dev.FromDevice(input_gpu.mData.data());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Compare results
    return ck::utils::check_err(input_gpu, input_ref);
}

// Backward weight convolution implementation
template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_bwd_weight_impl(const ck::utils::conv::ConvParam& params,
                               const Tensor<InDataType>& input_cpu,
                               const Tensor<OutDataType>& output_cpu,
                               const std::vector<index_t>& wei_lengths_cpu,
                               DeviceMem& input_dev,
                               DeviceMem& output_dev,
                               DeviceMem& weight_dev)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;
    using Layouts      = ConvLayoutTypes<NDimSpatial>;

    // Call GPU reference with ConvParam directly
    ref::naive_conv_bwd_weight<typename Layouts::InLayout,
                               typename Layouts::WeiLayout,
                               typename Layouts::OutLayout,
                               InDataType,
                               WeiDataType,
                               OutDataType,
                               InElementOp,
                               WeiElementOp,
                               OutElementOp>(
        reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
        reinterpret_cast<WeiDataType*>(weight_dev.GetDeviceBuffer()),
        reinterpret_cast<const OutDataType*>(output_dev.GetDeviceBuffer()),
        params,
        nullptr);

    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.conv_filter_strides_.begin(),
                                           params.conv_filter_strides_.end());
    std::vector<long_index_t> dilations_long(params.conv_filter_dilations_.begin(),
                                             params.conv_filter_dilations_.end());
    std::vector<long_index_t> pads_long(params.input_left_pads_.begin(),
                                        params.input_left_pads_.end());

    Tensor<InDataType> input_ref = input_cpu;
    Tensor<WeiDataType> weight_ref(wei_lengths_cpu);
    Tensor<OutDataType> output_ref = output_cpu;

    auto ref_conv    = tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                      InDataType,
                                                                      WeiDataType,
                                                                      OutDataType,
                                                                      InElementOp,
                                                                      WeiElementOp,
                                                                      OutElementOp>();
    auto ref_invoker = ref_conv.MakeInvoker();
    auto ref_arg     = ref_conv.MakeArgument(input_ref,
                                         weight_ref,
                                         output_ref,
                                         strides_long,
                                         dilations_long,
                                         pads_long,
                                         pads_long,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
    ref_invoker.Run(ref_arg);

    // Copy result from device and compare
    Tensor<WeiDataType> weight_gpu(wei_lengths_cpu);
    weight_dev.FromDevice(weight_gpu.mData.data());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Compare results
    return ck::utils::check_err(weight_gpu, weight_ref);
}

// Main test function - dispatches to specific implementations
template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_gpu_ref(const ck::utils::conv::ConvParam& params, ConvKernelType kernel_type)
{
    // Calculate dimensions
    const index_t N = params.N_;
    const index_t K = params.K_;
    const index_t C = params.C_;
    const index_t G = params.G_;

    // C and K in params are total channels, divide by G for per-group
    const index_t C_per_group = C / G;
    const index_t K_per_group = K / G;

    // Create tensors in CPU layout (GNCDHW/GKCZYX/GNKDHW)
    // The wrappers will handle transformations to/from naive kernel format
    std::vector<index_t> in_lengths = {G, N, C_per_group};
    for(auto d : params.input_spatial_lengths_)
        in_lengths.push_back(static_cast<index_t>(d));

    std::vector<index_t> wei_lengths = {G, K_per_group, C_per_group};
    for(auto d : params.filter_spatial_lengths_)
        wei_lengths.push_back(static_cast<index_t>(d));

    std::vector<index_t> out_lengths = {G, N, K_per_group};
    for(auto d : params.output_spatial_lengths_)
        out_lengths.push_back(static_cast<index_t>(d));

    Tensor<InDataType> input(in_lengths);
    Tensor<WeiDataType> weight(wei_lengths);
    Tensor<OutDataType> output(out_lengths);

    // Allocate device memory
    DeviceMem input_dev(input.mData.size() * sizeof(InDataType));
    DeviceMem weight_dev(weight.mData.size() * sizeof(WeiDataType));
    DeviceMem output_dev(output.mData.size() * sizeof(OutDataType));

    // Initialize and copy tensors based on kernel type
    if(kernel_type == ConvKernelType::Forward)
    {
        initialize_and_copy_tensor(input, input_dev);
        initialize_and_copy_tensor(weight, weight_dev);
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        initialize_and_copy_tensor(weight, weight_dev);
        initialize_and_copy_tensor(output, output_dev);
    }
    else // BackwardWeight
    {
        initialize_and_copy_tensor(input, input_dev);
        initialize_and_copy_tensor(output, output_dev);
    }

    // Dispatch to appropriate implementation
    // All tensors already in CPU layout (GNCDHW/GKCZYX/GNKDHW)
    // Wrappers will handle all transformations automatically
    if(kernel_type == ConvKernelType::Forward)
    {
        return test_conv_fwd_impl<NDimSpatial, InDataType, WeiDataType, OutDataType>(
            params, input, weight, out_lengths, input_dev, weight_dev, output_dev);
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        return test_conv_bwd_data_impl<NDimSpatial, InDataType, WeiDataType, OutDataType>(
            params, weight, output, in_lengths, weight_dev, output_dev, input_dev);
    }
    else // BackwardWeight
    {
        return test_conv_bwd_weight_impl<NDimSpatial, InDataType, WeiDataType, OutDataType>(
            params, input, output, wei_lengths, input_dev, output_dev, weight_dev);
    }
}

} // namespace test
} // namespace ck
