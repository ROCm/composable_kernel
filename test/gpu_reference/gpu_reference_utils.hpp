// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
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
#include "ck/library/reference_tensor_operation/gpu/conv_common.hpp"

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
void initialize_and_copy_tensor(Tensor<DataType>& host_tensor,
                                DeviceMem& device_mem,
                                bool should_initialize)
{
    if(should_initialize)
    {
        host_tensor.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        device_mem.ToDevice(host_tensor.mData.data());
    }
}

// Forward convolution implementation
template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_fwd_impl(const ConvParams<NDimSpatial>& params,
                        const Tensor<InDataType>& input_cpu,
                        const Tensor<WeiDataType>& weight_cpu,
                        const std::vector<index_t>& out_lengths_cpu,
                        DeviceMem& input_dev,
                        DeviceMem& weight_dev,
                        DeviceMem& output_dev,
                        DeviceMem& output_cpu_dev,
                        const ref::ConvDims& dims,
                        index_t G,
                        index_t N,
                        index_t K,
                        index_t Do,
                        index_t Ho,
                        index_t Wo)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;

    constexpr index_t block_size           = 256;
    constexpr index_t transform_block_size = 256;

    // Launch GPU kernel
    const long_index_t output_length = N * Do * Ho * Wo * K;
    const index_t grid_size          = (output_length + block_size - 1) / block_size;

    const auto kernel_fwd = ref::naive_conv_fwd_ndhwc_kzyxc_ndhwk<InDataType,
                                                                  WeiDataType,
                                                                  OutDataType,
                                                                  float,
                                                                  InElementOp,
                                                                  WeiElementOp,
                                                                  OutElementOp>;

    hipLaunchKernelGGL(kernel_fwd,
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       nullptr,
                       reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
                       reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
                       reinterpret_cast<OutDataType*>(output_dev.GetDeviceBuffer()),
                       dims);

    (void)hipDeviceSynchronize();

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.strides.begin(), params.strides.end());
    std::vector<long_index_t> dilations_long(params.dilations.begin(), params.dilations.end());
    std::vector<long_index_t> pads_long(params.pads.begin(), params.pads.end());

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

    // Transform GPU output back to CPU layout
    const index_t output_total = N * Do * Ho * Wo * K;
    const index_t output_grid  = (output_total + transform_block_size - 1) / transform_block_size;
    const index_t K_per_group  = K / G;

    Tensor<OutDataType> output_cpu(out_lengths_cpu);
    hipLaunchKernelGGL(ref::layout_transform::transform_output_NDHWGK_to_GNKDHW<OutDataType>,
                       dim3(output_grid),
                       dim3(transform_block_size),
                       0,
                       nullptr,
                       reinterpret_cast<const OutDataType*>(output_dev.GetDeviceBuffer()),
                       reinterpret_cast<OutDataType*>(output_cpu_dev.GetDeviceBuffer()),
                       G,
                       N,
                       K_per_group,
                       Do,
                       Ho,
                       Wo);
    output_cpu_dev.FromDevice(output_cpu.mData.data());
    (void)hipDeviceSynchronize();

    // Compare results
    return ck::utils::check_err(output_cpu, output_ref);
}

// Backward data convolution implementation
template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_bwd_data_impl(const ConvParams<NDimSpatial>& params,
                             const Tensor<WeiDataType>& weight_cpu,
                             const Tensor<OutDataType>& output_cpu,
                             const std::vector<index_t>& in_lengths_cpu,
                             DeviceMem& input_dev,
                             DeviceMem& weight_dev,
                             DeviceMem& output_dev,
                             DeviceMem& input_cpu_dev,
                             const ref::ConvDims& dims,
                             index_t G,
                             index_t N,
                             index_t C,
                             index_t Di,
                             index_t Hi,
                             index_t Wi)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;

    constexpr index_t block_size           = 256;
    constexpr index_t transform_block_size = 256;

    // Launch GPU kernel
    const long_index_t output_length = N * Di * Hi * Wi * C;
    const index_t grid_size          = (output_length + block_size - 1) / block_size;

    const auto kernel_bwd_data = ref::naive_conv_bwd_data_ndhwc_kzyxc_ndhwk<InDataType,
                                                                            WeiDataType,
                                                                            OutDataType,
                                                                            float,
                                                                            InElementOp,
                                                                            WeiElementOp,
                                                                            OutElementOp>;

    hipLaunchKernelGGL(kernel_bwd_data,
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       nullptr,
                       reinterpret_cast<InDataType*>(input_dev.GetDeviceBuffer()),
                       reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
                       reinterpret_cast<const OutDataType*>(output_dev.GetDeviceBuffer()),
                       dims);

    (void)hipDeviceSynchronize();

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.strides.begin(), params.strides.end());
    std::vector<long_index_t> dilations_long(params.dilations.begin(), params.dilations.end());
    std::vector<long_index_t> pads_long(params.pads.begin(), params.pads.end());

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

    // Transform GPU input result back to CPU layout
    const index_t input_total = N * Di * Hi * Wi * C;
    const index_t input_grid  = (input_total + transform_block_size - 1) / transform_block_size;

    const index_t C_per_group = C / G;

    Tensor<InDataType> input_cpu(in_lengths_cpu);
    hipLaunchKernelGGL(ref::layout_transform::transform_input_NDHWGC_to_GNCDHW<InDataType>,
                       dim3(input_grid),
                       dim3(transform_block_size),
                       0,
                       nullptr,
                       reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
                       reinterpret_cast<InDataType*>(input_cpu_dev.GetDeviceBuffer()),
                       G,
                       N,
                       C_per_group,
                       Di,
                       Hi,
                       Wi);
    input_cpu_dev.FromDevice(input_cpu.mData.data());
    (void)hipDeviceSynchronize();

    // Compare results
    return ck::utils::check_err(input_cpu, input_ref);
}

// Backward weight convolution implementation
template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_bwd_weight_impl(const ConvParams<NDimSpatial>& params,
                               const Tensor<InDataType>& input_cpu,
                               const Tensor<OutDataType>& output_cpu,
                               const std::vector<index_t>& wei_lengths_cpu,
                               DeviceMem& input_dev,
                               DeviceMem& weight_dev,
                               DeviceMem& output_dev,
                               DeviceMem& weight_cpu_dev,
                               const ref::ConvDims& dims,
                               index_t G,
                               index_t K,
                               index_t C,
                               index_t Z,
                               index_t Y,
                               index_t X)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;

    constexpr index_t block_size           = 256;
    constexpr index_t transform_block_size = 256;

    // Launch GPU kernel
    const long_index_t output_length = K * Z * Y * X * C;
    const index_t grid_size          = (output_length + block_size - 1) / block_size;

    const auto kernel_bwd_weight = ref::naive_conv_bwd_weight_ndhwc_kzyxc_ndhwk<InDataType,
                                                                                WeiDataType,
                                                                                OutDataType,
                                                                                float,
                                                                                InElementOp,
                                                                                WeiElementOp,
                                                                                OutElementOp>;

    hipLaunchKernelGGL(kernel_bwd_weight,
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       nullptr,
                       reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
                       reinterpret_cast<WeiDataType*>(weight_dev.GetDeviceBuffer()),
                       reinterpret_cast<const OutDataType*>(output_dev.GetDeviceBuffer()),
                       dims);

    (void)hipDeviceSynchronize();

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.strides.begin(), params.strides.end());
    std::vector<long_index_t> dilations_long(params.dilations.begin(), params.dilations.end());
    std::vector<long_index_t> pads_long(params.pads.begin(), params.pads.end());

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

    // Transform GPU weight result back to CPU layout
    const index_t weight_total = K * Z * Y * X * C;
    const index_t weight_grid  = (weight_total + transform_block_size - 1) / transform_block_size;
    const index_t K_per_group  = K / G;
    const index_t C_per_group  = C / G;

    Tensor<WeiDataType> weight_cpu_result(wei_lengths_cpu);
    hipLaunchKernelGGL(ref::layout_transform::transform_weight_KZYXGC_to_GKCZYX<WeiDataType>,
                       dim3(weight_grid),
                       dim3(transform_block_size),
                       0,
                       nullptr,
                       reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
                       reinterpret_cast<WeiDataType*>(weight_cpu_dev.GetDeviceBuffer()),
                       G,
                       K_per_group,
                       C_per_group,
                       Z,
                       Y,
                       X);
    weight_cpu_dev.FromDevice(weight_cpu_result.mData.data());
    (void)hipDeviceSynchronize();

    // Compare results
    return ck::utils::check_err(weight_cpu_result, weight_ref);
}

// Main test function - dispatches to specific implementations
template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_gpu_ref(const ConvParams<NDimSpatial>& params, ConvKernelType kernel_type)
{
    // Calculate dimensions
    const index_t N = params.N;
    const index_t K = params.K;
    const index_t C = params.C;
    const index_t G = params.G;

    // Create GPU tensors (NDHWC format - GPU kernels use total channels)
    std::vector<index_t> in_lengths_gpu = {N};
    for(auto d : params.input_spatial)
        in_lengths_gpu.push_back(d);
    in_lengths_gpu.push_back(C);

    std::vector<index_t> wei_lengths_gpu = {K};
    for(auto d : params.filter_spatial)
        wei_lengths_gpu.push_back(d);
    wei_lengths_gpu.push_back(C);

    std::vector<index_t> out_lengths_gpu = {N};
    for(auto d : params.output_spatial)
        out_lengths_gpu.push_back(d);
    out_lengths_gpu.push_back(K);

    Tensor<InDataType> input_gpu(in_lengths_gpu);
    Tensor<WeiDataType> weight_gpu(wei_lengths_gpu);
    Tensor<OutDataType> output_gpu(out_lengths_gpu);

    // Allocate device memory
    DeviceMem input_dev(input_gpu.mData.size() * sizeof(InDataType));
    DeviceMem weight_dev(weight_gpu.mData.size() * sizeof(WeiDataType));
    DeviceMem output_dev(output_gpu.mData.size() * sizeof(OutDataType));

    // Initialize and copy tensors based on kernel type
    if(kernel_type == ConvKernelType::Forward)
    {
        initialize_and_copy_tensor(input_gpu, input_dev, true);
        initialize_and_copy_tensor(weight_gpu, weight_dev, true);
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        initialize_and_copy_tensor(weight_gpu, weight_dev, true);
        initialize_and_copy_tensor(output_gpu, output_dev, true);
    }
    else // BackwardWeight
    {
        initialize_and_copy_tensor(input_gpu, input_dev, true);
        initialize_and_copy_tensor(output_gpu, output_dev, true);
    }

    // Extract dimensions based on NDimSpatial
    index_t Di = 1, Hi = 1, Wi = 1;
    index_t Z = 1, Y = 1, X = 1;
    index_t Do = 1, Ho = 1, Wo = 1;
    index_t stride_z = 1, stride_y = 1, stride_x = 1;
    index_t dilation_z = 1, dilation_y = 1, dilation_x = 1;
    index_t pad_z = 0, pad_y = 0, pad_x = 0;

    if(NDimSpatial == 1)
    {
        Wi         = params.input_spatial[0];
        X          = params.filter_spatial[0];
        Wo         = params.output_spatial[0];
        stride_x   = params.strides[0];
        dilation_x = params.dilations[0];
        pad_x      = params.pads[0];
    }
    else if(NDimSpatial == 2)
    {
        Hi         = params.input_spatial[0];
        Wi         = params.input_spatial[1];
        Y          = params.filter_spatial[0];
        X          = params.filter_spatial[1];
        Ho         = params.output_spatial[0];
        Wo         = params.output_spatial[1];
        stride_y   = params.strides[0];
        stride_x   = params.strides[1];
        dilation_y = params.dilations[0];
        dilation_x = params.dilations[1];
        pad_y      = params.pads[0];
        pad_x      = params.pads[1];
    }
    else if(NDimSpatial == 3)
    {
        Di         = params.input_spatial[0];
        Hi         = params.input_spatial[1];
        Wi         = params.input_spatial[2];
        Z          = params.filter_spatial[0];
        Y          = params.filter_spatial[1];
        X          = params.filter_spatial[2];
        Do         = params.output_spatial[0];
        Ho         = params.output_spatial[1];
        Wo         = params.output_spatial[2];
        stride_z   = params.strides[0];
        stride_y   = params.strides[1];
        stride_x   = params.strides[2];
        dilation_z = params.dilations[0];
        dilation_y = params.dilations[1];
        dilation_x = params.dilations[2];
        pad_z      = params.pads[0];
        pad_y      = params.pads[1];
        pad_x      = params.pads[2];
    }

    // Create ConvDims structure for kernels
    ref::ConvDims dims;
    dims.N          = N;
    dims.K          = K;
    dims.C          = C;
    dims.G          = G;
    dims.Di         = Di;
    dims.Hi         = Hi;
    dims.Wi         = Wi;
    dims.Z          = Z;
    dims.Y          = Y;
    dims.X          = X;
    dims.Do         = Do;
    dims.Ho         = Ho;
    dims.Wo         = Wo;
    dims.stride_z   = stride_z;
    dims.stride_y   = stride_y;
    dims.stride_x   = stride_x;
    dims.dilation_z = dilation_z;
    dims.dilation_y = dilation_y;
    dims.dilation_x = dilation_x;
    dims.pad_z      = pad_z;
    dims.pad_y      = pad_y;
    dims.pad_x      = pad_x;

    // Create CPU layout tensors for comparison (GNCDHW/GKCZYX/GNKDHW)
    // C and K in params are total channels, divide by G for per-group
    const index_t C_per_group = C / G;
    const index_t K_per_group = K / G;

    std::vector<index_t> in_lengths_cpu = {G, N, C_per_group};
    for(auto d : params.input_spatial)
        in_lengths_cpu.push_back(d);

    std::vector<index_t> wei_lengths_cpu = {G, K_per_group, C_per_group};
    for(auto d : params.filter_spatial)
        wei_lengths_cpu.push_back(d);

    std::vector<index_t> out_lengths_cpu = {G, N, K_per_group};
    for(auto d : params.output_spatial)
        out_lengths_cpu.push_back(d);

    Tensor<InDataType> input_cpu(in_lengths_cpu);
    Tensor<WeiDataType> weight_cpu(wei_lengths_cpu);
    Tensor<OutDataType> output_cpu(out_lengths_cpu);

    // Allocate device memory for layout transformations
    DeviceMem input_cpu_dev(input_cpu.mData.size() * sizeof(InDataType));
    DeviceMem weight_cpu_dev(weight_cpu.mData.size() * sizeof(WeiDataType));
    DeviceMem output_cpu_dev(output_cpu.mData.size() * sizeof(OutDataType));

    constexpr index_t transform_block_size = 256;

    // Transform INPUT tensors to CPU layout (before running GPU kernels)
    if(kernel_type == ConvKernelType::Forward || kernel_type == ConvKernelType::BackwardWeight)
    {
        const index_t input_total = N * Di * Hi * Wi * C;
        const index_t input_grid  = (input_total + transform_block_size - 1) / transform_block_size;

        hipLaunchKernelGGL(ref::layout_transform::transform_input_NDHWGC_to_GNCDHW<InDataType>,
                           dim3(input_grid),
                           dim3(transform_block_size),
                           0,
                           nullptr,
                           reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
                           reinterpret_cast<InDataType*>(input_cpu_dev.GetDeviceBuffer()),
                           G,
                           N,
                           C_per_group,
                           Di,
                           Hi,
                           Wi);
        input_cpu_dev.FromDevice(input_cpu.mData.data());
    }

    if(kernel_type == ConvKernelType::Forward || kernel_type == ConvKernelType::BackwardData)
    {
        const index_t weight_total = K * Z * Y * X * C;
        const index_t weight_grid =
            (weight_total + transform_block_size - 1) / transform_block_size;

        hipLaunchKernelGGL(ref::layout_transform::transform_weight_KZYXGC_to_GKCZYX<WeiDataType>,
                           dim3(weight_grid),
                           dim3(transform_block_size),
                           0,
                           nullptr,
                           reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
                           reinterpret_cast<WeiDataType*>(weight_cpu_dev.GetDeviceBuffer()),
                           G,
                           K_per_group,
                           C_per_group,
                           Z,
                           Y,
                           X);
        weight_cpu_dev.FromDevice(weight_cpu.mData.data());
    }

    if(kernel_type == ConvKernelType::BackwardData || kernel_type == ConvKernelType::BackwardWeight)
    {
        const index_t output_total = N * Do * Ho * Wo * K;
        const index_t output_grid =
            (output_total + transform_block_size - 1) / transform_block_size;

        hipLaunchKernelGGL(ref::layout_transform::transform_output_NDHWGK_to_GNKDHW<OutDataType>,
                           dim3(output_grid),
                           dim3(transform_block_size),
                           0,
                           nullptr,
                           reinterpret_cast<const OutDataType*>(output_dev.GetDeviceBuffer()),
                           reinterpret_cast<OutDataType*>(output_cpu_dev.GetDeviceBuffer()),
                           G,
                           N,
                           K_per_group,
                           Do,
                           Ho,
                           Wo);
        output_cpu_dev.FromDevice(output_cpu.mData.data());
    }

    (void)hipDeviceSynchronize();

    // Dispatch to appropriate implementation
    if(kernel_type == ConvKernelType::Forward)
    {
        return test_conv_fwd_impl<NDimSpatial, InDataType, WeiDataType, OutDataType>(
            params,
            input_cpu,
            weight_cpu,
            out_lengths_cpu,
            input_dev,
            weight_dev,
            output_dev,
            output_cpu_dev,
            dims,
            G,
            N,
            K,
            Do,
            Ho,
            Wo);
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        return test_conv_bwd_data_impl<NDimSpatial, InDataType, WeiDataType, OutDataType>(
            params,
            weight_cpu,
            output_cpu,
            in_lengths_cpu,
            input_dev,
            weight_dev,
            output_dev,
            input_cpu_dev,
            dims,
            G,
            N,
            C,
            Di,
            Hi,
            Wi);
    }
    else // BackwardWeight
    {
        return test_conv_bwd_weight_impl<NDimSpatial, InDataType, WeiDataType, OutDataType>(
            params,
            input_cpu,
            output_cpu,
            wei_lengths_cpu,
            input_dev,
            weight_dev,
            output_dev,
            weight_cpu_dev,
            dims,
            G,
            K,
            C,
            Z,
            Y,
            X);
    }
}

} // namespace test
} // namespace ck
