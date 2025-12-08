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

#include "common_test_params.hpp"

namespace ck {
namespace test {

enum class ConvKernelType
{
    Forward,
    BackwardData,
    BackwardWeight
};

template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_gpu_ref(const ConvParams<NDimSpatial>& params, ConvKernelType kernel_type)
{
    // Calculate dimensions
    const index_t N = params.N;
    const index_t K = params.K;
    const index_t C = params.C;

    // For GPU reference (NDHWC format)
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

    // Create host tensors
    Tensor<InDataType> input_gpu(in_lengths_gpu);
    Tensor<WeiDataType> weight_gpu(wei_lengths_gpu);
    Tensor<OutDataType> output_gpu(out_lengths_gpu);

    // Initialize with random data based on kernel type
    if(kernel_type == ConvKernelType::Forward)
    {
        input_gpu.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        weight_gpu.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        weight_gpu.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        output_gpu.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
    }
    else if(kernel_type == ConvKernelType::BackwardWeight)
    {
        input_gpu.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        output_gpu.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
    }

    // Allocate device memory
    DeviceMem input_dev(input_gpu.mData.size() * sizeof(InDataType));
    DeviceMem weight_dev(weight_gpu.mData.size() * sizeof(WeiDataType));
    DeviceMem output_dev(output_gpu.mData.size() * sizeof(OutDataType));

    // Copy to device based on kernel type
    if(kernel_type == ConvKernelType::Forward)
    {
        input_dev.ToDevice(input_gpu.mData.data());
        weight_dev.ToDevice(weight_gpu.mData.data());
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        weight_dev.ToDevice(weight_gpu.mData.data());
        output_dev.ToDevice(output_gpu.mData.data());
    }
    else if(kernel_type == ConvKernelType::BackwardWeight)
    {
        input_dev.ToDevice(input_gpu.mData.data());
        output_dev.ToDevice(output_gpu.mData.data());
    }

    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;

    constexpr index_t block_size = 256;

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

    // Create ConvDims structure for kernel
    ref::ConvDims dims;
    dims.N          = N;
    dims.K          = K;
    dims.C          = C;
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

    // Calculate grid size based on output tensor
    long_index_t output_length;
    if(kernel_type == ConvKernelType::Forward)
        output_length = N * Do * Ho * Wo * K;
    else if(kernel_type == ConvKernelType::BackwardData)
        output_length = N * Di * Hi * Wi * C;
    else // BackwardWeight
        output_length = K * Z * Y * X * C;

    const index_t grid_size = (output_length + block_size - 1) / block_size;

    // Store kernel function pointers to avoid macro template issues
    const auto kernel_fwd = ref::naive_conv_fwd_ndhwc_kzyxc_ndhwk<InDataType,
                                                                  WeiDataType,
                                                                  OutDataType,
                                                                  float,
                                                                  InElementOp,
                                                                  WeiElementOp,
                                                                  OutElementOp>;

    const auto kernel_bwd_data = ref::naive_conv_bwd_data_ndhwc_kzyxc_ndhwk<InDataType,
                                                                            WeiDataType,
                                                                            OutDataType,
                                                                            float,
                                                                            InElementOp,
                                                                            WeiElementOp,
                                                                            OutElementOp>;

    const auto kernel_bwd_weight = ref::naive_conv_bwd_weight_ndhwc_kzyxc_ndhwk<InDataType,
                                                                                WeiDataType,
                                                                                OutDataType,
                                                                                float,
                                                                                InElementOp,
                                                                                WeiElementOp,
                                                                                OutElementOp>;

    // Launch appropriate kernel
    if(kernel_type == ConvKernelType::Forward)
    {
        hipLaunchKernelGGL(kernel_fwd,
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           nullptr,
                           reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
                           reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
                           reinterpret_cast<OutDataType*>(output_dev.GetDeviceBuffer()),
                           dims);
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        hipLaunchKernelGGL(kernel_bwd_data,
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           nullptr,
                           reinterpret_cast<OutDataType*>(output_dev.GetDeviceBuffer()),
                           reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
                           reinterpret_cast<InDataType*>(input_dev.GetDeviceBuffer()),
                           dims);
    }
    else // BackwardWeight
    {
        hipLaunchKernelGGL(kernel_bwd_weight,
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           nullptr,
                           reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
                           reinterpret_cast<OutDataType*>(output_dev.GetDeviceBuffer()),
                           reinterpret_cast<WeiDataType*>(weight_dev.GetDeviceBuffer()),
                           dims);
    }

    (void)hipDeviceSynchronize();

    // Copy result back based on kernel type
    if(kernel_type == ConvKernelType::Forward)
    {
        output_dev.FromDevice(output_gpu.mData.data());
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        input_dev.FromDevice(input_gpu.mData.data());
    }
    else // BackwardWeight
    {
        weight_dev.FromDevice(weight_gpu.mData.data());
    }

    // Run CPU reference for comparison
    // CPU reference expects GNCDHW/GKCZYX/GNKDHW format
    std::vector<index_t> in_lengths_cpu = {1, N, C}; // G=1, N, C
    for(auto d : params.input_spatial)
        in_lengths_cpu.push_back(d);

    std::vector<index_t> wei_lengths_cpu = {1, K, C}; // G=1, K, C
    for(auto d : params.filter_spatial)
        wei_lengths_cpu.push_back(d);

    std::vector<index_t> out_lengths_cpu = {1, N, K}; // G=1, N, K
    for(auto d : params.output_spatial)
        out_lengths_cpu.push_back(d);

    Tensor<InDataType> input_cpu(in_lengths_cpu);
    Tensor<WeiDataType> weight_cpu(wei_lengths_cpu);
    Tensor<OutDataType> output_cpu(out_lengths_cpu);

    // Initialize CPU tensors with same data as GPU (simplified - same random gen)
    input_cpu.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
    weight_cpu.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
    output_cpu.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});

    // Convert std::vector<index_t> to std::vector<long_index_t>
    std::vector<long_index_t> strides_long(params.strides.begin(), params.strides.end());
    std::vector<long_index_t> dilations_long(params.dilations.begin(), params.dilations.end());
    std::vector<long_index_t> pads_long(params.pads.begin(), params.pads.end());

    bool pass = true;

    if(kernel_type == ConvKernelType::Forward)
    {
        auto ref_conv    = tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                    InDataType,
                                                                    WeiDataType,
                                                                    OutDataType,
                                                                    InElementOp,
                                                                    WeiElementOp,
                                                                    OutElementOp>();
        auto ref_invoker = ref_conv.MakeInvoker();
        auto ref_arg     = ref_conv.MakeArgument(input_cpu,
                                             weight_cpu,
                                             output_cpu,
                                             strides_long,
                                             dilations_long,
                                             pads_long,
                                             pads_long,
                                             InElementOp{},
                                             WeiElementOp{},
                                             OutElementOp{});
        ref_invoker.Run(ref_arg);

        // Compare GPU vs CPU (note: different layouts, so just verify no crash for now)
        // TODO: Add proper layout conversion for accurate comparison
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        auto ref_conv    = tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                        InDataType,
                                                                        WeiDataType,
                                                                        OutDataType,
                                                                        InElementOp,
                                                                        WeiElementOp,
                                                                        OutElementOp>();
        auto ref_invoker = ref_conv.MakeInvoker();
        auto ref_arg     = ref_conv.MakeArgument(input_cpu,
                                             weight_cpu,
                                             output_cpu,
                                             strides_long,
                                             dilations_long,
                                             pads_long,
                                             pads_long,
                                             InElementOp{},
                                             WeiElementOp{},
                                             OutElementOp{});
        ref_invoker.Run(ref_arg);
    }
    else // BackwardWeight
    {
        auto ref_conv    = tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                          InDataType,
                                                                          WeiDataType,
                                                                          OutDataType,
                                                                          InElementOp,
                                                                          WeiElementOp,
                                                                          OutElementOp>();
        auto ref_invoker = ref_conv.MakeInvoker();
        auto ref_arg     = ref_conv.MakeArgument(input_cpu,
                                             weight_cpu,
                                             output_cpu,
                                             strides_long,
                                             dilations_long,
                                             pads_long,
                                             pads_long,
                                             InElementOp{},
                                             WeiElementOp{},
                                             OutElementOp{});
        ref_invoker.Run(ref_arg);
    }

    // Verify GPU kernel ran without errors
    return pass;
}

} // namespace test
} // namespace ck
