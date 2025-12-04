// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// Standalone test program for Old CK GPU references
// Tests naive_conv_fwd (existing) and future backward ops

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

// CPU reference for validation
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

// GPU reference (OLD CK - already exists!)
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"

using namespace ck;

template <index_t NDimSpatial>
struct ConvParams
{
    index_t N, K, C;
    std::vector<index_t> input_spatial;
    std::vector<index_t> filter_spatial;
    std::vector<index_t> output_spatial;
    std::vector<index_t> strides;
    std::vector<index_t> dilations;
    std::vector<index_t> pads;
};

template <index_t NDimSpatial, typename InDataType, typename WeiDataType, typename OutDataType>
bool test_conv_forward_gpu_ref(const ConvParams<NDimSpatial>& params, const std::string& test_name)
{
    std::cout << "[TEST] " << test_name << std::endl;

    // Calculate dimensions
    const index_t N = params.N;
    const index_t K = params.K;
    const index_t C = params.C;

    // Create tensor descriptors (NDHWC layout for old CK)
    std::vector<index_t> in_lengths = {N};
    for(auto d : params.input_spatial)
        in_lengths.push_back(d);
    in_lengths.push_back(C);

    std::vector<index_t> wei_lengths = {K};
    for(auto d : params.filter_spatial)
        wei_lengths.push_back(d);
    wei_lengths.push_back(C);

    std::vector<index_t> out_lengths = {N};
    for(auto d : params.output_spatial)
        out_lengths.push_back(d);
    out_lengths.push_back(K);

    // Create host tensors
    Tensor<InDataType> input(in_lengths);
    Tensor<WeiDataType> weight(wei_lengths);
    Tensor<OutDataType> output_gpu(out_lengths);
    Tensor<OutDataType> output_ref(out_lengths);

    // Initialize with random data
    input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
    weight.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});

    // Allocate device memory
    DeviceMem input_dev(input.mData.size() * sizeof(InDataType));
    DeviceMem weight_dev(weight.mData.size() * sizeof(WeiDataType));
    DeviceMem output_dev(output_gpu.mData.size() * sizeof(OutDataType));

    // Copy to device
    input_dev.ToDevice(input.mData.data());
    weight_dev.ToDevice(weight.mData.data());

    // Run CPU reference for validation
    auto ref_conv =
        tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                 InDataType,
                                                 WeiDataType,
                                                 OutDataType,
                                                 tensor_operation::element_wise::PassThrough,
                                                 tensor_operation::element_wise::PassThrough,
                                                 tensor_operation::element_wise::PassThrough>();

    auto ref_invoker = ref_conv.MakeInvoker();
    auto ref_arg     = ref_conv.MakeArgument(input.mData.data(),
                                         weight.mData.data(),
                                         output_ref.mData.data(),
                                         N,
                                         K,
                                         C,
                                         params.input_spatial,
                                         params.filter_spatial,
                                         params.output_spatial,
                                         params.strides,
                                         params.dilations,
                                         params.pads,
                                         params.pads,
                                             {},
                                             {},
                                             {});

    ref_invoker.Run(ref_arg);

    // Run GPU reference (OLD CK)
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

    // Launch GPU reference kernel
    const long_index_t output_length = N * Do * Ho * Wo * K;
    const index_t grid_size          = (output_length + block_size - 1) / block_size;

    hipLaunchKernelGGL(ref::naive_conv_fwd_ndhwc_kzyxc_ndhwk<InDataType,
                                                             WeiDataType,
                                                             OutDataType,
                                                             float,
                                                             InElementOp,
                                                             WeiElementOp,
                                                             OutElementOp>,
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       nullptr,
                       reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
                       reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
                       reinterpret_cast<OutDataType*>(output_dev.GetDeviceBuffer()),
                       N,
                       K,
                       C,
                       Di,
                       Hi,
                       Wi,
                       Z,
                       Y,
                       X,
                       Do,
                       Ho,
                       Wo,
                       stride_z,
                       stride_y,
                       stride_x,
                       dilation_z,
                       dilation_y,
                       dilation_x,
                       pad_z,
                       pad_y,
                       pad_x);

    hipDeviceSynchronize();

    // Copy result back
    output_dev.FromDevice(output_gpu.mData.data());

    // Compare GPU ref vs CPU ref
    bool pass = check_err(output_gpu.mData, output_ref.mData, "GPU vs CPU ref", 1e-3, 1e-3);

    std::cout << "  Result: " << (pass ? "✅ PASS" : "❌ FAIL") << std::endl;

    return pass;
}

int main(int argc, char* argv[])
{
    std::cout << "========================================" << std::endl;
    std::cout << "Old CK GPU Reference Test Program" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int failed = 0;

    // Test 1: 2D Conv, FP16, Small
    {
        ConvParams<2> params;
        params.N              = 2;
        params.K              = 8;
        params.C              = 8;
        params.input_spatial  = {7, 7};
        params.filter_spatial = {3, 3};
        params.output_spatial = {5, 5};
        params.strides        = {1, 1};
        params.dilations      = {1, 1};
        params.pads           = {0, 0};

        if(test_conv_forward_gpu_ref<2, half_t, half_t, half_t>(params, "2D-FP16-Small"))
            passed++;
        else
            failed++;
    }

    // Test 2: 2D Conv, FP32, Medium
    {
        ConvParams<2> params;
        params.N              = 4;
        params.K              = 16;
        params.C              = 16;
        params.input_spatial  = {14, 14};
        params.filter_spatial = {3, 3};
        params.output_spatial = {12, 12};
        params.strides        = {1, 1};
        params.dilations      = {1, 1};
        params.pads           = {0, 0};

        if(test_conv_forward_gpu_ref<2, float, float, float>(params, "2D-FP32-Medium"))
            passed++;
        else
            failed++;
    }

    // Test 3: 1D Conv, FP16
    {
        ConvParams<1> params;
        params.N              = 2;
        params.K              = 8;
        params.C              = 8;
        params.input_spatial  = {16};
        params.filter_spatial = {3};
        params.output_spatial = {14};
        params.strides        = {1};
        params.dilations      = {1};
        params.pads           = {0};

        if(test_conv_forward_gpu_ref<1, half_t, half_t, half_t>(params, "1D-FP16"))
            passed++;
        else
            failed++;
    }

    // Test 4: 3D Conv, FP16, Small
    {
        ConvParams<3> params;
        params.N              = 1;
        params.K              = 8;
        params.C              = 8;
        params.input_spatial  = {5, 5, 5};
        params.filter_spatial = {3, 3, 3};
        params.output_spatial = {3, 3, 3};
        params.strides        = {1, 1, 1};
        params.dilations      = {1, 1, 1};
        params.pads           = {0, 0, 0};

        if(test_conv_forward_gpu_ref<3, half_t, half_t, half_t>(params, "3D-FP16-Small"))
            passed++;
        else
            failed++;
    }

    // Test 5: 2D Conv with stride
    {
        ConvParams<2> params;
        params.N              = 2;
        params.K              = 8;
        params.C              = 8;
        params.input_spatial  = {8, 8};
        params.filter_spatial = {3, 3};
        params.output_spatial = {3, 3};
        params.strides        = {2, 2};
        params.dilations      = {1, 1};
        params.pads           = {0, 0};

        if(test_conv_forward_gpu_ref<2, half_t, half_t, half_t>(params, "2D-FP16-Stride2"))
            passed++;
        else
            failed++;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total:  " << (passed + failed) << std::endl;
    std::cout << "Passed: " << passed << " ✅" << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << std::endl;

    if(failed == 0)
    {
        std::cout << "🎉 ALL TESTS PASSED!" << std::endl;
        std::cout << "Old CK Forward GPU Reference: WORKING ✅" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "❌ SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
