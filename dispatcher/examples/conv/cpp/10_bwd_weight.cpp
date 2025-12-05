// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 10: Backward Weight Convolution
 *
 * Demonstrates backward weight gradient computation (dL/dWeight).
 * Used during neural network training to update filter weights.
 *
 * Build: cd dispatcher/build && cmake .. && make conv_10_bwd_weight
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL DECLARATIONS - Backward Weight (compv3 only)
// =============================================================================

DECL_CONV_KERNEL_SET(conv_bwd_weight_kernels,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("bwd_weight").dims(2),
                          ConvAlgo()
                              .tile(1, 16, 64)
                              .wave(1, 4, 1)
                              .warp(16, 16, 32)
                              .pipeline("compv3")
                              .scheduler("intrawave")
                              .vector_sizes(4, 8, 8)
                              .block_per_cu(1),
                          "gfx942"));

// =============================================================================
// DATA TYPES
// =============================================================================

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 10: Conv Backward Weight",
                     "Backward weight gradient computation (dL/dWeight)");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "64", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("--size", "28", "Spatial size (H=W)");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 10: Backward Weight Convolution\n";
    std::cout << "======================================================================\n\n";

    // Show declared kernels
    std::cout << "Step 1: Declared Kernels\n";
    std::cout << "------------------------\n";
    ConvKernelSetRegistry::instance().print();
    std::cout << "\n";

    // Define problem
    int N  = args.get_int("-n", 1);
    int C  = args.get_int("-c", 64);
    int K  = args.get_int("-k", 128);
    int Hi = args.get_int("--size", 28);
    int Wi = Hi;
    int Y = 3, X = 3;

    std::cout << "Step 2: Problem Configuration\n";
    std::cout << "-----------------------------\n";
    std::cout << "  Backward Weight: Compute dWeight from Input and dOutput\n";
    std::cout << "  Input shape: N=" << N << ", C=" << C << ", H=" << Hi << ", W=" << Wi << "\n";
    std::cout << "  Output gradient shape: N=" << N << ", K=" << K << ", Ho=" << Hi << ", Wo=" << Wi
              << "\n";
    std::cout << "  Weight gradient shape: K=" << K << ", C=" << C << ", Y=" << Y << ", X=" << X
              << "\n\n";

    // Create conv param
    ck_tile::conv::ConvParam conv_param{
        2,
        1,
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(C),
        {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},
        {static_cast<ck_tile::index_t>(Hi), static_cast<ck_tile::index_t>(Wi)},
        {1, 1},
        {1, 1},
        {1, 1},
        {1, 1}};

    using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;
    using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;

    // For backward weight: Input + dOutput -> dWeight
    auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    auto dout_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);
    auto dwei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    ck_tile::HostTensor<InDataType> input(in_desc);
    ck_tile::HostTensor<OutDataType> doutput(dout_desc);
    ck_tile::HostTensor<WeiDataType> dweight(dwei_desc);

    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<OutDataType>{-0.5f, 0.5f}(doutput);
    dweight.SetZero();

    std::cout << "Step 3: GPU Execution\n";
    std::cout << "---------------------\n";
    std::cout << "  Input:   " << input.get_element_space_size() << " elements\n";
    std::cout << "  dOutput: " << doutput.get_element_space_size() << " elements\n";
    std::cout << "  dWeight: " << dweight.get_element_space_size() << " elements\n";

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem doutput_dev(doutput.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dweight_dev(dweight.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    doutput_dev.ToDevice(doutput.data());
    dweight_dev.SetZero();

    // Backward weight kernel args
    ck_tile::GroupedConvBwdWeightHostArgs kernel_args(
        conv_param,
        input_dev.GetDeviceBuffer(),   // Input (forward activation)
        dweight_dev.GetDeviceBuffer(), // dWeight (output)
        {},
        doutput_dev.GetDeviceBuffer(), // dOutput (gradient from next layer)
        1);

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 3, 10};

    // Use backward weight launcher
    using Launcher   = generated::FirstKernelLauncher;
    float elapsed_ms = Launcher::launch(kernel_args, stream_cfg);

    dweight_dev.FromDevice(dweight.data());

    double flops  = 2.0 * N * K * C * Y * X * Hi * Wi;
    double tflops = flops / (elapsed_ms * 1e9);

    // Basic output check
    size_t non_zero = 0;
    for(size_t i = 0; i < dweight.get_element_space_size(); ++i)
        if(std::abs(static_cast<float>(dweight.data()[i])) > 1e-6f)
            ++non_zero;
    bool passed = (non_zero > 0);

    std::cout << "\n  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";
    std::cout << "  Non-zero outputs: " << non_zero << "/" << dweight.get_element_space_size()
              << "\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";

    std::cout << "\n======================================================================\n";
    std::cout << "Backward Weight: Computes dL/dWeight for training\n";
    std::cout << "======================================================================\n";

    return passed ? 0 : 1;
}
