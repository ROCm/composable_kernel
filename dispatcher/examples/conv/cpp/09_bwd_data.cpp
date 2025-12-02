// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 10: Backward Data Convolution with GPU Execution and Validation
 *
 * Demonstrates backward data gradient computation (dL/dInput).
 * Used during neural network backpropagation.
 * Includes CPU reference validation to verify GPU results.
 *
 * Complexity: ★★★☆☆
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/conv_utils.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_bwd_data.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;

// =============================================================================
// KERNEL DECLARATIONS - Backward Data
// =============================================================================

DECL_CONV_KERNEL_SET(conv_bwd_data_kernels,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("bwd_data").dims(2),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv3")
                              .scheduler("intrawave"),
                          "gfx942"));

// =============================================================================
// DATA TYPES
// =============================================================================

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;
using AccDataType = float;

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    // Parse args for validation flag
    bool verify = false;
    for(int i = 1; i < argc; ++i)
    {
        if(std::string(argv[i]) == "--verify" || std::string(argv[i]) == "-v")
        {
            verify = true;
        }
    }

    std::cout << "======================================================================\n";
    std::cout << "Example 10: Backward Data Convolution" << (verify ? " (with validation)" : "")
              << "\n";
    std::cout << "======================================================================\n\n";

    // -------------------------------------------------------------------------
    // Step 1: Show declared kernels
    // -------------------------------------------------------------------------
    std::cout << "Step 1: Declared Backward Data Kernels\n";
    std::cout << "---------------------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_bwd_data_kernels");
    kernel_set.print(std::cout);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 2: Define problem
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Define Problem\n";
    std::cout << "----------------------\n";

    int N = 1, C = 64, K = 128, Hi = 28, Wi = 28, Y = 3, X = 3;
    auto problem = create_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1, ConvOp::BackwardData);
    print_problem(problem);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 3: Create registry
    // -------------------------------------------------------------------------
    std::cout << "Step 3: Create Registry\n";
    std::cout << "-----------------------\n";

    ConvRegistry registry;
    registry.register_set(kernel_set, ConvRegistry::Priority::High);
    std::cout << "  Registered " << registry.size() << " kernels\n\n";

    // -------------------------------------------------------------------------
    // Step 4: GPU Execution
    // -------------------------------------------------------------------------
    std::cout << "Step 4: GPU Execution\n";
    std::cout << "---------------------\n";

#ifdef CONV_KERNEL_AVAILABLE
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

    // For backward data: input is dOutput, weight is filter, output is dInput
    auto dout_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);
    auto wei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    auto din_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    ck_tile::HostTensor<OutDataType> doutput(dout_desc);  // Gradient from next layer
    ck_tile::HostTensor<WeiDataType> weight(wei_desc);    // Filter weights
    ck_tile::HostTensor<InDataType> dinput_gpu(din_desc); // GPU result
    ck_tile::HostTensor<InDataType> dinput_cpu(din_desc); // CPU reference

    ck_tile::FillUniformDistribution<OutDataType>{-0.5f, 0.5f}(doutput);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
    dinput_gpu.SetZero();
    dinput_cpu.SetZero();

    std::cout << "  dOutput: " << doutput.mDesc << "\n";
    std::cout << "  Weight:  " << weight.mDesc << "\n";
    std::cout << "  dInput:  " << dinput_gpu.mDesc << "\n";

    ck_tile::DeviceMem doutput_dev(doutput.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dinput_dev(dinput_gpu.get_element_space_size_in_bytes());

    doutput_dev.ToDevice(doutput.data());
    weight_dev.ToDevice(weight.data());
    dinput_dev.SetZero();

    // Backward data: compute dInput from dOutput and Weight
    // GroupedConvBwdDataHostArgs: (in_ptr=dInput, wei_ptr=Weight, out_ptr=dOutput)
    ck_tile::GroupedConvBwdDataHostArgs args(
        conv_param,
        dinput_dev.GetDeviceBuffer(),  // dInput (output of bwd_data)
        weight_dev.GetDeviceBuffer(),  // Weight
        {},                            // D tensors (empty)
        doutput_dev.GetDeviceBuffer(), // dOutput (input to bwd_data)
        1                              // k_batch
    );

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 5, 20};
    float elapsed_ms = SelectedConvBwdDataLauncher::launch(args, stream_cfg);

    // Copy results back
    dinput_dev.FromDevice(dinput_gpu.data());

    double flops  = problem.get_flops();
    double tflops = flops / (elapsed_ms * 1e9);

    std::cout << "\n  *** BACKWARD DATA GPU EXECUTION ***\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";
    std::cout << "  GPU[0,0,0,0,0]: " << std::fixed << std::setprecision(4)
              << static_cast<float>(dinput_gpu(0, 0, 0, 0, 0)) << "\n";

    // -------------------------------------------------------------------------
    // Step 5: CPU Reference and Validation
    // -------------------------------------------------------------------------
    if(verify)
    {
        std::cout << "\nStep 5: CPU Reference Validation\n";
        std::cout << "---------------------------------\n";

        std::vector<ck_tile::long_index_t> strides    = {1, 1};
        std::vector<ck_tile::long_index_t> dilations  = {1, 1};
        std::vector<ck_tile::long_index_t> left_pads  = {1, 1};
        std::vector<ck_tile::long_index_t> right_pads = {1, 1};

        // Compute CPU reference
        ck_tile::reference_grouped_conv_bwd_data<2, InDataType, WeiDataType, OutDataType>(
            dinput_cpu, weight, doutput, strides, dilations, left_pads, right_pads);

        std::cout << "  CPU[0,0,0,0,0]: " << std::fixed << std::setprecision(4)
                  << static_cast<float>(dinput_cpu(0, 0, 0, 0, 0)) << "\n";

        // Compare GPU and CPU results
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;

        for(size_t i = 0; i < dinput_gpu.get_element_space_size(); ++i)
        {
            float gpu_val   = static_cast<float>(dinput_gpu.data()[i]);
            float cpu_val   = static_cast<float>(dinput_cpu.data()[i]);
            double abs_diff = std::abs(gpu_val - cpu_val);
            double rel_diff = cpu_val != 0.0f ? abs_diff / std::abs(cpu_val) : abs_diff;
            max_abs_diff    = std::max(max_abs_diff, abs_diff);
            max_rel_diff    = std::max(max_rel_diff, rel_diff);
        }

        std::cout << "\n  Max abs diff: " << std::scientific << std::setprecision(4) << max_abs_diff
                  << "\n";
        std::cout << "  Max rel diff: " << std::scientific << std::setprecision(4) << max_rel_diff
                  << "\n";

        // FP16 tolerance - allow higher error due to limited precision
        bool passed = max_rel_diff < 0.05; // 5% relative error for FP16
        std::cout << "  Status: " << (passed ? "PASSED" : "FAILED") << "\n";
    }

#else
    std::cout << "  [Kernel not compiled]\n";
    std::cout << "  Note: Backward data requires proper CK Tile backward kernel codegen\n";
#endif

    std::cout << "\n======================================================================\n";
    std::cout << "Backward Data: Computes dL/dInput for backpropagation\n";
    std::cout << "======================================================================\n";

    return 0;
}
