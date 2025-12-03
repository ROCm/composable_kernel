// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 10: Backward Weight Convolution with GPU Execution and Validation
 *
 * Demonstrates backward weight gradient computation (dL/dWeight).
 * Used during neural network training to update filter weights.
 * Includes CPU reference validation to verify GPU results.
 *
 * Complexity: ★★★☆☆
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
#include "ck_tile/host/reference/reference_grouped_conv_bwd_weight.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL DECLARATIONS - Backward Weight
// =============================================================================

// Use ConvConfigComputeV3 validated configuration:
// M=16 (batch*spatial), N=64 (output channels), K=64 (input channels)
// Wave=(1,4,1), Warp=(16,16,32)
DECL_CONV_KERNEL_SET(conv_bwd_weight_kernels,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("bwd_weight").dims(2),
                          ConvAlgo()
                              .tile(16, 64, 64)
                              .wave(1, 4, 1)
                              .warp(16, 16, 32)
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
    ExampleArgs args("Example 10: Conv Backward Weight",
                     "Backward weight gradient computation (dL/dWeight)");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "64", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("--size", "28", "Spatial size (H=W)");
    args.add_flag("--verify", "Enable CPU validation");
    args.add_flag("-v", "Enable CPU validation");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    bool verify = args.has("--verify") || args.has("-v");

    std::cout << "======================================================================\n";
    std::cout << "Example 10: Backward Weight Convolution" << (verify ? " (with validation)" : "")
              << "\n";
    std::cout << "======================================================================\n\n";

    if(args.has("--list"))
    {
        std::cout << "Declared Kernel Sets:\n";
        ConvKernelSetRegistry::instance().print();
        return 0;
    }

    // -------------------------------------------------------------------------
    // Step 1: Show declared kernels
    // -------------------------------------------------------------------------
    std::cout << "Step 1: Declared Backward Weight Kernels\n";
    std::cout << "-----------------------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_bwd_weight_kernels");
    kernel_set.print(std::cout);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 2: Define problem
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Define Problem\n";
    std::cout << "----------------------\n";

    int N  = args.get_int("-n", 1);
    int C  = args.get_int("-c", 64);
    int K  = args.get_int("-k", 128);
    int Hi = args.get_int("--size", 28);
    int Wi = Hi;
    int Y = 3, X = 3;
    auto problem = create_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1, ConvOp::BackwardWeight);
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

    // For backward weight: Input is forward activation, dOutput is gradient
    auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    auto dout_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);
    auto dwei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    ck_tile::HostTensor<InDataType> input(in_desc);          // Forward activation
    ck_tile::HostTensor<OutDataType> doutput(dout_desc);     // Gradient from next layer
    ck_tile::HostTensor<WeiDataType> dweight_gpu(dwei_desc); // GPU result
    ck_tile::HostTensor<WeiDataType> dweight_cpu(dwei_desc); // CPU reference

    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<OutDataType>{-0.5f, 0.5f}(doutput);
    dweight_gpu.SetZero();
    dweight_cpu.SetZero();

    std::cout << "  Input:   " << input.mDesc << "\n";
    std::cout << "  dOutput: " << doutput.mDesc << "\n";
    std::cout << "  dWeight: " << dweight_gpu.mDesc << "\n";

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem doutput_dev(doutput.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dweight_dev(dweight_gpu.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    doutput_dev.ToDevice(doutput.data());
    dweight_dev.SetZero();

    // Backward weight: compute dWeight from Input and dOutput
    // GroupedConvBwdWeightHostArgs: (in_ptr=Input, wei_ptr=dWeight, out_ptr=dOutput)
    ck_tile::GroupedConvBwdWeightHostArgs kernel_args(
        conv_param,
        input_dev.GetDeviceBuffer(),   // Input (forward activation)
        dweight_dev.GetDeviceBuffer(), // dWeight (output of bwd_weight)
        {},                            // D tensors (empty)
        doutput_dev.GetDeviceBuffer(), // dOutput (gradient from next layer)
        1                              // k_batch
    );

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 5, 20};
    float elapsed_ms = SelectedConvBwdWeightLauncher::launch(kernel_args, stream_cfg);

    // Copy results back
    dweight_dev.FromDevice(dweight_gpu.data());

    double flops  = problem.get_flops();
    double tflops = flops / (elapsed_ms * 1e9);

    std::cout << "\n  *** BACKWARD WEIGHT GPU EXECUTION ***\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";
    std::cout << "  GPU[0,0,0,0,0]: " << std::fixed << std::setprecision(4)
              << static_cast<float>(dweight_gpu(0, 0, 0, 0, 0)) << "\n";

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
        ck_tile::reference_grouped_conv_bwd_weight<2, InDataType, WeiDataType, OutDataType>(
            input, dweight_cpu, doutput, strides, dilations, left_pads, right_pads);

        std::cout << "  CPU[0,0,0,0,0]: " << std::fixed << std::setprecision(4)
                  << static_cast<float>(dweight_cpu(0, 0, 0, 0, 0)) << "\n";

        // Compare GPU and CPU results
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;

        for(size_t i = 0; i < dweight_gpu.get_element_space_size(); ++i)
        {
            float gpu_val   = static_cast<float>(dweight_gpu.data()[i]);
            float cpu_val   = static_cast<float>(dweight_cpu.data()[i]);
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
    std::cout << "  Generate with: python3 codegen/unified_conv_codegen.py --variant bwd_weight\n";
#endif

    std::cout << "\n======================================================================\n";
    std::cout << "Backward Weight: Computes dL/dWeight for training\n";
    std::cout << "======================================================================\n";

    return 0;
}
