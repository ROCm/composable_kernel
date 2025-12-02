// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 02: Convolution with CPU Validation - Declarative
 *
 * Demonstrates convolution with CPU reference verification.
 * Uses the Signature/Algorithm/Arch declarative pattern.
 *
 * Self-contained build:
 *   python3 scripts/compile_conv_examples.py examples/conv/cpp/02_conv_validation.cpp
 *
 * Complexity: ★★★☆☆
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>

// Declarative utilities
#include "ck_tile/dispatcher/conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

// CK Tile includes
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_fwd.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL DECLARATIONS
// =============================================================================

DECL_CONV_KERNEL_SET(conv_validation_kernels,
                     // Validation kernel
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv4")
                              .scheduler("intrawave"),
                          "gfx942"));

// =============================================================================
// TYPES
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
    ExampleArgs args("Example 02: Conv Validation", "Convolution with CPU reference verification");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "64", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("--size", "14", "Spatial size (H=W)");
    args.add_flag("--no-verify", "Skip CPU validation");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 02: Convolution with CPU Validation (Declarative)\n";
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
    std::cout << "Step 1: Declared Kernels\n";
    std::cout << "------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_validation_kernels");
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
    int Hi = args.get_int("--size", 14);
    int Wi = Hi;
    int Y = 3, X = 3;
    bool verify = !args.has("--no-verify");

    auto problem = create_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1, ConvOp::Forward);
    print_problem(problem);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 3: Create CK Tile parameters
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Step 4: Allocate tensors
    // -------------------------------------------------------------------------
    std::cout << "Step 3: Allocate Tensors\n";
    std::cout << "------------------------\n";

    auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    auto wei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    auto out_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    ck_tile::HostTensor<InDataType> input(in_desc);
    ck_tile::HostTensor<WeiDataType> weight(wei_desc);
    ck_tile::HostTensor<OutDataType> output_gpu(out_desc);
    ck_tile::HostTensor<OutDataType> output_cpu(out_desc);

    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
    output_gpu.SetZero();
    output_cpu.SetZero();

    std::cout << "  Input:  " << input.mDesc << "\n";
    std::cout << "  Weight: " << weight.mDesc << "\n";
    std::cout << "  Output: " << output_gpu.mDesc << "\n\n";

    // -------------------------------------------------------------------------
    // Step 5: CPU Reference
    // -------------------------------------------------------------------------
    if(verify)
    {
        std::cout << "Step 4: CPU Reference Computation\n";
        std::cout << "----------------------------------\n";

        // reference_grouped_conv_fwd requires stride, dilation, padding vectors
        std::vector<ck_tile::long_index_t> strides    = {1, 1};
        std::vector<ck_tile::long_index_t> dilations  = {1, 1};
        std::vector<ck_tile::long_index_t> left_pads  = {1, 1};
        std::vector<ck_tile::long_index_t> right_pads = {1, 1};

        ck_tile::reference_grouped_conv_fwd<2, InDataType, WeiDataType, OutDataType>(
            input, weight, output_cpu, strides, dilations, left_pads, right_pads);

        std::cout << "  CPU reference computed\n";
        std::cout << "  Output[0,0,0,0,0]: " << static_cast<float>(output_cpu(0, 0, 0, 0, 0))
                  << "\n\n";
    }

    // -------------------------------------------------------------------------
    // Step 6: GPU Execution
    // -------------------------------------------------------------------------
    std::cout << "Step 5: GPU Execution\n";
    std::cout << "---------------------\n";

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output_gpu.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());
    output_dev.SetZero();

#ifdef CONV_KERNEL_AVAILABLE
    ck_tile::GroupedConvFwdHostArgs<> args(conv_param,
                                           input_dev.GetDeviceBuffer(),
                                           weight_dev.GetDeviceBuffer(),
                                           {},
                                           output_dev.GetDeviceBuffer(),
                                           1);

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 3, 10};
    float elapsed_ms = SelectedConvKernelLauncher::launch(args, stream_cfg);

    output_dev.FromDevice(output_gpu.data());

    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  GPU[0,0,0,0,0]: " << static_cast<float>(output_gpu(0, 0, 0, 0, 0)) << "\n\n";

    // Validation
    if(verify)
    {
        std::cout << "Step 6: Validation\n";
        std::cout << "------------------\n";

        float max_diff      = 0.0f;
        float max_rel       = 0.0f;
        size_t num_elements = output_gpu.get_element_space_size();

        for(size_t i = 0; i < num_elements; ++i)
        {
            float gpu_val = static_cast<float>(output_gpu.data()[i]);
            float cpu_val = static_cast<float>(output_cpu.data()[i]);
            float diff    = std::abs(gpu_val - cpu_val);
            float rel     = diff / (std::abs(cpu_val) + 1e-6f);
            max_diff      = std::max(max_diff, diff);
            max_rel       = std::max(max_rel, rel);
        }

        bool passed = max_rel < 0.01f; // 1% tolerance

        std::cout << "  Max abs diff: " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff: " << std::scientific << max_rel << "\n";
        std::cout << "  Status: " << (passed ? "PASSED" : "FAILED") << "\n";
    }
#else
    std::cout << "  [Kernel not compiled]\n";
    std::cout << "  Run: python3 scripts/compile_conv_examples.py "
                 "examples/conv/cpp/03_conv_validation.cpp\n";
#endif

    std::cout << "\n======================================================================\n";
    return 0;
}
