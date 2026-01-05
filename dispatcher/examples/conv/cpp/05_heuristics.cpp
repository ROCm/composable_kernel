// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 05: Convolution with Autocorrect/Auto-fill
 *
 * Demonstrates the autocorrect functionality where kernels declared with
 * minimal parameters are automatically filled with sensible defaults.
 *
 * You can declare a kernel with just dtype, layout, conv_type and tile size,
 * and the system will fill in wave, warp, pipeline, etc. with defaults.
 *
 * Build: cd dispatcher/build && cmake .. && make conv_05_heuristics
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
// KERNEL DECLARATIONS - Minimal specification (autocorrect fills the rest)
// =============================================================================

// This declaration uses minimal parameters - the build system will auto-fill:
// - wave(1, 4, 1) - default wave distribution
// - warp(16, 16, 32) - default warp tile sizes
// - vector_sizes(4, 8, 8) - default vector sizes
// - block_per_cu(1) - default occupancy hint
DECL_CONV_KERNEL_SET(conv_heuristic_kernels,
                     // Minimal declaration: just dtype, layout, conv_type, tile, pipeline
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 16, 64)         // Required: tile size
                              .pipeline("compv3")      // Required: pipeline
                              .scheduler("intrawave"), // Required: scheduler
                          "gfx942"));

// =============================================================================
// DATA TYPES
// =============================================================================

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;

// =============================================================================
// HEURISTIC FUNCTION
// =============================================================================

std::string select_tile_heuristic(int C, int K, int spatial)
{
    // Simple heuristic based on problem characteristics
    if(spatial < 256)
        return "small_tile"; // Small spatial: use smaller tiles
    else if(C * K > 20000)
        return "large_tile"; // Many channels: use larger tiles
    else
        return "medium_tile"; // Balanced
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 05: Conv Autocorrect", "Demonstrates auto-fill of kernel parameters");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 05: Convolution with Autocorrect\n";
    std::cout << "======================================================================\n\n";

    // Show autocorrect concept
    std::cout << "AUTOCORRECT FUNCTIONALITY:\n";
    std::cout << "--------------------------\n";
    std::cout << "Minimal declaration:\n";
    std::cout
        << "  .add(ConvSig().dtype(\"fp16\").layout(\"nhwgc\").conv_type(\"forward\").dims(2),\n";
    std::cout
        << "       ConvAlgo().tile(1, 16, 64).pipeline(\"compv3\").scheduler(\"intrawave\"),\n";
    std::cout << "       \"gfx942\")\n\n";
    std::cout << "Auto-filled parameters:\n";
    std::cout << "  - wave(1, 4, 1)        <- default wave distribution\n";
    std::cout << "  - warp(16, 16, 32)     <- default warp tile sizes\n";
    std::cout << "  - vector_sizes(4,8,8)  <- default vector sizes\n";
    std::cout << "  - block_per_cu(1)      <- default occupancy\n\n";

    // Show declared kernels
    std::cout << "Step 1: Declared Kernels\n";
    std::cout << "------------------------\n";
    ConvKernelSetRegistry::instance().print();
    std::cout << "\n";

    // Run GPU test
    std::cout << "Step 2: GPU Execution with Heuristic Selection\n";
    std::cout << "-----------------------------------------------\n\n";

    using Launcher  = generated::FirstKernelLauncher;
    bool all_passed = true;

    struct TestCase
    {
        std::string name;
        int C, K, H, W;
    };
    std::vector<TestCase> cases = {
        {"Small spatial (7x7)", 256, 256, 7, 7},
        {"Medium (28x28)", 128, 256, 28, 28},
        {"Large channels", 256, 512, 14, 14},
    };

    for(const auto& tc : cases)
    {
        std::string heuristic = select_tile_heuristic(tc.C, tc.K, tc.H * tc.W);

        std::cout << tc.name << ":\n";
        std::cout << "  Problem: C=" << tc.C << " K=" << tc.K << " " << tc.H << "x" << tc.W << "\n";
        std::cout << "  Heuristic suggests: " << heuristic << "\n";

        int N = 1, G = 1, Y = 3, X = 3;

        ck_tile::conv::ConvParam conv_param{
            2,
            static_cast<ck_tile::index_t>(G),
            static_cast<ck_tile::index_t>(N),
            static_cast<ck_tile::index_t>(tc.K),
            static_cast<ck_tile::index_t>(tc.C),
            {3, 3},
            {static_cast<ck_tile::index_t>(tc.H), static_cast<ck_tile::index_t>(tc.W)},
            {1, 1},
            {1, 1},
            {1, 1},
            {1, 1}};

        using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
        using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;
        using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;

        auto in_desc =
            ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
        auto wei_desc =
            ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);
        auto out_desc =
            ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        ck_tile::HostTensor<InDataType> input(in_desc);
        ck_tile::HostTensor<WeiDataType> weight(wei_desc);
        ck_tile::HostTensor<OutDataType> output(out_desc);

        ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
        ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);

        ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
        ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
        ck_tile::DeviceMem output_dev(output.get_element_space_size_in_bytes());

        input_dev.ToDevice(input.data());
        weight_dev.ToDevice(weight.data());
        output_dev.SetZero();

        ck_tile::GroupedConvFwdHostArgs<> kernel_args(conv_param,
                                                      input_dev.GetDeviceBuffer(),
                                                      weight_dev.GetDeviceBuffer(),
                                                      {},
                                                      output_dev.GetDeviceBuffer(),
                                                      1);

        ck_tile::stream_config stream_cfg{nullptr, true, 1, 3, 10};
        float elapsed_ms = Launcher::launch(kernel_args, stream_cfg);

        double flops  = 2.0 * G * N * tc.K * tc.C * Y * X * tc.H * tc.W;
        double tflops = flops / (elapsed_ms * 1e9);

        // Output check
        output_dev.FromDevice(output.data());
        size_t non_zero = 0;
        for(size_t i = 0; i < output.get_element_space_size(); ++i)
            if(std::abs(static_cast<float>(output.data()[i])) > 1e-6f)
                ++non_zero;
        bool passed = (non_zero > 0);
        if(!passed)
            all_passed = false;

        std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
        std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";
        std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n\n";
    }

    std::cout << "Overall: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << "\n";
    std::cout << "======================================================================\n";
    return all_passed ? 0 : 1;
}
