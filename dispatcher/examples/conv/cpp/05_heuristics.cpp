// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 05: Convolution Heuristics with GPU Execution
 *
 * Demonstrates heuristic-based kernel selection with GPU execution.
 *
 * Complexity: ★★★☆☆
 */

#include <iostream>
#include <iomanip>
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
// KERNEL DECLARATIONS
// =============================================================================

DECL_CONV_KERNEL_SET(conv_heuristic_kernels,
                     // Small tile for latency-sensitive workloads
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 64, 64)
                              .wave(2, 2, 1)
                              .warp(16, 16, 32)
                              .pipeline("compv3")
                              .scheduler("intrawave")
                              .vector_sizes(4, 8, 8)
                              .block_per_cu(2),
                          "gfx942")
                         // Large tile for throughput-bound workloads
                         .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                              ConvAlgo()
                                  .tile(1, 128, 128)
                                  .wave(2, 2, 1)
                                  .warp(32, 32, 16)
                                  .pipeline("compv3")
                                  .scheduler("intrawave")
                                  .vector_sizes(4, 8, 8)
                                  .block_per_cu(1),
                              "gfx942"));

// =============================================================================
// HEURISTIC FUNCTION
// =============================================================================

std::string select_tile_size(const ConvProblem& p)
{
    // Heuristic: Use smaller tiles for small spatial dimensions
    int spatial  = p.input_spatial[1] * p.input_spatial[2];
    int channels = p.C * p.K;

    if(spatial < 256)
    {
        return "small"; // 64x64 tiles for small images
    }
    else if(channels > 10000)
    {
        return "large"; // 128x128 tiles for many channels
    }
    else
    {
        return "medium"; // Default
    }
}

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
    ExampleArgs args("Example 05: Conv Heuristics", "Heuristic-based kernel selection");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 05: Convolution Heuristics with GPU Execution\n";
    std::cout << "======================================================================\n\n";

    if(args.has("--list"))
    {
        std::cout << "Declared Kernel Sets:\n";
        ConvKernelSetRegistry::instance().print();
        return 0;
    }

    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------
    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_heuristic_kernels");

    std::cout << "Available kernels:\n";
    kernel_set.print(std::cout);
    std::cout << "\n";

    ConvRegistry registry;
    registry.register_set(kernel_set, ConvRegistry::Priority::High);
    ConvDispatcher dispatcher(&registry);

    // -------------------------------------------------------------------------
    // Test heuristics with different problems
    // -------------------------------------------------------------------------
    std::cout << "Heuristic Selection + GPU Execution:\n";
    std::cout << std::string(60, '-') << "\n\n";

    struct TestCase
    {
        std::string name;
        int N, C, K, H, W;
    };

    std::vector<TestCase> cases = {
        {"Small image (7x7)", 1, 512, 512, 7, 7},
        {"Medium image (28x28)", 1, 128, 256, 28, 28},
        {"Large channels", 1, 256, 512, 14, 14},
    };

#ifdef CONV_KERNEL_AVAILABLE
    for(const auto& tc : cases)
    {
        auto problem = create_conv2d_problem(tc.N, tc.C, tc.K, tc.H, tc.W, 3, 3, 1, 1);

        std::string heuristic_result = select_tile_size(problem);
        const auto* selected         = dispatcher.select(problem);

        std::cout << tc.name << ":\n";
        std::cout << "  Problem: N=" << tc.N << " C=" << tc.C << " K=" << tc.K << " " << tc.H << "x"
                  << tc.W << "\n";
        std::cout << "  Heuristic says: " << heuristic_result << "\n";
        std::cout << "  Dispatcher selected: " << (selected ? selected->name() : "(none)") << "\n";

        // Run on GPU
        ck_tile::conv::ConvParam conv_param{
            2,
            1,
            static_cast<ck_tile::index_t>(tc.N),
            static_cast<ck_tile::index_t>(tc.K),
            static_cast<ck_tile::index_t>(tc.C),
            {static_cast<ck_tile::index_t>(3), static_cast<ck_tile::index_t>(3)},
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

        ck_tile::stream_config stream_cfg{nullptr, true, 1, 5, 20};
        float elapsed_ms = SelectedConvKernelLauncher::launch(kernel_args, stream_cfg);

        double flops  = problem.get_flops();
        double tflops = flops / (elapsed_ms * 1e9);

        std::cout << "  GPU Time: " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
        std::cout << "  TFLOPS:   " << std::fixed << std::setprecision(2) << tflops << "\n\n";
    }
#else
    for(const auto& tc : cases)
    {
        auto problem = create_conv2d_problem(tc.N, tc.C, tc.K, tc.H, tc.W, 3, 3, 1, 1);
        std::string heuristic_result = select_tile_size(problem);

        std::cout << tc.name << ":\n";
        std::cout << "  Heuristic says: " << heuristic_result << "\n";
        std::cout << "  [GPU execution requires compiled kernels]\n\n";
    }
#endif

    std::cout << "======================================================================\n";
    return 0;
}
