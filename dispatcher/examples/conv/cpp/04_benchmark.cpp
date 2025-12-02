// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 04: Convolution Benchmark with GPU Execution
 *
 * Benchmarks different kernel configurations on actual GPU hardware.
 *
 * Usage:
 *   ./conv_04_benchmark
 *   ./conv_04_benchmark --help
 *   ./conv_04_benchmark --warmup 10 --iterations 100
 *
 * Complexity: ★★★☆☆
 */

#include <iostream>
#include <iomanip>
#include <vector>
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
// KERNEL DECLARATIONS - Benchmark configurations
// =============================================================================

DECL_CONV_KERNEL_SET(conv_benchmark,
                     // CompV3 pipeline
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv3")
                              .scheduler("intrawave"),
                          "gfx942")
                         // CompV4 pipeline (usually faster)
                         .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                              ConvAlgo()
                                  .tile(1, 128, 128)
                                  .wave(2, 2, 1)
                                  .warp(32, 32, 16)
                                  .pipeline("compv4")
                                  .scheduler("intrawave"),
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
    // Parse command line arguments
    ExampleArgs args("Example 04: Convolution Benchmark",
                     "Benchmarks conv kernel configurations on GPU");
    args.add_option("--warmup", "10", "Warmup iterations");
    args.add_option("--iterations", "50", "Benchmark iterations");

    if(!args.parse(argc, argv))
    {
        return 0; // --help was printed
    }

    int warmup     = args.get_int("--warmup", 10);
    int iterations = args.get_int("--iterations", 50);

    std::cout << "======================================================================\n";
    std::cout << "Example 04: Convolution Benchmark with GPU Execution\n";
    std::cout << "======================================================================\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Warmup iterations:    " << warmup << "\n";
    std::cout << "  Benchmark iterations: " << iterations << "\n\n";

    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------
    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_benchmark");

    std::cout << "Kernels to benchmark:\n";
    kernel_set.print(std::cout);
    std::cout << "\n";

    ConvRegistry registry;
    registry.register_set(kernel_set, ConvRegistry::Priority::High);
    ConvDispatcher dispatcher(&registry);

    // -------------------------------------------------------------------------
    // Benchmark problems
    // -------------------------------------------------------------------------
    std::cout << "Benchmark Results:\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::setw(30) << "Problem" << std::setw(15) << "Time (ms)" << std::setw(15)
              << "TFLOPS" << std::setw(10) << "Status" << "\n";
    std::cout << std::string(70, '-') << "\n";

    std::vector<std::tuple<std::string, int, int, int, int, int>> problems = {
        {"ResNet50 Layer1", 1, 64, 64, 56, 56},
        {"ResNet50 Layer2", 1, 128, 128, 28, 28},
        {"ResNet50 Layer3", 1, 256, 256, 14, 14},
        {"ResNet50 Layer4", 1, 512, 512, 7, 7},
        {"VGG-16 Conv1", 1, 64, 64, 224, 224},
        {"VGG-16 Conv2", 1, 128, 128, 112, 112},
    };

#ifdef CONV_KERNEL_AVAILABLE
    for(const auto& [label, N, C, K, H, W] : problems)
    {
        auto problem = create_conv2d_problem(N, C, K, H, W, 3, 3, 1, 1);

        // Create conv param
        ck_tile::conv::ConvParam conv_param{
            2,
            1,
            static_cast<ck_tile::index_t>(N),
            static_cast<ck_tile::index_t>(K),
            static_cast<ck_tile::index_t>(C),
            {static_cast<ck_tile::index_t>(3), static_cast<ck_tile::index_t>(3)},
            {static_cast<ck_tile::index_t>(H), static_cast<ck_tile::index_t>(W)},
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

        ck_tile::GroupedConvFwdHostArgs<> args(conv_param,
                                               input_dev.GetDeviceBuffer(),
                                               weight_dev.GetDeviceBuffer(),
                                               {},
                                               output_dev.GetDeviceBuffer(),
                                               1);

        ck_tile::stream_config stream_cfg{nullptr, true, 1, warmup, iterations};
        float elapsed_ms = SelectedConvKernelLauncher::launch(args, stream_cfg);

        double flops  = problem.get_flops();
        double tflops = flops / (elapsed_ms * 1e9);

        std::cout << std::setw(30) << label << std::setw(15) << std::fixed << std::setprecision(4)
                  << elapsed_ms << std::setw(15) << std::fixed << std::setprecision(2) << tflops
                  << std::setw(10) << "OK" << "\n";
    }
#else
    for(const auto& [label, N, C, K, H, W] : problems)
    {
        (void)N;
        (void)C;
        (void)K;
        (void)H;
        (void)W;
        std::cout << std::setw(30) << label << std::setw(15) << "-" << std::setw(15) << "-"
                  << std::setw(10) << "NO KERNEL" << "\n";
    }
    std::cout << "\n[Kernels not compiled - generate with unified_conv_codegen.py]\n";
#endif

    std::cout << std::string(70, '-') << "\n";
    std::cout << "\n======================================================================\n";
    return 0;
}
