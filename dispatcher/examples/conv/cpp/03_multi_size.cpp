// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 03: Multi-Size Convolution with GPU Execution
 *
 * Demonstrates using different kernel tile sizes for different problem sizes,
 * with actual GPU execution for each.
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
// KERNEL DECLARATIONS - Multiple tile sizes
// =============================================================================

DECL_CONV_KERNEL_SET(conv_multi_size,
                     // Small tiles (64x64) - for small problems, higher occupancy
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
                         // Medium tiles (128x128) - balanced
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
// DATA TYPES
// =============================================================================

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;

// =============================================================================
// GPU RUN HELPER
// =============================================================================

#ifdef CONV_KERNEL_AVAILABLE
void run_conv_on_gpu(const ConvProblem& problem, const std::string& label)
{
    std::cout << "  Running " << label << " on GPU...\n";

    int N = problem.N, C = problem.C, K = problem.K;
    int Hi = problem.input_spatial[1], Wi = problem.input_spatial[2];
    int Y = problem.filter_spatial[1], X = problem.filter_spatial[2];

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

    auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    auto wei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    auto out_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    ck_tile::HostTensor<InDataType> input(in_desc);
    ck_tile::HostTensor<WeiDataType> weight(wei_desc);
    ck_tile::HostTensor<OutDataType> output(out_desc);

    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
    output.SetZero();

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

    std::cout << "    Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "    TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";
}
#endif

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 03: Multi-Size Conv",
                     "Different tile sizes for different problem sizes");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 03: Multi-Size Convolution with GPU Execution\n";
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
    std::cout << "Step 1: Declared Kernel Sets\n";
    std::cout << "----------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_multi_size");
    kernel_set.print(std::cout);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 2: Create registry
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Create Registry\n";
    std::cout << "-----------------------\n";

    ConvRegistry registry;
    registry.set_name("multi_size_registry");
    registry.register_set(kernel_set, ConvRegistry::Priority::High);

    std::cout << "  Total kernels: " << registry.size() << "\n\n";

    // -------------------------------------------------------------------------
    // Step 3: Run multiple problem sizes on GPU
    // -------------------------------------------------------------------------
    std::cout << "Step 3: GPU Execution for Multiple Problem Sizes\n";
    std::cout << "------------------------------------------------\n\n";

    std::vector<std::tuple<std::string, int, int, int, int, int>> problems = {
        {"Small (14x14)", 1, 32, 64, 14, 14},
        {"Medium (28x28)", 1, 64, 128, 28, 28},
        {"Large (56x56)", 1, 128, 256, 56, 56},
    };

    ConvDispatcher dispatcher(&registry);

    for(const auto& [label, N, C, K, H, W] : problems)
    {
        auto problem = create_conv2d_problem(N, C, K, H, W, 3, 3, 1, 1);

        std::cout << label << " - N=" << N << " C=" << C << " K=" << K << " " << H << "x" << W
                  << ":\n";
        std::cout << "  FLOPs: " << std::scientific << std::setprecision(2) << problem.get_flops()
                  << "\n";

        const auto* selected = dispatcher.select(problem);
        std::cout << "  Selected: " << (selected ? selected->name() : "(none)") << "\n";

#ifdef CONV_KERNEL_AVAILABLE
        run_conv_on_gpu(problem, label);
#else
        std::cout << "  [GPU execution requires compiled kernels]\n";
#endif
        std::cout << "\n";
    }

    std::cout << "======================================================================\n";
    return 0;
}
