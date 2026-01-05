// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 03: Multi-Size Convolution with Multiple Kernels
 *
 * Demonstrates declaring MULTIPLE kernel configurations for different problem sizes.
 * The dispatcher can select the best kernel based on problem characteristics.
 *
 * Build: cd dispatcher/build && cmake .. && make conv_03_multi_size
 */

#include <iostream>
#include <iomanip>
#include <vector>
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
// KERNEL DECLARATIONS - Multiple kernel configurations
// =============================================================================

DECL_CONV_KERNEL_SET(conv_multi_size,
                     // Kernel 1: Small tiles (16x64) - for small problems, higher occupancy
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 16, 64)
                              .wave(1, 4, 1)
                              .warp(16, 16, 32)
                              .pipeline("compv3")
                              .scheduler("intrawave")
                              .vector_sizes(4, 8, 8)
                              .block_per_cu(2),
                          "gfx942")
                         // Kernel 2: Medium tiles (64x64) - balanced
                         .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                              ConvAlgo()
                                  .tile(1, 64, 64)
                                  .wave(2, 2, 1)
                                  .warp(16, 16, 32)
                                  .pipeline("compv3")
                                  .scheduler("intrawave")
                                  .vector_sizes(4, 8, 8)
                                  .block_per_cu(1),
                              "gfx942")
                         // Kernel 3: Large tiles (128x128) - for large problems
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
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 03: Multi-Size Conv", "Multiple kernel configurations");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 03: Multi-Size Convolution (Multiple Kernels)\n";
    std::cout << "======================================================================\n\n";

    // Show declared kernels
    std::cout << "Step 1: Declared Kernels (3 different tile sizes)\n";
    std::cout << "-------------------------------------------------\n";
    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_multi_size");
    std::cout << "  Total declarations: " << kernel_set.size() << "\n";
    for(const auto& d : kernel_set.declarations())
    {
        std::cout << "    - Tile: " << d.algorithm.tile_m_ << "x" << d.algorithm.tile_n_ << "x"
                  << d.algorithm.tile_k_ << "\n";
    }
    std::cout << "\n";

    // Run multiple sizes
    std::cout << "Step 2: GPU Execution for Multiple Sizes\n";
    std::cout << "-----------------------------------------\n\n";

    using Launcher  = generated::FirstKernelLauncher;
    bool all_passed = true;

    std::vector<std::tuple<std::string, int, int, int, int>> problems = {
        {"Small (14x14)", 64, 128, 14, 14},
        {"Medium (28x28)", 64, 128, 28, 28},
        {"Large (56x56)", 128, 256, 56, 56},
    };

    std::cout << std::left << std::setw(18) << "Problem" << std::right << std::setw(8) << "C"
              << std::setw(8) << "K" << std::setw(10) << "HxW" << std::setw(12) << "Time(ms)"
              << std::setw(12) << "TFLOPS" << std::setw(10) << "Status" << "\n";
    std::cout << std::string(78, '-') << "\n";

    for(const auto& [label, C, K, H, W] : problems)
    {
        int N = 1, G = 1, Y = 3, X = 3;

        ck_tile::conv::ConvParam conv_param{
            2,
            static_cast<ck_tile::index_t>(G),
            static_cast<ck_tile::index_t>(N),
            static_cast<ck_tile::index_t>(K),
            static_cast<ck_tile::index_t>(C),
            {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},
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

        ck_tile::stream_config stream_cfg{nullptr, true, 1, 3, 10};
        float elapsed_ms = Launcher::launch(kernel_args, stream_cfg);

        double flops  = 2.0 * G * N * K * C * Y * X * H * W;
        double tflops = flops / (elapsed_ms * 1e9);

        // Basic output check
        output_dev.FromDevice(output.data());
        size_t non_zero = 0;
        for(size_t i = 0; i < output.get_element_space_size(); ++i)
            if(std::abs(static_cast<float>(output.data()[i])) > 1e-6f)
                ++non_zero;
        bool passed = (non_zero > 0);
        if(!passed)
            all_passed = false;

        std::cout << std::left << std::setw(18) << label << std::right << std::setw(8) << C
                  << std::setw(8) << K << std::setw(5) << H << "x" << std::setw(4) << W
                  << std::setw(12) << std::fixed << std::setprecision(4) << elapsed_ms
                  << std::setw(12) << std::setprecision(2) << tflops << std::setw(10)
                  << (passed ? "PASS" : "FAIL") << "\n";
    }

    std::cout << std::string(78, '-') << "\n";
    std::cout << "Overall: " << (all_passed ? "ALL PASSED" : "SOME FAILED") << "\n";

    std::cout << "\n======================================================================\n";
    std::cout << "NOTE: Multiple kernels declared, dispatcher selects best match\n";
    std::cout << "======================================================================\n";

    return all_passed ? 0 : 1;
}
