// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 08: Multiple Convolution Registries with GPU Execution
 *
 * Demonstrates using separate registries for different use cases,
 * each running on GPU.
 *
 * Complexity: ★★★★☆
 */

#include <iostream>
#include <iomanip>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/conv_utils.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;

// =============================================================================
// KERNEL DECLARATIONS - Different registries for different use cases
// =============================================================================

// Throughput-optimized (large tiles, high occupancy)
DECL_CONV_KERNEL_SET(conv_throughput,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv3")
                              .scheduler("intrawave"),
                          "gfx942"));

// Latency-optimized (small tiles, fast completion)
DECL_CONV_KERNEL_SET(conv_latency,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 64, 64)
                              .wave(2, 2, 1)
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

// =============================================================================
// GPU RUN HELPER
// =============================================================================

#ifdef CONV_KERNEL_AVAILABLE
float run_conv(int N, int C, int K, int H, int W)
{
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
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    auto out_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

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

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 5, 20};
    return SelectedConvKernelLauncher::launch(args, stream_cfg);
}
#endif

// =============================================================================
// MAIN
// =============================================================================

int main()
{
    std::cout << "======================================================================\n";
    std::cout << "Example 08: Multiple Convolution Registries with GPU Execution\n";
    std::cout << "======================================================================\n\n";

    // -------------------------------------------------------------------------
    // Create separate registries
    // -------------------------------------------------------------------------
    std::cout << "Step 1: Create Separate Registries\n";
    std::cout << "-----------------------------------\n\n";

    // Throughput registry (inference with batching)
    ConvRegistry throughput_reg;
    throughput_reg.set_name("throughput");
    throughput_reg.register_set(ConvKernelSetRegistry::instance().get("conv_throughput"),
                                ConvRegistry::Priority::High);

    // Latency registry (interactive/real-time)
    ConvRegistry latency_reg;
    latency_reg.set_name("latency");
    latency_reg.register_set(ConvKernelSetRegistry::instance().get("conv_latency"),
                             ConvRegistry::Priority::High);

    std::cout << "Throughput Registry:\n";
    for(const auto* k : throughput_reg.all_kernels())
    {
        std::cout << "  - " << k->name() << "\n";
    }

    std::cout << "\nLatency Registry:\n";
    for(const auto* k : latency_reg.all_kernels())
    {
        std::cout << "  - " << k->name() << "\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Create dispatchers
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Create Dispatchers\n";
    std::cout << "--------------------------\n";

    ConvDispatcher throughput_dispatcher(&throughput_reg);
    ConvDispatcher latency_dispatcher(&latency_reg);

    std::cout << "  Created throughput_dispatcher and latency_dispatcher\n\n";

    // -------------------------------------------------------------------------
    // Run on GPU with different registries
    // -------------------------------------------------------------------------
    std::cout << "Step 3: GPU Execution with Each Registry\n";
    std::cout << "-----------------------------------------\n\n";

    // Large batch (use throughput registry)
    auto large_problem = create_conv2d_problem(4, 128, 256, 56, 56, 3, 3, 1, 1);
    std::cout << "Large batch problem (N=4, 56x56, C=128, K=256):\n";

    const auto* tp_kernel = throughput_dispatcher.select(large_problem);
    std::cout << "  Throughput registry selected: " << (tp_kernel ? tp_kernel->name() : "(none)")
              << "\n";

#ifdef CONV_KERNEL_AVAILABLE
    float tp_time    = run_conv(4, 128, 256, 56, 56);
    double tp_flops  = large_problem.get_flops();
    double tp_tflops = tp_flops / (tp_time * 1e9);
    std::cout << "  GPU Time: " << std::fixed << std::setprecision(4) << tp_time << " ms\n";
    std::cout << "  TFLOPS:   " << std::fixed << std::setprecision(2) << tp_tflops << "\n\n";
#else
    std::cout << "  [GPU execution requires compiled kernels]\n\n";
#endif

    // Small interactive (use latency registry)
    auto small_problem = create_conv2d_problem(1, 64, 64, 14, 14, 3, 3, 1, 1);
    std::cout << "Small interactive problem (N=1, 14x14, C=64, K=64):\n";

    const auto* lat_kernel = latency_dispatcher.select(small_problem);
    std::cout << "  Latency registry selected: " << (lat_kernel ? lat_kernel->name() : "(none)")
              << "\n";

#ifdef CONV_KERNEL_AVAILABLE
    float lat_time    = run_conv(1, 64, 64, 14, 14);
    double lat_flops  = small_problem.get_flops();
    double lat_tflops = lat_flops / (lat_time * 1e9);
    std::cout << "  GPU Time: " << std::fixed << std::setprecision(4) << lat_time << " ms\n";
    std::cout << "  TFLOPS:   " << std::fixed << std::setprecision(2) << lat_tflops << "\n";
#else
    std::cout << "  [GPU execution requires compiled kernels]\n";
#endif

    std::cout << "\n======================================================================\n";
    std::cout << "Use Case Summary:\n";
    std::cout << "  - throughput_dispatcher: Batch inference, training\n";
    std::cout << "  - latency_dispatcher:    Interactive, real-time\n";
    std::cout << "======================================================================\n";

    return 0;
}
