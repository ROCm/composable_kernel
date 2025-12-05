// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 07: Multiple Convolution Registries
 *
 * Demonstrates using separate registries for different use cases.
 *
 * Build: cd dispatcher/build && cmake .. && make conv_07_multi_registry
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
// KERNEL DECLARATIONS - Multiple registry demo (single kernel for simplicity)
// =============================================================================

DECL_CONV_KERNEL_SET(conv_multi_registry,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
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
// HELPER: Run conv on GPU
// =============================================================================

float run_conv(int N, int C, int K, int H, int W)
{
    ck_tile::conv::ConvParam conv_param{
        2,
        1,
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(C),
        {3, 3},
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

    ck_tile::GroupedConvFwdHostArgs<> kernel_args(conv_param,
                                                  input_dev.GetDeviceBuffer(),
                                                  weight_dev.GetDeviceBuffer(),
                                                  {},
                                                  output_dev.GetDeviceBuffer(),
                                                  1);

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 3, 10};

    using Launcher = generated::FirstKernelLauncher;
    return Launcher::launch(kernel_args, stream_cfg);
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 07: Multi-Registry Conv",
                     "Separate registries for different use cases");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 07: Multiple Convolution Registries\n";
    std::cout << "======================================================================\n\n";

    // Create separate registries
    std::cout << "Step 1: Create Separate Registries\n";
    std::cout << "-----------------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_multi_registry");

    ConvRegistry throughput_reg;
    throughput_reg.set_name("throughput");
    throughput_reg.register_set(kernel_set, ConvRegistry::Priority::High);

    ConvRegistry latency_reg;
    latency_reg.set_name("latency");
    latency_reg.register_set(kernel_set, ConvRegistry::Priority::High);

    std::cout << "  Throughput Registry: " << throughput_reg.size() << " kernels\n";
    std::cout << "  Latency Registry:    " << latency_reg.size() << " kernels\n\n";

    // Create dispatchers
    std::cout << "Step 2: Create Dispatchers\n";
    std::cout << "--------------------------\n";

    ConvDispatcher throughput_dispatcher(&throughput_reg);
    ConvDispatcher latency_dispatcher(&latency_reg);

    std::cout << "  Created throughput_dispatcher and latency_dispatcher\n\n";

    // Run with different registries
    std::cout << "Step 3: GPU Execution\n";
    std::cout << "---------------------\n\n";

    // Large batch (throughput registry)
    std::cout << "Large batch (N=4, 56x56, C=128, K=256):\n";
    float tp_time    = run_conv(4, 128, 256, 56, 56);
    double tp_flops  = 2.0 * 4 * 256 * 128 * 9 * 56 * 56;
    double tp_tflops = tp_flops / (tp_time * 1e9);
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << tp_time << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tp_tflops << "\n\n";

    // Small interactive (latency registry)
    std::cout << "Small interactive (N=1, 14x14, C=64, K=64):\n";
    float lat_time    = run_conv(1, 64, 64, 14, 14);
    double lat_flops  = 2.0 * 1 * 64 * 64 * 9 * 14 * 14;
    double lat_tflops = lat_flops / (lat_time * 1e9);
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << lat_time << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << lat_tflops << "\n";

    std::cout << "\n======================================================================\n";
    std::cout << "Use Case Summary:\n";
    std::cout << "  - throughput_dispatcher: Batch inference, training\n";
    std::cout << "  - latency_dispatcher:    Interactive, real-time\n";
    std::cout << "======================================================================\n";

    return 0;
}
