// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 07: Convolution JSON Export with GPU Execution
 *
 * Exports kernel configurations to JSON and runs on GPU.
 *
 * Complexity: ★★☆☆☆
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/conv_utils.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;

// =============================================================================
// KERNEL DECLARATIONS
// =============================================================================

DECL_CONV_KERNEL_SET(conv_json_kernels,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv3")
                              .scheduler("intrawave"),
                          "gfx942")
                         .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                              ConvAlgo()
                                  .tile(1, 64, 64)
                                  .wave(2, 2, 1)
                                  .warp(16, 16, 32)
                                  .pipeline("compv3")
                                  .scheduler("intrawave"),
                              "gfx942"));

// =============================================================================
// JSON EXPORT HELPER
// =============================================================================

std::string to_json(const ConvKernelSet& kernel_set)
{
    std::ostringstream json;
    json << "{\n";
    json << "  \"kernels\": [\n";

    const auto& decls = kernel_set.declarations();
    for(size_t i = 0; i < decls.size(); ++i)
    {
        const auto& d = decls[i];
        json << "    {\n";
        json << "      \"name\": \"" << d.name() << "\",\n";
        json << "      \"signature\": {\n";
        json << "        \"dtype_in\": \"" << d.signature.dtype_in_ << "\",\n";
        json << "        \"dtype_out\": \"" << d.signature.dtype_out_ << "\",\n";
        json << "        \"layout\": \"" << d.signature.layout_ << "\",\n";
        json << "        \"direction\": \"" << d.signature.conv_op_ << "\",\n";
        json << "        \"dims\": " << d.signature.num_dims_ << "\n";
        json << "      },\n";
        json << "      \"algorithm\": {\n";
        json << "        \"tile_k\": " << d.algorithm.tile_k_ << ",\n";
        json << "        \"tile_c\": " << d.algorithm.tile_c_ << ",\n";
        json << "        \"pipeline\": \"" << d.algorithm.pipeline_ << "\",\n";
        json << "        \"scheduler\": \"" << d.algorithm.scheduler_ << "\"\n";
        json << "      },\n";
        json << "      \"arch\": \"" << d.arch << "\"\n";
        json << "    }";
        if(i < decls.size() - 1)
            json << ",";
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";
    return json.str();
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

int main()
{
    std::cout << "======================================================================\n";
    std::cout << "Example 07: Convolution JSON Export with GPU Execution\n";
    std::cout << "======================================================================\n\n";

    // -------------------------------------------------------------------------
    // Export to JSON
    // -------------------------------------------------------------------------
    std::cout << "Step 1: Export Kernel Set to JSON\n";
    std::cout << "----------------------------------\n\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_json_kernels");
    std::string json       = to_json(kernel_set);

    std::cout << json << "\n";

    // Write to file
    std::ofstream file("conv_kernels.json");
    if(file)
    {
        file << json;
        file.close();
        std::cout << "[Saved to conv_kernels.json]\n\n";
    }

    // -------------------------------------------------------------------------
    // Setup and run on GPU
    // -------------------------------------------------------------------------
    std::cout << "Step 2: GPU Execution\n";
    std::cout << "---------------------\n";

    ConvRegistry registry;
    registry.register_set(kernel_set, ConvRegistry::Priority::High);
    ConvDispatcher dispatcher(&registry);

    auto problem         = create_conv2d_problem(1, 64, 128, 28, 28, 3, 3, 1, 1);
    const auto* selected = dispatcher.select(problem);

    std::cout << "  Problem: N=1 C=64 K=128 28x28\n";
    std::cout << "  Selected: " << (selected ? selected->name() : "(none)") << "\n";

#ifdef CONV_KERNEL_AVAILABLE
    ck_tile::conv::ConvParam conv_param{
        2,
        1,
        static_cast<ck_tile::index_t>(1),
        static_cast<ck_tile::index_t>(128),
        static_cast<ck_tile::index_t>(64),
        {static_cast<ck_tile::index_t>(3), static_cast<ck_tile::index_t>(3)},
        {static_cast<ck_tile::index_t>(28), static_cast<ck_tile::index_t>(28)},
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
    float elapsed_ms = SelectedConvKernelLauncher::launch(args, stream_cfg);

    double flops  = problem.get_flops();
    double tflops = flops / (elapsed_ms * 1e9);

    std::cout << "  GPU Time: " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS:   " << std::fixed << std::setprecision(2) << tflops << "\n";
#else
    std::cout << "  [GPU execution requires compiled kernels]\n";
#endif

    std::cout << "\n======================================================================\n";
    return 0;
}
