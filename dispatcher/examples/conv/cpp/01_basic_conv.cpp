// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 01: Basic Convolution with GPU Execution
 *
 * Demonstrates the Signature/Algorithm/Arch pattern with actual GPU execution.
 *
 * Build:
 *   cd dispatcher/build && cmake .. && make conv_01_basic
 *
 * Complexity: ★★☆☆☆
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
// KERNEL DECLARATIONS
// =============================================================================

DECL_CONV_KERNEL_SET(conv_fwd_kernels,
                     // Forward 2D kernels with different tile sizes
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
    std::cout << "Example 01: Basic Convolution with GPU Execution\n";
    std::cout << "======================================================================\n\n";

    // -------------------------------------------------------------------------
    // Step 1: Show pattern structure
    // -------------------------------------------------------------------------
    std::cout << "Step 1: Signature/Algorithm/Arch Pattern\n";
    std::cout << "-----------------------------------------\n";
    print_pattern_docs();

    // -------------------------------------------------------------------------
    // Step 2: Show declared kernels
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Declared Kernels\n";
    std::cout << "------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_fwd_kernels");
    kernel_set.print(std::cout);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 3: Define problem
    // -------------------------------------------------------------------------
    std::cout << "Step 3: Define Problem\n";
    std::cout << "----------------------\n";

    int N = 1, C = 64, K = 128, Hi = 28, Wi = 28, Y = 3, X = 3;
    auto problem = create_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1, ConvOp::Forward);
    print_problem(problem);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 4: Create registry and dispatcher
    // -------------------------------------------------------------------------
    std::cout << "Step 4: Create Registry\n";
    std::cout << "-----------------------\n";

    ConvRegistry registry;
    registry.set_name("basic_conv_registry");
    registry.register_set(kernel_set, ConvRegistry::Priority::High);

    std::cout << "  Registered " << registry.size() << " kernels\n";
    for(const auto* k : registry.all_kernels())
    {
        std::cout << "    - " << k->name() << "\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 5: Dispatch kernel selection
    // -------------------------------------------------------------------------
    std::cout << "Step 5: Dispatch\n";
    std::cout << "----------------\n";

    ConvDispatcher dispatcher(&registry);
    const auto* selected = dispatcher.select(problem);

    if(selected)
    {
        std::cout << "  Selected: " << selected->name() << "\n\n";
    }
    else
    {
        std::cout << "  No kernel found\n\n";
    }

    // -------------------------------------------------------------------------
    // Step 6: GPU Execution
    // -------------------------------------------------------------------------
    std::cout << "Step 6: GPU Execution\n";
    std::cout << "---------------------\n";

#ifdef CONV_KERNEL_AVAILABLE
    // Create CK Tile conv param
    ck_tile::conv::ConvParam conv_param{
        2,
        1, // num_dim_spatial, groups
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(C),
        {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},
        {static_cast<ck_tile::index_t>(Hi), static_cast<ck_tile::index_t>(Wi)},
        {1, 1},
        {1, 1},
        {1, 1},
        {1, 1} // stride, dilation, left_pad, right_pad
    };

    // Allocate tensors
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

    std::cout << "  Input:  " << input.mDesc << "\n";
    std::cout << "  Weight: " << weight.mDesc << "\n";
    std::cout << "  Output: " << output.mDesc << "\n";

    // Transfer to GPU
    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());
    output_dev.SetZero();

    // Launch kernel
    ck_tile::GroupedConvFwdHostArgs<> args(conv_param,
                                           input_dev.GetDeviceBuffer(),
                                           weight_dev.GetDeviceBuffer(),
                                           {},
                                           output_dev.GetDeviceBuffer(),
                                           1 // k_batch
    );

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 5, 20};
    float elapsed_ms = SelectedConvKernelLauncher::launch(args, stream_cfg);

    double flops  = problem.get_flops();
    double tflops = flops / (elapsed_ms * 1e9);

    std::cout << "  Kernel executed!\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";
#else
    std::cout << "  [Kernel not compiled - generate kernels first]\n";
    std::cout << "  Run: python3 codegen/unified_conv_codegen.py --datatype fp16 --variant forward "
                 "--ndim 2\n";
#endif

    std::cout << "\n======================================================================\n";
    return 0;
}
