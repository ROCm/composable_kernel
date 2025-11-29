// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 02: 2D Convolution Forward - Declarative with Self-Contained Generation
 *
 * This example demonstrates the complete declarative workflow:
 * 1. Declare kernels using DECL_CONV_KERNEL_SET (Signature/Algorithm/Arch)
 * 2. Generate kernels using the unified codegen
 * 3. Run the convolution with the generated kernel
 *
 * Self-contained build (generates its own kernels):
 *   cd dispatcher
 *   python3 scripts/compile_conv_examples.py examples/conv/cpp/02_conv_forward.cpp
 *
 * Or manual build:
 *   python3 codegen/unified_conv_codegen.py -o build/generated_kernels \
 *           --dtype fp16 --variant forward --ndim 2 --tile-m 128 --tile-n 128
 *   hipcc -std=c++20 -O2 -I include -I ../include -I build/generated_kernels \
 *         -include build/generated_kernels/conv_fwd_fp16_2d_*.hpp \
 *         --offload-arch=gfx942 examples/conv/cpp/02_conv_forward.cpp -o build/conv_02
 *
 * Complexity: ★★☆☆☆
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <hip/hip_runtime.h>

// Use the unified conv utilities
#include "ck_tile/dispatcher/conv_utils.hpp"

// CK Tile core includes
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;

// =============================================================================
// KERNEL DECLARATIONS (Signature/Algorithm/Arch Pattern)
// =============================================================================

// Declare kernels for this example - these will be generated at build time
DECL_CONV_KERNEL_SET(conv_fwd_kernels,
                     // Main kernel: 128x128 tiles, compv4 pipeline
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv4")
                              .scheduler("intrawave"),
                          "gfx942")
                         // Smaller kernel for smaller problems
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

int main(int argc, char* argv[])
{
    std::cout << "======================================================================\n";
    std::cout << "Example 02: 2D Convolution Forward (Declarative)\n";
    std::cout << "======================================================================\n\n";

    // -------------------------------------------------------------------------
    // Step 1: Show declared kernels
    // -------------------------------------------------------------------------
    std::cout << "Step 1: Declared Kernels (Signature/Algorithm/Arch)\n";
    std::cout << "----------------------------------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_fwd_kernels");
    kernel_set.print(std::cout);

    // Print detailed info for first kernel
    if(!kernel_set.declarations().empty())
    {
        std::cout << "\nFirst kernel details:\n";
        print_kernel_decl(kernel_set.declarations()[0]);
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 2: Define problem using utilities
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Define ConvProblem\n";
    std::cout << "--------------------------\n";

    // Parse command line args
    int N = 1, C = 64, K = 128, Hi = 28, Wi = 28, Y = 3, X = 3;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "-n" && i + 1 < argc)
            N = std::stoi(argv[++i]);
        else if(arg == "-c" && i + 1 < argc)
            C = std::stoi(argv[++i]);
        else if(arg == "-k" && i + 1 < argc)
            K = std::stoi(argv[++i]);
        else if(arg == "-h" && i + 1 < argc)
            Hi = Wi = std::stoi(argv[++i]);
        else if(arg == "-y" && i + 1 < argc)
            Y = X = std::stoi(argv[++i]);
    }

    auto problem = create_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1, ConvOp::Forward);
    print_problem(problem);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 3: Create registry and register kernels
    // -------------------------------------------------------------------------
    std::cout << "Step 3: Create Registry\n";
    std::cout << "-----------------------\n";

    ConvRegistry registry;
    registry.set_name("conv_fwd_registry");
    registry.register_set(kernel_set, ConvRegistry::Priority::High);

    std::cout << "  Registered " << registry.size() << " kernels\n";
    for(const auto* k : registry.all_kernels())
    {
        std::cout << "    - " << k->name() << "\n";
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 4: Select kernel using dispatcher
    // -------------------------------------------------------------------------
    std::cout << "Step 4: Select Kernel via Dispatcher\n";
    std::cout << "-------------------------------------\n";

    ConvDispatcher dispatcher(&registry);
    const auto* selected = dispatcher.select(problem);

    if(selected)
    {
        std::cout << "  Selected: " << selected->name() << "\n\n";
    }
    else
    {
        std::cout << "  No kernel selected (expected without compiled kernels)\n\n";
    }

    // -------------------------------------------------------------------------
    // Step 5: Create CK Tile conv param (for actual execution)
    // -------------------------------------------------------------------------
    std::cout << "Step 5: Create CK Tile ConvParam\n";
    std::cout << "---------------------------------\n";

    ck_tile::conv::ConvParam conv_param{
        2, // num_dim_spatial (2D)
        1, // G (groups)
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(C),
        {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},
        {static_cast<ck_tile::index_t>(Hi), static_cast<ck_tile::index_t>(Wi)},
        {1, 1}, // stride
        {1, 1}, // dilation
        {1, 1}, // left pad
        {1, 1}  // right pad
    };

    std::cout << "  Created 2D convolution parameters\n\n";

    // -------------------------------------------------------------------------
    // Step 6: Allocate tensors
    // -------------------------------------------------------------------------
    std::cout << "Step 6: Allocate Tensors\n";
    std::cout << "------------------------\n";

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

    // Initialize
    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
    output.SetZero();

    std::cout << "  Input:  " << input.mDesc << "\n";
    std::cout << "  Weight: " << weight.mDesc << "\n";
    std::cout << "  Output: " << output.mDesc << "\n\n";

    // -------------------------------------------------------------------------
    // Step 7: Transfer to GPU and run
    // -------------------------------------------------------------------------
    std::cout << "Step 7: GPU Execution\n";
    std::cout << "---------------------\n";

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());
    output_dev.SetZero();

    std::cout << "  Data transferred to GPU\n";

#ifdef CONV_KERNEL_AVAILABLE
    // If kernel was generated and compiled, launch it
    ck_tile::GroupedConvFwdHostArgs<> args(conv_param,
                                           input_dev.GetDeviceBuffer(),
                                           weight_dev.GetDeviceBuffer(),
                                           {},
                                           output_dev.GetDeviceBuffer(),
                                           1 // k_batch
    );

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 5, 20};

    // Use generated launcher (SelectedConvKernel is the Config, Launcher has the launch method)
    float elapsed_ms = SelectedConvKernelLauncher::launch(args, stream_cfg);

    double flops  = problem.get_flops();
    double tflops = flops / (elapsed_ms * 1e9);

    std::cout << "  Kernel executed!\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";
#else
    std::cout << "  [Kernel not compiled - run with generated headers]\n";
    std::cout << "  To generate kernels, run:\n";
    std::cout
        << "    python3 scripts/compile_conv_examples.py examples/conv/cpp/02_conv_forward.cpp\n";
#endif

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::cout << "\n======================================================================\n";
    std::cout << "DECLARATIVE PATTERN USED\n";
    std::cout << "======================================================================\n";
    std::cout << R"(
DECL_CONV_KERNEL_SET(conv_fwd_kernels,
    .add(
        ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
        ConvAlgo().tile(1, 128, 128).wave(2, 2, 1).warp(32, 32, 16)
                  .pipeline("compv4").scheduler("intrawave"),
        "gfx942"
    )
);

// Self-contained generation:
python3 scripts/compile_conv_examples.py examples/conv/cpp/02_conv_forward.cpp
)";
    std::cout << "======================================================================\n";

    return 0;
}
