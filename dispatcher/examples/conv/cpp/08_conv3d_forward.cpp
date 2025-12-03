// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 08: 3D Convolution Forward with GPU Execution
 *
 * Demonstrates 3D convolution (e.g., for video or volumetric data).
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
// KERNEL DECLARATIONS - 3D Forward
// =============================================================================

DECL_CONV_KERNEL_SET(conv3d_fwd_kernels,
                     .add(ConvSig().dtype("fp16").layout("ndhwgc").conv_type("forward").dims(3),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv3")
                              .scheduler("intrawave"),
                          "gfx942")
                         .add(ConvSig().dtype("fp16").layout("ndhwgc").conv_type("forward").dims(3),
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
    ExampleArgs args("Example 08: Conv3D Forward", "3D convolution for video/volumetric data");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "32", "Input channels C");
    args.add_option("-k", "64", "Output channels K");
    args.add_option("--depth", "8", "Depth D");
    args.add_option("--size", "16", "Spatial size (H=W)");
    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 08: 3D Convolution Forward with GPU Execution\n";
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
    std::cout << "Step 1: Declared 3D Kernels\n";
    std::cout << "---------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv3d_fwd_kernels");
    kernel_set.print(std::cout);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 2: Define 3D problem
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Define 3D Problem\n";
    std::cout << "-------------------------\n";

    // 3D problem from args
    int N  = args.get_int("-n", 1);
    int C  = args.get_int("-c", 32);
    int K  = args.get_int("-k", 64);
    int Di = args.get_int("--depth", 8);
    int Hi = args.get_int("--size", 16);
    int Wi = Hi;
    int Z = 3, Y = 3, X = 3;

    auto problem = create_conv3d_problem(N, C, K, Di, Hi, Wi, Z, Y, X, 1, 1, ConvOp::Forward);
    print_problem(problem);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 3: Create registry
    // -------------------------------------------------------------------------
    std::cout << "Step 3: Create Registry\n";
    std::cout << "-----------------------\n";

    ConvRegistry registry;
    registry.register_set(kernel_set, ConvRegistry::Priority::High);
    std::cout << "  Registered " << registry.size() << " kernels\n\n";

    // -------------------------------------------------------------------------
    // Step 4: GPU Execution
    // -------------------------------------------------------------------------
    std::cout << "Step 4: GPU Execution\n";
    std::cout << "---------------------\n";

#ifdef CONV_KERNEL_AVAILABLE
    // Create 3D conv param
    ck_tile::conv::ConvParam conv_param{3,
                                        1, // 3D, 1 group
                                        static_cast<ck_tile::index_t>(N),
                                        static_cast<ck_tile::index_t>(K),
                                        static_cast<ck_tile::index_t>(C),
                                        {static_cast<ck_tile::index_t>(Z),
                                         static_cast<ck_tile::index_t>(Y),
                                         static_cast<ck_tile::index_t>(X)},
                                        {static_cast<ck_tile::index_t>(Di),
                                         static_cast<ck_tile::index_t>(Hi),
                                         static_cast<ck_tile::index_t>(Wi)},
                                        {1, 1, 1},
                                        {1, 1, 1},
                                        {1, 1, 1},
                                        {1, 1, 1}};

    using InLayout  = ck_tile::tensor_layout::convolution::NDHWGC;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKZYXC;
    using OutLayout = ck_tile::tensor_layout::convolution::NDHWGK;

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

    std::cout << "  Input (3D):  " << input.mDesc << "\n";
    std::cout << "  Weight:      " << weight.mDesc << "\n";
    std::cout << "  Output (3D): " << output.mDesc << "\n";

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

    std::cout << "\n  *** 3D CONV GPU EXECUTION ***\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";
#else
    std::cout << "  [Kernel not compiled]\n";
    std::cout << "  Generate with: python3 codegen/unified_conv_codegen.py --ndim 3\n";
#endif

    std::cout << "\n======================================================================\n";
    std::cout << "3D Convolution: Used for video, medical imaging, volumetric data\n";
    std::cout << "======================================================================\n";

    return 0;
}
