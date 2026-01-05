// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 01: 2D Convolution Forward
 *
 * Demonstrates declarative conv kernel workflow using DECL_CONV_KERNEL_SET.
 * Uses configuration matching CK Tile example/ck_tile/20_grouped_convolution.
 *
 * Build: cd dispatcher/build && cmake .. && make conv_01_forward
 * Usage: ./conv_01_forward [-n N] [-c C] [-k K] [-h H] [-y Y]
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
// KERNEL DECLARATIONS - matches ConvConfigComputeV3 from CK Tile examples
// =============================================================================

// Configuration based on ConvConfigComputeV3
// (example/ck_tile/20_grouped_convolution/conv_configs.hpp) This is a known-working configuration
// for gfx942
DECL_CONV_KERNEL_SET(conv_fwd_kernels,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 16, 64) // M_Tile=16, N_Tile=64 (matches ConvConfigComputeV3)
                              .wave(1, 4, 1)   // M_Warp=1, N_Warp=4, K_Warp=1
                              .warp(16, 16, 32)   // M_Warp_Tile=16, N_Warp_Tile=16, K_Warp_Tile=32
                              .pipeline("compv3") // COMPUTE_V3
                              .scheduler("intrawave") // Intrawave scheduler
                              .epilogue("cshuffle")   // CShuffleEpilogue
                              .vector_sizes(4, 8, 8)  // VectorSizeA=4, B=8, C=8
                              .block_per_cu(1)
                              .num_wave_groups(1)
                              .num_groups_to_merge(1),
                          "gfx942"));

// =============================================================================
// DATA TYPES
// =============================================================================

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;
using AccDataType = float;

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 01: 2D Convolution Forward",
                     "Demonstrates declarative conv kernel workflow");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-g", "1", "Groups G");
    args.add_option("-c", "64", "Input channels C (per group)");
    args.add_option("-k", "128", "Output channels K (per group)");
    args.add_option("-h", "28", "Input height Hi");
    args.add_option("-w", "28", "Input width Wi");
    args.add_option("-y", "3", "Filter height Y");
    args.add_option("-x", "3", "Filter width X");

    if(!args.parse(argc, argv))
        return 0;

    const int N  = args.get_int("-n", 1);
    const int G  = args.get_int("-g", 1);
    const int C  = args.get_int("-c", 64);
    const int K  = args.get_int("-k", 128);
    const int Hi = args.get_int("-h", 28);
    const int Wi = args.get_int("-w", 28);
    const int Y  = args.get_int("-y", 3);
    const int X  = args.get_int("-x", 3);

    std::cout << "======================================================================\n";
    std::cout << "Example 01: 2D Convolution Forward\n";
    std::cout << "======================================================================\n\n";

    // -------------------------------------------------------------------------
    // Step 1: Show declared kernels
    // -------------------------------------------------------------------------
    std::cout << "Step 1: Declared Kernels\n";
    std::cout << "------------------------\n";
    ConvKernelSetRegistry::instance().print();
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Step 2: Create CK Tile conv param
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Problem Configuration\n";
    std::cout << "-----------------------------\n";

    ck_tile::conv::ConvParam conv_param{
        2,                                // num_dim_spatial (2D)
        static_cast<ck_tile::index_t>(G), // groups
        static_cast<ck_tile::index_t>(N), // N
        static_cast<ck_tile::index_t>(K), // K (per group)
        static_cast<ck_tile::index_t>(C), // C (per group)
        {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},   // filter
        {static_cast<ck_tile::index_t>(Hi), static_cast<ck_tile::index_t>(Wi)}, // input spatial
        {1, 1},                                                                 // stride
        {1, 1},                                                                 // dilation
        {1, 1},                                                                 // left pad
        {1, 1}                                                                  // right pad
    };

    // Calculate output spatial dimensions
    auto Ho = (Hi + 1 + 1 - 1 - (Y - 1)) + 1; // (Hi + pad_l + pad_r - dilation*(Y-1)) / stride + 1
    auto Wo = (Wi + 1 + 1 - 1 - (X - 1)) + 1;

    std::cout << "  Input:  N=" << N << ", G=" << G << ", C=" << C << ", Hi=" << Hi << ", Wi=" << Wi
              << "\n";
    std::cout << "  Filter: Y=" << Y << ", X=" << X << ", K=" << K << " (per group)\n";
    std::cout << "  Output: Ho=" << Ho << ", Wo=" << Wo << "\n";
    std::cout << "  Layout: NHWGC (input/output), GKYXC (weight)\n\n";

    // -------------------------------------------------------------------------
    // Step 3: Allocate tensors
    // -------------------------------------------------------------------------
    std::cout << "Step 3: Allocate Tensors\n";
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

    // Initialize with uniform values
    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
    output.SetZero();

    std::cout << "  Input:  " << input.get_element_space_size() << " elements\n";
    std::cout << "  Weight: " << weight.get_element_space_size() << " elements\n";
    std::cout << "  Output: " << output.get_element_space_size() << " elements\n\n";

    // -------------------------------------------------------------------------
    // Step 4: Transfer to GPU and run
    // -------------------------------------------------------------------------
    std::cout << "Step 4: GPU Execution\n";
    std::cout << "---------------------\n";

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());
    output_dev.SetZero();

    ck_tile::GroupedConvFwdHostArgs<> kernel_args(conv_param,
                                                  input_dev.GetDeviceBuffer(),
                                                  weight_dev.GetDeviceBuffer(),
                                                  {}, // no D tensors
                                                  output_dev.GetDeviceBuffer(),
                                                  1 // k_batch
    );

    ck_tile::stream_config stream_cfg{nullptr, true, 1, 5, 20};

    // Use the first generated kernel launcher
    using Launcher   = generated::FirstKernelLauncher;
    float elapsed_ms = Launcher::launch(kernel_args, stream_cfg);

    // Calculate FLOPS
    double flops  = 2.0 * G * N * K * C * Y * X * Ho * Wo;
    double tflops = flops / (elapsed_ms * 1e9);

    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n\n";

    // -------------------------------------------------------------------------
    // Step 5: Basic output check
    // -------------------------------------------------------------------------
    std::cout << "Step 5: Output Check\n";
    std::cout << "--------------------\n";

    output_dev.FromDevice(output.data());

    // Check that output is non-zero (basic sanity check)
    float sum       = 0.0f;
    size_t non_zero = 0;
    for(size_t i = 0; i < output.get_element_space_size(); ++i)
    {
        float val = static_cast<float>(output.mData[i]);
        sum += val;
        if(std::abs(val) > 1e-6f)
            ++non_zero;
    }

    bool passed = (non_zero > 0);
    std::cout << "  Non-zero outputs: " << non_zero << "/" << output.get_element_space_size()
              << "\n";
    std::cout << "  Output sum: " << sum << "\n";
    std::cout << "  Status: " << (passed ? "PASS (kernel executed)" : "FAIL (all zeros)") << "\n";

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::cout << "\n======================================================================\n";
    std::cout << "DECLARATIVE PATTERN:\n";
    std::cout << "======================================================================\n";
    std::cout << R"(
DECL_CONV_KERNEL_SET(conv_fwd_kernels,
    .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
         ConvAlgo()
             .tile(1, 16, 64)
             .wave(1, 4, 1)
             .warp(16, 16, 32)
             .pipeline("compv3")
             .scheduler("intrawave")
             .vector_sizes(4, 8, 8)
             .block_per_cu(1),
         "gfx942")
);
)";
    std::cout << "======================================================================\n";

    return passed ? 0 : 1;
}
