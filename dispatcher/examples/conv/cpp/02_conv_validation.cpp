// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 02: Convolution with CPU Validation
 *
 * Demonstrates:
 * 1. CPU reference verification for correctness
 * 2. Multiple dtype support: fp16 and bf16
 *
 * Build: cd dispatcher/build && cmake .. && make conv_02_validation
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
#include "ck_tile/host/reference/reference_grouped_conv_fwd.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_utils;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL DECLARATIONS - Multiple dtypes (fp16 and bf16)
// =============================================================================

DECL_CONV_KERNEL_SET(conv_validation_kernels,
                     // FP16 kernel (In=fp16, Wei=fp16, Out=fp16, Acc=fp32)
                     .add(ConvSig()
                              .dtype("fp16", "fp16", "fp16", "fp32")
                              .layout("nhwgc")
                              .conv_type("forward")
                              .dims(2),
                          ConvAlgo()
                              .tile(1, 16, 64)
                              .wave(1, 4, 1)
                              .warp(16, 16, 32)
                              .pipeline("compv3")
                              .scheduler("intrawave")
                              .vector_sizes(4, 8, 8)
                              .block_per_cu(1),
                          "gfx942")
                         // BF16 kernel (In=bf16, Wei=bf16, Out=bf16, Acc=fp32)
                         .add(ConvSig()
                                  .dtype("bf16", "bf16", "bf16", "fp32")
                                  .layout("nhwgc")
                                  .conv_type("forward")
                                  .dims(2),
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
// TYPES (FP16 for this example)
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
    ExampleArgs args("Example 02: Conv Validation", "Convolution with CPU reference verification");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-g", "1", "Groups G");
    args.add_option("-c", "64", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("--size", "14", "Spatial size (H=W)");
    args.add_flag("--no-verify", "Skip CPU validation");

    if(!args.parse(argc, argv))
        return 0;

    std::cout << "======================================================================\n";
    std::cout << "Example 02: Convolution with CPU Validation\n";
    std::cout << "======================================================================\n\n";

    // Show declared kernels
    std::cout << "Step 1: Declared Kernels (FP16 + BF16)\n";
    std::cout << "--------------------------------------\n";
    const auto& kernel_set = ConvKernelSetRegistry::instance().get("conv_validation_kernels");
    std::cout << "  Total declarations: " << kernel_set.size() << "\n";
    for(const auto& d : kernel_set.declarations())
    {
        std::cout << "    - dtype: In=" << d.signature.dtype_in_
                  << ", Wei=" << d.signature.dtype_wei_ << ", Out=" << d.signature.dtype_out_
                  << ", Acc=" << d.signature.dtype_acc_ << "\n";
    }
    std::cout << "\n";

    // Define problem
    int N  = args.get_int("-n", 1);
    int G  = args.get_int("-g", 1);
    int C  = args.get_int("-c", 64);
    int K  = args.get_int("-k", 128);
    int Hi = args.get_int("--size", 14);
    int Wi = Hi;
    int Y = 3, X = 3;
    bool verify = !args.has("--no-verify");

    std::cout << "Step 2: Problem Configuration\n";
    std::cout << "-----------------------------\n";
    std::cout << "  Input:  N=" << N << ", G=" << G << ", C=" << C << ", Hi=" << Hi << ", Wi=" << Wi
              << "\n";
    std::cout << "  Filter: Y=" << Y << ", X=" << X << ", K=" << K << "\n";
    std::cout << "  Using FP16 (In/Wei/Out) with FP32 accumulator\n\n";

    // Create CK Tile parameters
    ck_tile::conv::ConvParam conv_param{
        2,
        static_cast<ck_tile::index_t>(G),
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

    // Allocate tensors
    std::cout << "Step 3: Allocate Tensors\n";
    std::cout << "------------------------\n";

    auto in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    auto wei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    auto out_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    ck_tile::HostTensor<InDataType> input(in_desc);
    ck_tile::HostTensor<WeiDataType> weight(wei_desc);
    ck_tile::HostTensor<OutDataType> output_gpu(out_desc);
    ck_tile::HostTensor<OutDataType> output_cpu(out_desc);

    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight);
    output_gpu.SetZero();
    output_cpu.SetZero();

    std::cout << "  Input:  " << input.get_element_space_size() << " elements\n";
    std::cout << "  Weight: " << weight.get_element_space_size() << " elements\n";
    std::cout << "  Output: " << output_gpu.get_element_space_size() << " elements\n\n";

    // CPU Reference
    if(verify)
    {
        std::cout << "Step 4: CPU Reference Computation\n";
        std::cout << "----------------------------------\n";

        std::vector<ck_tile::long_index_t> strides    = {1, 1};
        std::vector<ck_tile::long_index_t> dilations  = {1, 1};
        std::vector<ck_tile::long_index_t> left_pads  = {1, 1};
        std::vector<ck_tile::long_index_t> right_pads = {1, 1};

        ck_tile::reference_grouped_conv_fwd<2, InDataType, WeiDataType, OutDataType>(
            input, weight, output_cpu, strides, dilations, left_pads, right_pads);

        std::cout << "  CPU reference computed\n\n";
    }

    // GPU Execution
    std::cout << "Step 5: GPU Execution\n";
    std::cout << "---------------------\n";

    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output_gpu.get_element_space_size_in_bytes());

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

    using Launcher   = generated::FirstKernelLauncher;
    float elapsed_ms = Launcher::launch(kernel_args, stream_cfg);

    output_dev.FromDevice(output_gpu.data());

    double flops  = 2.0 * G * N * K * C * Y * X * Hi * Wi;
    double tflops = flops / (elapsed_ms * 1e9);

    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n\n";

    // Validation
    bool passed = true;
    if(verify)
    {
        std::cout << "Step 6: Validation\n";
        std::cout << "------------------\n";

        float max_diff      = 0.0f;
        float max_rel       = 0.0f;
        size_t num_elements = output_gpu.get_element_space_size();

        for(size_t i = 0; i < num_elements; ++i)
        {
            float gpu_val = static_cast<float>(output_gpu.data()[i]);
            float cpu_val = static_cast<float>(output_cpu.data()[i]);
            float diff    = std::abs(gpu_val - cpu_val);
            float rel     = diff / (std::abs(cpu_val) + 1e-6f);
            max_diff      = std::max(max_diff, diff);
            max_rel       = std::max(max_rel, rel);
        }

        passed = max_rel < 0.05f; // 5% tolerance for FP16

        std::cout << "  Max abs diff: " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff: " << std::scientific << max_rel << "\n";
        std::cout << "  Status: " << (passed ? "PASSED" : "FAILED") << "\n";
    }

    std::cout << "\n======================================================================\n";
    std::cout << "Multi-dtype support: .dtype(\"fp16\", \"fp16\", \"fp16\", \"fp32\")\n";
    std::cout << "  - In/Wei/Out can be different (e.g., fp16, bf16, fp32)\n";
    std::cout << "  - Accumulator is typically fp32 for precision\n";
    std::cout << "======================================================================\n";

    return passed ? 0 : 1;
}
