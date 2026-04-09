// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Example 01: Basic Grouped Convolution
//
// Demonstrates three declaration patterns (mirrors GEMM 01):
// 1. AUTOFILL  - tile + pipeline only, wave/warp auto-filled
// 2. AUTOCORRECT - invalid wave(1,1,1) corrected to valid config
// 3. FULL - all parameters explicit (matches validated gfx942 config)
//
// Then runs the forward convolution on GPU and verifies output.
//
// Build: cd dispatcher/build && cmake .. && make grouped_conv_01_basic

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

#include "ck_tile/dispatcher/grouped_conv_utils.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_utils;
using GroupedConvSig  = grouped_conv_decl::GroupedConvSignature;
using GroupedConvAlgo = grouped_conv_decl::GroupedConvAlgorithm;

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;

// Three declaration patterns -- codegen auto-fills/auto-corrects as needed
DECL_GROUPED_CONV_KERNEL_SET(
    basic_conv_kernels,
    // Pattern 1: AUTOFILL - only tile + pipeline, rest auto-filled
    .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
         GroupedConvAlgo().tile(1, 128, 128).pipeline("compv4").scheduler("intrawave"),
         "gfx950")
        // Pattern 2: AUTOCORRECT - wave(1,1,1) invalid, corrected to (1,4,1)
        .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
             GroupedConvAlgo()
                 .tile(1, 64, 64)
                 .wave(1, 1, 1)
                 .warp(16, 16, 32)
                 .pipeline("compv3")
                 .scheduler("intrawave")
                 .epilogue("cshuffle")
                 .vector_sizes(4, 8, 8),
             "gfx950")
        // Pattern 3: FULL - all parameters explicit (validated config)
        .add(GroupedConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
             GroupedConvAlgo()
                 .tile(1, 128, 128)
                 .wave(2, 2, 1)
                 .warp(32, 32, 16)
                 .pipeline("compv3")
                 .scheduler("intrawave")
                 .epilogue("cshuffle")
                 .vector_sizes(4, 8, 8)
                 .block_per_cu(1),
             "gfx950"));

int main(int argc, char* argv[])
{
    utils::ExampleArgs args("Example 01: Basic Grouped Convolution",
                            "Declaration patterns + GPU execution");
    args.add_option("--arch", "gfx950", "GPU architecture");
    args.add_option("--size", "14", "Spatial size (H=W)");
    args.add_option("-n", "1", "Batch size");
    args.add_option("-g", "1", "Groups");
    args.add_option("-c", "64", "Input channels C");
    args.add_option("-k", "128", "Output channels K");

    if(!args.parse(argc, argv))
        return 0;

    utils::print_header("Example 01: Basic Grouped Convolution");

    std::string gfx_arch = args.get("--arch", "gfx950");
    int N                = args.get_int("-n", 1);
    int G                = args.get_int("-g", 1);
    int C                = args.get_int("-c", 64);
    int K                = args.get_int("-k", 128);
    int HW               = args.get_int("--size", 14);
    int Y = 3, X = 3;

    // Step 1: Show declared kernel sets
    std::cout << "\nStep 1: Declared Kernel Sets\n";
    GroupedConvKernelSetRegistry::instance().print();

    // Step 2: Register kernels
    std::cout << "\nStep 2: Register Kernels\n";
    GroupedConvRegistry registry;
    registry.set_name("basic_conv");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    // Step 3: Create dispatcher
    std::cout << "\nStep 3: Create Dispatcher\n";
    GroupedConvDispatcher dispatcher(&registry);

    // Step 4: Build problem using CK Tile ConvParam
    std::cout << "\nStep 4: Problem\n";
    auto problem = create_grouped_conv2d_problem(N, C, K, HW, HW, Y, X, 1, 1);
    problem.op   = GroupedConvOp::Forward;
    print_grouped_conv_problem(problem);

    ck_tile::conv::ConvParam conv_param{
        2,
        static_cast<ck_tile::index_t>(G),
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(C),
        {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},
        {static_cast<ck_tile::index_t>(HW), static_cast<ck_tile::index_t>(HW)},
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

    ck_tile::HostTensor<InDataType> input_host(in_desc);
    ck_tile::HostTensor<WeiDataType> weight_host(wei_desc);
    ck_tile::HostTensor<OutDataType> output_host(out_desc);

    ck_tile::FillUniformDistribution<InDataType>{-0.5f, 0.5f}(input_host);
    ck_tile::FillUniformDistribution<WeiDataType>{-0.5f, 0.5f}(weight_host);

    ck_tile::DeviceMem input_dev(input_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output_host.get_element_space_size_in_bytes());

    input_dev.ToDevice(input_host.data());
    weight_dev.ToDevice(weight_host.data());

    // Step 5: Select and run
    std::cout << "\nStep 5: Select and Run\n";

    auto* selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cerr << "  ERROR: No kernel found for problem!\n";
        return 1;
    }
    std::cout << "  Selected: " << selected->name() << "\n";

    float time_ms = dispatcher.run(input_dev.GetDeviceBuffer(),
                                   weight_dev.GetDeviceBuffer(),
                                   output_dev.GetDeviceBuffer(),
                                   problem,
                                   nullptr);

    double tflops = calculate_conv_tflops(problem, time_ms);
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n";

    // Step 6: Verify
    std::cout << "\nStep 6: Verify\n";
    output_dev.FromDevice(output_host.data());

    size_t total    = output_host.get_element_space_size();
    size_t nonzero  = 0;
    double checksum = 0.0;
    for(size_t i = 0; i < total; ++i)
    {
        float v = static_cast<float>(output_host.data()[i]);
        if(v != 0.0f)
            ++nonzero;
        checksum += v;
    }

    bool passed = nonzero > 0;
    std::cout << "  Output elements: " << total << "\n";
    std::cout << "  Non-zero: " << nonzero << "/" << total
              << (nonzero > 0 ? " (kernel produced output)" : " WARNING: all zeros!") << "\n";
    std::cout << "  Checksum: " << std::fixed << std::setprecision(2) << checksum << "\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";

    utils::print_separator();
    std::cout << "DECLARATION PATTERNS:\n";
    std::cout << "  1. AUTOFILL:     tile + pipeline only, wave/warp auto-filled\n";
    std::cout << "  2. AUTOCORRECT:  invalid wave(1,1,1) corrected\n";
    std::cout << "  3. FULL:         all parameters explicit\n";
    utils::print_separator();

    return passed ? 0 : 1;
}
