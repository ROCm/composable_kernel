// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 04: Advanced Convolution Benchmark
 *
 * Demonstrates benchmark parameters matching CK Tile stream_config.
 *
 * Build: cd dispatcher/build && cmake .. && make conv_04_benchmark
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
// KERNEL DECLARATIONS
// =============================================================================

DECL_CONV_KERNEL_SET(conv_benchmark,
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
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 04: Conv Benchmark", "Benchmark with stream_config parameters");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "128", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("-h", "28", "Input height/width H=W");
    args.add_option("-y", "3", "Filter size Y=X");
    args.add_option("--warmup", "5", "Warmup iterations");
    args.add_option("--repeat", "20", "Benchmark iterations");

    if(!args.parse(argc, argv))
        return 0;

    int N      = args.get_int("-n", 1);
    int C      = args.get_int("-c", 128);
    int K      = args.get_int("-k", 128);
    int Hi     = args.get_int("-h", 28);
    int Wi     = Hi;
    int Y      = args.get_int("-y", 3);
    int X      = Y;
    int warmup = args.get_int("--warmup", 5);
    int repeat = args.get_int("--repeat", 20);

    std::cout << "======================================================================\n";
    std::cout << "Example 04: Convolution Benchmark\n";
    std::cout << "======================================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Problem: N=" << N << ", C=" << C << ", K=" << K << ", " << Hi << "x" << Wi
              << "\n";
    std::cout << "  Filter:  " << Y << "x" << X << "\n";
    std::cout << "  Warmup:  " << warmup << " iterations\n";
    std::cout << "  Repeat:  " << repeat << " iterations\n\n";

    // Create conv param
    ck_tile::conv::ConvParam conv_param{
        2,
        1,
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(C),
        {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},
        {static_cast<ck_tile::index_t>(Hi), static_cast<ck_tile::index_t>(Wi)},
        {1, 1},
        {1, 1},
        {1, 1},
        {1, 1}};

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

    std::cout << "Tensors:\n";
    std::cout << "  Input:  " << input.get_element_space_size() << " elements\n";
    std::cout << "  Weight: " << weight.get_element_space_size() << " elements\n";
    std::cout << "  Output: " << output.get_element_space_size() << " elements\n\n";

    // Transfer to GPU
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

    // Run benchmark
    ck_tile::stream_config stream_cfg{nullptr, true, 1, warmup, repeat};

    std::cout << "Running Benchmark...\n";
    std::cout << "----------------------------------------------------------------------\n";

    using Launcher    = generated::FirstKernelLauncher;
    float avg_time_ms = Launcher::launch(kernel_args, stream_cfg);

    double flops  = 2.0 * N * K * C * Y * X * Hi * Wi;
    double tflops = flops / (avg_time_ms * 1e9);
    double bandwidth_gb =
        (input.get_element_space_size_in_bytes() + weight.get_element_space_size_in_bytes() +
         output.get_element_space_size_in_bytes()) /
        1e9 / (avg_time_ms / 1000);

    std::cout << "\n*** BENCHMARK RESULTS ***\n";
    std::cout << "  Average Time: " << std::fixed << std::setprecision(4) << avg_time_ms << " ms\n";
    std::cout << "  TFLOPS:       " << std::setprecision(2) << tflops << "\n";
    std::cout << "  Bandwidth:    " << bandwidth_gb << " GB/s\n";

    std::cout << "\n======================================================================\n";
    return 0;
}
