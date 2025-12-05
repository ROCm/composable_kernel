// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 11: Advanced Conv Benchmarking
 *
 * Demonstrates detailed benchmarking with statistics.
 *
 * Build: cd dispatcher/build && cmake .. && make conv_11_advanced_benchmark
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
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

DECL_CONV_KERNEL_SET(benchmark_kernels,
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
    ExampleArgs args("Example 11: Advanced Conv Benchmarking",
                     "Detailed benchmark with statistics");
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "128", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("--size", "56", "Spatial size (H=W)");
    args.add_option("--warmup", "5", "Warmup iterations");
    args.add_option("--iterations", "20", "Benchmark iterations");

    if(!args.parse(argc, argv))
        return 0;

    int N  = args.get_int("-n", 1);
    int C  = args.get_int("-c", 128);
    int K  = args.get_int("-k", 128);
    int Hi = args.get_int("--size", 56);
    int Wi = Hi;
    int Y = 3, X = 3;
    int warmup     = args.get_int("--warmup", 5);
    int iterations = args.get_int("--iterations", 20);

    std::cout << "======================================================================\n";
    std::cout << "Example 11: Advanced Conv Benchmarking\n";
    std::cout << "======================================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Problem:    N=" << N << ", C=" << C << ", K=" << K << ", " << Hi << "x" << Wi
              << "\n";
    std::cout << "  Filter:     " << Y << "x" << X << "\n";
    std::cout << "  Warmup:     " << warmup << " iterations\n";
    std::cout << "  Benchmark:  " << iterations << " iterations\n\n";

    // Allocate
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

    ck_tile::GroupedConvFwdHostArgs<> conv_args(conv_param,
                                                input_dev.GetDeviceBuffer(),
                                                weight_dev.GetDeviceBuffer(),
                                                {},
                                                output_dev.GetDeviceBuffer(),
                                                1);

    using Launcher = generated::FirstKernelLauncher;

    // Warmup
    std::cout << "Running warmup (" << warmup << " iterations)...\n";
    ck_tile::stream_config warmup_cfg{nullptr, true, 0, warmup, 1};
    (void)Launcher::launch(conv_args, warmup_cfg);

    // Benchmark
    std::cout << "Running benchmark (" << iterations << " iterations)...\n";
    std::vector<float> times;
    times.reserve(iterations);

    ck_tile::stream_config bench_cfg{nullptr, true, 0, 0, 1};
    for(int i = 0; i < iterations; ++i)
    {
        output_dev.SetZero();
        float time_ms = Launcher::launch(conv_args, bench_cfg);
        times.push_back(time_ms);
    }

    // Statistics
    std::sort(times.begin(), times.end());
    float min_time    = times.front();
    float max_time    = times.back();
    float median_time = times[times.size() / 2];
    float mean_time   = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();

    // TFLOPS
    int Ho             = Hi;
    int Wo             = Wi;
    double flops       = 2.0 * N * K * C * Ho * Wo * Y * X;
    double min_tflops  = flops / (min_time * 1e9);
    double mean_tflops = flops / (mean_time * 1e9);

    std::cout << "\n======================================================================\n";
    std::cout << "BENCHMARK RESULTS\n";
    std::cout << "======================================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Min time:    " << min_time << " ms (" << std::setprecision(2) << min_tflops
              << " TFLOPS)\n";
    std::cout << std::setprecision(4);
    std::cout << "  Max time:    " << max_time << " ms\n";
    std::cout << "  Mean time:   " << mean_time << " ms (" << std::setprecision(2) << mean_tflops
              << " TFLOPS)\n";
    std::cout << std::setprecision(4);
    std::cout << "  Median time: " << median_time << " ms\n";
    std::cout << "======================================================================\n";

    return 0;
}
