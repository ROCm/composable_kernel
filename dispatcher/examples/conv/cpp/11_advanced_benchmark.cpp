// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 11: Advanced Conv Benchmarking
 *
 * Demonstrates all available benchmark parameters matching CK Tile stream_config:
 *   - warmup: Number of warmup iterations (default: 5)
 *   - iterations: Number of benchmark iterations (default: 100)
 *   - flush_cache: Flush GPU L2 cache between iterations (default: false)
 *   - rotating_count: Number of rotating buffers for cache simulation (default: 1)
 *   - timer: Use GPU timer (HIP events) or CPU timer (default: gpu)
 *   - init: Initialization method - random, linear, constant (default: random)
 *
 * Build:
 *   cd dispatcher/build && cmake .. -DBUILD_DISPATCHER_EXAMPLES=ON && make
 * conv_11_advanced_benchmark
 *
 * Usage:
 *   ./conv_11_advanced_benchmark
 *   ./conv_11_advanced_benchmark --help
 *   ./conv_11_advanced_benchmark -n 4 -c 256 -k 512 --size 56 --warmup 10 --iterations 100
 *   ./conv_11_advanced_benchmark --flush-cache --rotating-count 4
 *
 * Complexity: ★★★☆☆
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
// KERNEL DECLARATIONS - High performance kernel for benchmarking
// =============================================================================

DECL_CONV_KERNEL_SET(benchmark_kernels,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv3")
                              .scheduler("intrawave"),
                          "gfx942"));

// =============================================================================
// DATA TYPES
// =============================================================================

using InDataType  = ck_tile::half_t;
using WeiDataType = ck_tile::half_t;
using OutDataType = ck_tile::half_t;
using AccDataType = float;

// =============================================================================
// INITIALIZATION METHODS
// =============================================================================

template <typename T>
void fill_random(ck_tile::HostTensor<T>& tensor)
{
    ck_tile::FillUniformDistribution<T>{-0.5f, 0.5f}(tensor);
}

template <typename T>
void fill_linear(ck_tile::HostTensor<T>& tensor)
{
    size_t n = tensor.get_element_space_size();
    for(size_t i = 0; i < n; ++i)
    {
        tensor.data()[i] = static_cast<T>(static_cast<float>(i % 256) / 256.0f - 0.5f);
    }
}

template <typename T>
void fill_constant(ck_tile::HostTensor<T>& tensor, float value = 1.0f)
{
    size_t n = tensor.get_element_space_size();
    for(size_t i = 0; i < n; ++i)
    {
        tensor.data()[i] = static_cast<T>(value);
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 11: Advanced Conv Benchmarking",
                     "All benchmark parameters: warmup, iterations, cache flush, rotating buffers");

    // Problem dimensions
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "256", "Input channels C");
    args.add_option("-k", "256", "Output channels K");
    args.add_option("--size", "56", "Spatial size (H=W)");
    args.add_option("-y", "3", "Filter height");
    args.add_option("-x", "3", "Filter width");
    args.add_option("--stride", "1", "Convolution stride");
    args.add_option("--pad", "1", "Convolution padding");

    // Benchmark parameters
    args.add_option("--warmup", "5", "Warmup iterations");
    args.add_option("--iterations", "100", "Benchmark iterations");
    args.add_flag("--flush-cache", "Flush GPU L2 cache between iterations");
    args.add_option("--rotating-count", "1", "Rotating buffer count for cache simulation");
    args.add_option("--timer", "gpu", "Timer type: gpu or cpu");
    args.add_option("--init", "random", "Initialization: random, linear, constant");

    args.add_flag("--list", "List all kernel sets");

    if(!args.parse(argc, argv))
        return 0;

    // Parse arguments
    int N              = args.get_int("-n", 1);
    int C              = args.get_int("-c", 256);
    int K              = args.get_int("-k", 256);
    int Hi             = args.get_int("--size", 56);
    int Wi             = Hi;
    int Y              = args.get_int("-y", 3);
    int X              = args.get_int("-x", 3);
    int stride         = args.get_int("--stride", 1);
    int pad            = args.get_int("--pad", 1);
    int warmup         = args.get_int("--warmup", 5);
    int iterations     = args.get_int("--iterations", 100);
    bool flush_cache   = args.has("--flush-cache");
    int rotating_count = args.get_int("--rotating-count", 1);
    std::string timer  = args.get("--timer", "gpu");
    std::string init   = args.get("--init", "random");
    bool use_gpu_timer = (timer == "gpu");

    std::cout << "======================================================================\n";
    std::cout << "Example 11: Advanced Conv Benchmarking\n";
    std::cout << "======================================================================\n\n";

    if(args.has("--list"))
    {
        std::cout << "Declared Kernel Sets:\n";
        ConvKernelSetRegistry::instance().print();
        return 0;
    }

    // -------------------------------------------------------------------------
    // Step 1: Configuration Summary
    // -------------------------------------------------------------------------
    std::cout << "Step 1: Configuration Summary\n";
    std::cout << "-----------------------------\n";

    // Calculate output size
    int Ho = (Hi + 2 * pad - Y) / stride + 1;
    int Wo = (Wi + 2 * pad - X) / stride + 1;

    std::cout << "Problem:\n";
    std::cout << "  Batch:     N=" << N << "\n";
    std::cout << "  Channels:  C=" << C << ", K=" << K << "\n";
    std::cout << "  Input:     " << Hi << "x" << Wi << "\n";
    std::cout << "  Filter:    " << Y << "x" << X << "\n";
    std::cout << "  Output:    " << Ho << "x" << Wo << "\n";
    std::cout << "  Stride:    " << stride << ", Pad: " << pad << "\n";

    std::cout << "\nBenchmark Parameters:\n";
    std::cout << "  Warmup:         " << warmup << " iterations\n";
    std::cout << "  Benchmark:      " << iterations << " iterations\n";
    std::cout << "  Flush Cache:    " << (flush_cache ? "YES" : "NO") << "\n";
    std::cout << "  Rotating Count: " << rotating_count << "\n";
    std::cout << "  Timer:          " << timer << "\n";
    std::cout << "  Initialization: " << init << "\n\n";

    // -------------------------------------------------------------------------
    // Step 2: Show declared kernels
    // -------------------------------------------------------------------------
    std::cout << "Step 2: Declared Kernels\n";
    std::cout << "------------------------\n";

    const auto& kernel_set = ConvKernelSetRegistry::instance().get("benchmark_kernels");
    kernel_set.print(std::cout);
    std::cout << "\n";

#ifdef CONV_KERNEL_AVAILABLE
    // -------------------------------------------------------------------------
    // Step 3: Allocate and Initialize
    // -------------------------------------------------------------------------
    std::cout << "Step 3: Allocate and Initialize\n";
    std::cout << "--------------------------------\n";

    ck_tile::conv::ConvParam conv_param{
        2,
        1,
        static_cast<ck_tile::index_t>(N),
        static_cast<ck_tile::index_t>(K),
        static_cast<ck_tile::index_t>(C),
        {static_cast<ck_tile::index_t>(Y), static_cast<ck_tile::index_t>(X)},
        {static_cast<ck_tile::index_t>(Hi), static_cast<ck_tile::index_t>(Wi)},
        {stride, stride},
        {1, 1},
        {pad, pad},
        {pad, pad}};

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

    // Initialize based on method
    if(init == "random")
    {
        fill_random(input);
        fill_random(weight);
    }
    else if(init == "linear")
    {
        fill_linear(input);
        fill_linear(weight);
    }
    else
    { // constant
        fill_constant(input, 1.0f);
        fill_constant(weight, 1.0f);
    }

    std::cout << "  Input:  " << input.mDesc << " (" << init << ")\n";
    std::cout << "  Weight: " << weight.mDesc << " (" << init << ")\n";
    std::cout << "  Output: " << output.mDesc << "\n";

    // Calculate memory sizes
    size_t input_bytes  = input.get_element_space_size_in_bytes();
    size_t weight_bytes = weight.get_element_space_size_in_bytes();
    size_t output_bytes = output.get_element_space_size_in_bytes();
    size_t total_bytes  = input_bytes + weight_bytes + output_bytes;

    std::cout << "  Memory: " << std::fixed << std::setprecision(2)
              << (total_bytes / 1024.0 / 1024.0) << " MB\n\n";

    // Allocate GPU buffers
    ck_tile::DeviceMem input_dev(input_bytes);
    ck_tile::DeviceMem weight_dev(weight_bytes);
    ck_tile::DeviceMem output_dev(output_bytes);

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());

    // Create kernel args
    ck_tile::GroupedConvFwdHostArgs<> conv_args(conv_param,
                                                input_dev.GetDeviceBuffer(),
                                                weight_dev.GetDeviceBuffer(),
                                                {},
                                                output_dev.GetDeviceBuffer(),
                                                1);

    // -------------------------------------------------------------------------
    // Step 4: Warmup
    // -------------------------------------------------------------------------
    std::cout << "Step 4: Warmup (" << warmup << " iterations)\n";
    std::cout << "-------------------------------------------\n";

    ck_tile::stream_config stream_cfg{nullptr,
                                      true,
                                      0,
                                      warmup,
                                      1,
                                      use_gpu_timer,
                                      false, // no cache flush during warmup
                                      1};

    float warmup_time = SelectedConvKernelLauncher::launch(conv_args, stream_cfg);
    std::cout << "  Warmup complete. Last iteration: " << std::fixed << std::setprecision(4)
              << warmup_time << " ms\n\n";

    // -------------------------------------------------------------------------
    // Step 5: Benchmark
    // -------------------------------------------------------------------------
    std::cout << "Step 5: Benchmark (" << iterations << " iterations)\n";
    std::cout << "---------------------------------------------------\n";

    std::vector<float> times;
    times.reserve(iterations);

    // Configure stream for benchmark
    ck_tile::stream_config bench_cfg{nullptr,
                                     true,
                                     0,
                                     0, // no warmup
                                     1, // single iteration per call
                                     use_gpu_timer,
                                     flush_cache,
                                     rotating_count};

    for(int i = 0; i < iterations; ++i)
    {
        output_dev.SetZero();
        float time_ms = SelectedConvKernelLauncher::launch(conv_args, bench_cfg);
        times.push_back(time_ms);
    }

    // -------------------------------------------------------------------------
    // Step 6: Statistics
    // -------------------------------------------------------------------------
    std::cout << "\nStep 6: Statistics\n";
    std::cout << "------------------\n";

    std::sort(times.begin(), times.end());

    float min_time    = times.front();
    float max_time    = times.back();
    float median_time = times[times.size() / 2];
    float mean_time   = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();

    // Trimmed mean (remove 10% outliers from each end)
    size_t trim = times.size() / 10;
    float trimmed_mean =
        std::accumulate(times.begin() + trim, times.end() - trim, 0.0f) / (times.size() - 2 * trim);

    // Standard deviation
    float variance = 0.0f;
    for(float t : times)
    {
        variance += (t - mean_time) * (t - mean_time);
    }
    variance /= times.size();
    float std_dev = std::sqrt(variance);

    // Calculate TFLOPS
    double flops         = 2.0 * N * K * C * Ho * Wo * Y * X;
    double min_tflops    = (flops / (min_time / 1000.0)) / 1e12;
    double mean_tflops   = (flops / (mean_time / 1000.0)) / 1e12;
    double median_tflops = (flops / (median_time / 1000.0)) / 1e12;

    // Calculate bandwidth (GB/s)
    double bandwidth_min = (total_bytes / (min_time / 1000.0)) / 1e9;

    std::cout << "\n======================================================================\n";
    std::cout << "BENCHMARK RESULTS (" << iterations << " iterations)\n";
    std::cout << "======================================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Min time:      " << min_time << " ms (" << std::setprecision(2) << min_tflops
              << " TFLOPS)\n";
    std::cout << std::setprecision(4);
    std::cout << "  Max time:      " << max_time << " ms\n";
    std::cout << "  Mean time:     " << mean_time << " ms (" << std::setprecision(2) << mean_tflops
              << " TFLOPS)\n";
    std::cout << std::setprecision(4);
    std::cout << "  Median time:   " << median_time << " ms (" << std::setprecision(2)
              << median_tflops << " TFLOPS)\n";
    std::cout << std::setprecision(4);
    std::cout << "  Trimmed mean:  " << trimmed_mean << " ms\n";
    std::cout << "  Std deviation: " << std_dev << " ms\n";
    std::cout << "  Bandwidth:     " << std::setprecision(2) << bandwidth_min << " GB/s (peak)\n";
    std::cout << "======================================================================\n";

    // -------------------------------------------------------------------------
    // Step 7: Parameter Reference
    // -------------------------------------------------------------------------
    std::cout << "\nBENCHMARK PARAMETERS REFERENCE\n";
    std::cout << "==============================\n\n";
    std::cout << "  --warmup N          Warmup iterations (discard results)\n";
    std::cout << "                      Higher = more stable, longer run time\n\n";
    std::cout << "  --iterations N      Benchmark iterations\n";
    std::cout << "                      Higher = more accurate average\n\n";
    std::cout << "  --flush-cache       Flush GPU L2 cache between iterations\n";
    std::cout << "                      Use for memory-bound workloads\n\n";
    std::cout << "  --rotating-count N  Rotating buffer count\n";
    std::cout << "                      Simulates real workload cache behavior\n";
    std::cout << "                      Works with --flush-cache\n\n";
    std::cout << "  --timer TYPE        gpu: HIP events (accurate kernel time)\n";
    std::cout << "                      cpu: std::chrono (includes launch overhead)\n\n";
    std::cout << "  --init METHOD       random: uniform [-0.5, 0.5]\n";
    std::cout << "                      linear: sequential values\n";
    std::cout << "                      constant: all ones\n";

#else
    std::cout << "  [Kernel not compiled]\n";
    std::cout << "  Rebuild with generated kernels to enable GPU execution.\n";
#endif

    return 0;
}
