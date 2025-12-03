// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 04: Advanced Convolution Benchmark
 *
 * Demonstrates all available benchmark parameters matching CK Tile stream_config:
 *   - warmup: Number of warmup iterations (default: 5)
 *   - repeat: Number of benchmark iterations (default: 20)
 *   - flush_cache: Flush GPU L2 cache between iterations (default: false)
 *   - rotating_count: Number of rotating buffers for cache simulation (default: 1)
 *   - timer: Timer type - GPU events (default) or CPU chrono
 *
 * Build:
 *   cd dispatcher/build && cmake .. -DBUILD_DISPATCHER_EXAMPLES=ON && make conv_04_benchmark
 *
 * Usage:
 *   ./conv_04_benchmark
 *   ./conv_04_benchmark --help
 *   ./conv_04_benchmark --warmup 10 --repeat 100
 *   ./conv_04_benchmark --flush-cache --rotating-count 4
 *
 * Complexity: ★★★☆☆
 */

#include <iostream>
#include <iomanip>
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
// KERNEL DECLARATIONS - Benchmark configurations
// =============================================================================

DECL_CONV_KERNEL_SET(conv_benchmark,
                     .add(ConvSig().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                          ConvAlgo()
                              .tile(1, 128, 128)
                              .wave(2, 2, 1)
                              .warp(32, 32, 16)
                              .pipeline("compv4")
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
    // Parse command line arguments
    ExampleArgs args("Example 04: Advanced Convolution Benchmark",
                     "Demonstrates all benchmark parameters (like CK Tile stream_config)");

    // Problem size
    args.add_option("-n", "1", "Batch size N");
    args.add_option("-c", "128", "Input channels C");
    args.add_option("-k", "128", "Output channels K");
    args.add_option("-h", "28", "Input height/width H=W");
    args.add_option("-y", "3", "Filter size Y=X");

    // Benchmark parameters (matching CK Tile stream_config)
    args.add_option("--warmup", "5", "Warmup iterations (cold_niters_)");
    args.add_option("--repeat", "20", "Benchmark iterations (nrepeat_)");
    args.add_flag("--flush-cache", "Flush L2 cache between iterations (flush_cache_)");
    args.add_option("--rotating-count", "1", "Rotating buffer count (rotating_count_)");
    args.add_flag("--cpu-timer", "Use CPU timer instead of GPU events");

    if(!args.parse(argc, argv))
    {
        return 0; // --help was printed
    }

    // Parse values
    int N  = args.get_int("-n", 1);
    int C  = args.get_int("-c", 128);
    int K  = args.get_int("-k", 128);
    int Hi = args.get_int("-h", 28);
    int Wi = Hi;
    int Y  = args.get_int("-y", 3);
    int X  = Y;

    int warmup         = args.get_int("--warmup", 5);
    int repeat         = args.get_int("--repeat", 20);
    bool flush_cache   = args.has("--flush-cache");
    int rotating_count = args.get_int("--rotating-count", 1);
    bool use_gpu_timer = !args.has("--cpu-timer");

    std::cout << "======================================================================\n";
    std::cout << "Example 04: Advanced Convolution Benchmark\n";
    std::cout << "======================================================================\n\n";

    // -------------------------------------------------------------------------
    // Show configuration
    // -------------------------------------------------------------------------
    std::cout << "Benchmark Configuration:\n";
    std::cout << "  Problem:        N=" << N << ", C=" << C << ", K=" << K << ", " << Hi << "x"
              << Wi << ", " << Y << "x" << X << "\n";
    std::cout << "  Warmup:         " << warmup << " iterations\n";
    std::cout << "  Repeat:         " << repeat << " iterations\n";
    std::cout << "  Flush Cache:    " << (flush_cache ? "Yes" : "No") << "\n";
    std::cout << "  Rotating Count: " << rotating_count << "\n";
    std::cout << "  Timer:          " << (use_gpu_timer ? "GPU" : "CPU") << "\n\n";

    // -------------------------------------------------------------------------
    // Create CK Tile conv param
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Allocate tensors
    // -------------------------------------------------------------------------
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

    std::cout << "Tensors:\n";
    std::cout << "  Input:  " << input.mDesc << "\n";
    std::cout << "  Weight: " << weight.mDesc << "\n";
    std::cout << "  Output: " << output.mDesc << "\n\n";

    // -------------------------------------------------------------------------
    // Transfer to GPU
    // -------------------------------------------------------------------------
    ck_tile::DeviceMem input_dev(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev(output.get_element_space_size_in_bytes());

    input_dev.ToDevice(input.data());
    weight_dev.ToDevice(weight.data());
    output_dev.SetZero();

#ifdef CONV_KERNEL_AVAILABLE
    // -------------------------------------------------------------------------
    // Create kernel args and stream config
    // -------------------------------------------------------------------------
    ck_tile::GroupedConvFwdHostArgs<> kernel_args(conv_param,
                                                  input_dev.GetDeviceBuffer(),
                                                  weight_dev.GetDeviceBuffer(),
                                                  {},
                                                  output_dev.GetDeviceBuffer(),
                                                  1 // k_batch
    );

    // Create stream_config with all benchmark parameters
    // struct stream_config {
    //     hipStream_t stream_id_ = nullptr;
    //     bool time_kernel_      = false;
    //     int log_level_         = 0;
    //     int cold_niters_       = 3;   // warmup
    //     int nrepeat_           = 10;  // benchmark iterations
    //     bool is_gpu_timer_     = true;
    //     bool flush_cache_      = false;
    //     int rotating_count_    = 1;
    // };
    ck_tile::stream_config stream_cfg{
        nullptr,       // stream_id
        true,          // time_kernel
        1,             // log_level
        warmup,        // cold_niters (warmup)
        repeat,        // nrepeat (benchmark iterations)
        use_gpu_timer, // is_gpu_timer
        flush_cache,   // flush_cache
        rotating_count // rotating_count
    };

    std::cout << "Running Benchmark...\n";
    std::cout << "----------------------------------------------------------------------\n";

    // Run benchmark
    float avg_time_ms = SelectedConvKernelLauncher::launch(kernel_args, stream_cfg);

    // Calculate metrics
    auto problem  = create_conv2d_problem(N, C, K, Hi, Wi, Y, X, 1, 1);
    double flops  = problem.get_flops();
    double tflops = flops / (avg_time_ms * 1e9);
    double bandwidth_gb =
        (input.get_element_space_size_in_bytes() + weight.get_element_space_size_in_bytes() +
         output.get_element_space_size_in_bytes()) /
        1e9 / (avg_time_ms / 1000);

    std::cout << "\n*** BENCHMARK RESULTS ***\n";
    std::cout << "  Average Time:   " << std::fixed << std::setprecision(4) << avg_time_ms
              << " ms\n";
    std::cout << "  TFLOPS:         " << std::fixed << std::setprecision(2) << tflops << "\n";
    std::cout << "  Bandwidth:      " << std::fixed << std::setprecision(2) << bandwidth_gb
              << " GB/s\n";
    std::cout << "  FLOPs:          " << std::scientific << std::setprecision(2) << flops << "\n";
#else
    std::cout << "  [Kernel not compiled - build with CMake or compile_conv_examples.py]\n";
    std::cout << "  To build:\n";
    std::cout << "    cd dispatcher/build && cmake .. -DBUILD_DISPATCHER_EXAMPLES=ON && make "
                 "conv_04_benchmark\n";
#endif

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    std::cout << "\n======================================================================\n";
    std::cout << "BENCHMARK PARAMETERS REFERENCE (CK Tile stream_config)\n";
    std::cout << "======================================================================\n";
    std::cout << R"(
ck_tile::stream_config cfg{
    nullptr,    // stream_id       - HIP stream (nullptr = default)
    true,       // time_kernel     - Enable timing
    1,          // log_level       - Verbosity (0=quiet, 1=normal, 2=verbose)
    5,          // cold_niters     - Warmup iterations (discarded)
    20,         // nrepeat         - Benchmark iterations (averaged)
    true,       // is_gpu_timer    - Use GPU events (true) or CPU chrono (false)
    false,      // flush_cache     - Flush L2 cache between iterations
    1           // rotating_count  - Rotating buffers for cache simulation
};

Parameter usage:
  --warmup N          Warmup iterations (cold_niters_)
  --repeat N          Benchmark iterations (nrepeat_)
  --flush-cache       Flush L2 cache (for memory-bound analysis)
  --rotating-count N  Rotating buffers (requires --flush-cache)
  --cpu-timer         Use CPU timer instead of GPU events
)";
    std::cout << "======================================================================\n";

    return 0;
}
