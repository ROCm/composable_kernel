// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 09: ML-Based Kernel Selection (Native C++)
 *
 * Uses a trained LightGBM model loaded via the C API to predict TFLOPS
 * for each kernel in the registry and select the best one. The kernels
 * are JIT-compiled at build time via DECL_KERNEL_SET (same as other examples).
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_09_ml_heuristic
 * Run:   ./gemm_09_ml_heuristic --model <path_to_model.lgbm>
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"
#include "ck_tile/dispatcher/ml_heuristic.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// Multiple kernel configs for ML to choose from
DECL_KERNEL_SET(ml_kernels,
                // Small tiles
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(64, 64, 32)
                         .wave(2, 2, 1)
                         .warp(16, 16, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(64, 64, 64)
                             .wave(2, 2, 1)
                             .warp(16, 16, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942")
                    // Medium tiles
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 32)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 64)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 64)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv4")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942")
                    // Large tiles
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(256, 256, 32)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(256, 128, 32)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942")
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 256, 32)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx942"));

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 09: ML-Based Kernel Selection",
                     "Uses trained LightGBM model for kernel selection");
    args.add_option("--arch", "gfx942", "GPU architecture");
    args.add_option("--model", "", "Path to LightGBM model file (.lgbm)");
    args.add_option("--log_transform", "false", "Model uses log1p transform");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 09: ML-Based Kernel Selection");

    std::string gfx_arch   = args.get("--arch", "gfx942");
    std::string model_path = args.get("--model", "");
    bool log_transform     = (args.get("--log_transform", "false") == "true");

    if(model_path.empty())
    {
        std::cerr << "Error: --model <path> is required" << std::endl;
        std::cerr << "Usage: ./gemm_09_ml_heuristic --model path/to/model_tflops.lgbm" << std::endl;
        return 1;
    }

    // Setup Registry (kernels are JIT compiled from DECL_KERNEL_SET above)
    Registry registry;
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);
    std::cout << "Registry: " << registry.size() << " kernel(s)" << std::endl;

    // Load ML model and create heuristic
    HardwareProfile hw;
    MLHeuristic ml_heuristic(model_path, &registry, hw, log_transform);
    if(!ml_heuristic.is_loaded())
    {
        std::cerr << "Failed to load model. Exiting." << std::endl;
        return 1;
    }

    // Wire ML heuristic into dispatcher
    Dispatcher dispatcher(&registry);
    dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
    dispatcher.set_heuristic([&ml_heuristic](const Problem& p) { return ml_heuristic(p); });

    std::cout << "Strategy: ML Heuristic (LightGBM)" << std::endl;

    // Test with different problem sizes
    using DataType                               = ck_tile::fp16_t;
    std::vector<std::tuple<int, int, int>> sizes = {
        {128, 128, 64},
        {512, 512, 256},
        {1024, 1024, 512},
        {2048, 2048, 1024},
    };

    std::cout << std::endl
              << std::setw(20) << "Shape" << std::setw(30) << "Selected Kernel" << std::setw(15)
              << "Pred TFLOPS" << std::setw(12) << "Select ms" << std::setw(10) << "Status"
              << std::endl;
    std::cout << std::string(87, '-') << std::endl;

    bool all_passed = true;

    for(const auto& [M, N, K] : sizes)
    {
        Problem problem;
        problem.M       = M;
        problem.N       = N;
        problem.K       = K;
        problem.k_batch = 1;

        auto t0          = std::chrono::high_resolution_clock::now();
        auto kernel      = dispatcher.select_kernel(problem);
        auto t1          = std::chrono::high_resolution_clock::now();
        double select_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::string size_str =
            std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);

        if(!kernel)
        {
            std::cout << std::setw(20) << size_str << std::setw(30) << "NONE" << std::setw(15)
                      << "N/A" << std::setw(12) << std::fixed << std::setprecision(2) << select_ms
                      << std::setw(10) << "FAIL" << std::endl;
            all_passed = false;
            continue;
        }

        double pred      = ml_heuristic.predict_tflops(problem, kernel->get_key());
        std::string name = kernel->get_key().encode_identifier();
        if(name.length() > 27)
            name = name.substr(0, 27) + "..";

        std::cout << std::setw(20) << size_str << std::setw(30) << name << std::setw(15)
                  << std::fixed << std::setprecision(2) << pred << std::setw(12)
                  << std::setprecision(2) << select_ms << std::setw(10) << "OK" << std::endl;
    }

    std::cout << std::endl
              << (all_passed ? "*** ALL TESTS PASSED ***" : "*** SOME TESTS FAILED ***")
              << std::endl;

    return all_passed ? 0 : 1;
}
