// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 07: PreShuffle Pipeline
 *
 * Demonstrates the PreShuffle pipeline variant which improves performance
 * by pre-shuffling data in LDS before computation.
 *
 * Complexity: ★★★★☆
 *
 * PreShuffle Pipeline Overview:
 *   - PreShuffleV1: Basic pre-shuffling in LDS
 *   - PreShuffleV2: Enhanced version with better memory access patterns
 *
 * Benefits:
 *   - Reduces bank conflicts in shared memory
 *   - Better data reuse patterns
 *   - Typically faster than standard CompV4 on large matrices
 *
 * Requirements:
 *   - Must generate preshuffle kernels: --pipeline preshuffle
 *   - Larger LDS usage than standard pipelines
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;

// =============================================================================
// KERNEL CONFIGURATION - PreShuffle V1
// =============================================================================
// PreShuffle kernels have different optimal configurations due to
// their unique memory access patterns.

namespace preshuffle_config {

using ADataType   = ck_tile::fp16_t;
using BDataType   = ck_tile::fp16_t;
using CDataType   = ck_tile::fp16_t;
using AccDataType = float;

// PreShuffle works best with larger tiles
constexpr int TileM = 256;
constexpr int TileN = 256;
constexpr int TileK = 64;

constexpr int WavesM = 4;
constexpr int WavesN = 4;
constexpr int WavesK = 1;

constexpr int WarpM = 32;
constexpr int WarpN = 32;
constexpr int WarpK = 16;

constexpr int BlockSize = 256;

} // namespace preshuffle_config

// =============================================================================
// Helper: Configure PreShuffle kernel
// =============================================================================

KernelKey make_preshuffle_key(Pipeline version)
{
    using namespace preshuffle_config;

    KernelKeyBuilder builder;

    // Data types
    builder.dtype_a   = DataType::FP16;
    builder.dtype_b   = DataType::FP16;
    builder.dtype_c   = DataType::FP16;
    builder.dtype_acc = DataType::FP32;

    // Layouts (Row-Col-Row)
    builder.layout_a = LayoutTag::RowMajor;
    builder.layout_b = LayoutTag::ColMajor;
    builder.layout_c = LayoutTag::RowMajor;

    // Tile configuration
    builder.tile_m = TileM;
    builder.tile_n = TileN;
    builder.tile_k = TileK;

    builder.wave_m = WavesM;
    builder.wave_n = WavesN;
    builder.wave_k = WavesK;

    builder.warp_m = WarpM;
    builder.warp_n = WarpN;
    builder.warp_k = WarpK;

    builder.block_size = BlockSize;

    // PreShuffle-specific settings
    builder.pipeline   = version;
    builder.preshuffle = true;
    builder.scheduler  = Scheduler::Intrawave;

    return builder.build();
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv)
{
    print_header("Example 07: PreShuffle Pipeline");

    using namespace preshuffle_config;

    // Parse problem size
    int M = 2048, N = 2048, K = 2048;
    if(argc >= 4)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    std::cout << "Problem: " << format_size(M, N, K) << "\n\n";

    // -------------------------------------------------------------------------
    // Demonstrate PreShuffle configuration
    // -------------------------------------------------------------------------
    std::cout << "PreShuffle Configuration:\n";
    std::cout << "  Tile: " << TileM << "x" << TileN << "x" << TileK << "\n";
    std::cout << "  Waves: " << WavesM << "x" << WavesN << "x" << WavesK << "\n";
    std::cout << "  Note: PreShuffle requires larger tiles for best performance\n\n";

    // -------------------------------------------------------------------------
    // Compare pipelines (conceptually)
    // -------------------------------------------------------------------------
    std::cout << "Pipeline Comparison:\n";
    print_separator('-', 60);

    struct PipelineInfo
    {
        const char* name;
        Pipeline pipeline;
        const char* description;
    };

    std::vector<PipelineInfo> pipelines = {
        {"CompV4 (baseline)", Pipeline::CompV4, "Standard compute pipeline"},
        {"PreShuffleV1", Pipeline::PreShuffleV1, "Pre-shuffle in LDS (basic)"},
        {"PreShuffleV2", Pipeline::PreShuffleV2, "Pre-shuffle in LDS (optimized)"},
    };

    for(const auto& info : pipelines)
    {
        std::cout << "  " << info.name << ":\n";
        std::cout << "    " << info.description << "\n";

        // Show key configuration
        KernelKeyBuilder builder;
        builder.pipeline = info.pipeline;
        builder.preshuffle =
            (info.pipeline == Pipeline::PreShuffleV1 || info.pipeline == Pipeline::PreShuffleV2);

        std::cout << "    preshuffle=" << (builder.preshuffle ? "true" : "false") << "\n\n";
    }

    // -------------------------------------------------------------------------
    // Build PreShuffle kernel key
    // -------------------------------------------------------------------------
    std::cout << "Building PreShuffle V2 kernel key...\n\n";

    KernelKey key = make_preshuffle_key(Pipeline::PreShuffleV2);

    std::cout << "Key configuration:\n";
    std::cout << "  pipeline: PreShuffleV2\n";
    std::cout << "  preshuffle: true\n";
    std::cout << "  tile: " << static_cast<int>(key.algorithm.tile_shape.m) << "x"
              << static_cast<int>(key.algorithm.tile_shape.n) << "x"
              << static_cast<int>(key.algorithm.tile_shape.k) << "\n\n";

    // -------------------------------------------------------------------------
    // Note about kernel generation
    // -------------------------------------------------------------------------
    print_separator('-', 60);
    std::cout << "To generate PreShuffle kernels:\n\n";
    std::cout << "  cd dispatcher/codegen\n";
    std::cout << "  python3 unified_gemm_codegen.py \\\n";
    std::cout << "    --pipeline preshuffle \\\n";
    std::cout << "    --tile 256x256x64 \\\n";
    std::cout << "    --output-dir ../build/generated_kernels\n\n";

    std::cout << "Then update CMakeLists.txt to include the preshuffle kernel header.\n\n";
    print_separator('-', 60);

    // -------------------------------------------------------------------------
    // Fallback: Run with standard kernel if available
    // -------------------------------------------------------------------------
    std::cout << "\nRunning with current kernel (CompV4 fallback)...\n";

    // Use the currently loaded kernel (from -include)
    KernelKeyBuilder fallback = KernelKeyBuilder::fp16_rcr();
    fallback.tile_m           = SelectedKernel::TileM;
    fallback.tile_n           = SelectedKernel::TileN;
    fallback.tile_k           = SelectedKernel::TileK;
    fallback.wave_m           = SelectedKernel::WarpPerBlock_M;
    fallback.wave_n           = SelectedKernel::WarpPerBlock_N;
    fallback.wave_k           = SelectedKernel::WarpPerBlock_K;
    fallback.warp_m           = SelectedKernel::WarpTileM;
    fallback.warp_n           = SelectedKernel::WarpTileN;
    fallback.warp_k           = SelectedKernel::WarpTileK;
    fallback.block_size       = SelectedKernel::BlockSize;

    KernelKey fallback_key = fallback.build();

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            fallback_key, "fp16_rcr_fallback");

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel);

    // Run
    Problem problem(M, N, K);
    GpuBuffer<ADataType> a_dev(M * K);
    GpuBuffer<BDataType> b_dev(K * N);
    GpuBuffer<CDataType> c_dev(M * N);

    std::vector<ADataType> a_host(M * K, ADataType(0.1f));
    std::vector<BDataType> b_host(K * N, BDataType(0.1f));

    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());
    c_dev.zero();

    Dispatcher dispatcher;
    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);

    double tflops = calculate_tflops(M, N, K, time_ms);

    std::cout << "\nResults:\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << tflops << "\n\n";

    print_separator();
    std::cout << "PreShuffle example complete!\n";
    std::cout << "(Note: Actual preshuffle kernel requires separate generation)\n";
    print_separator();

    return 0;
}
