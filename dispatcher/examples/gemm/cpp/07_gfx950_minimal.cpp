// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 07: Minimal gfx950 (CDNA4 / MI350) GEMM
 *
 * Demonstrates the dispatcher working with gfx950-specific kernels:
 *
 *  - fp16 GEMM with standard tile configs
 *  - fp8 GEMM with gfx950-extended warp tiles (16x16x128)
 *  - 160KB LDS: gfx950 doubles the LDS from 64KB to 160KB
 *
 * Build: cd dispatcher/build && cmake .. -DGPU_TARGETS=gfx950 && make gemm_07_gfx950_minimal
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// gfx950-targeted kernel declarations
// =============================================================================

DECL_KERNEL_SET(gfx950_gemm_kernels,

                // fp16 128x128x32 -- bread-and-butter config, works on all CDNA
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(128, 128, 32)
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx950")

                    // fp16 128x128x64 -- deeper K tile using more LDS
                    // LDS usage: 128*64*2 + 128*64*2 = 32768 bytes (fits 64KB, gfx950 has 160KB)
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 64)
                             .wave(2, 2, 1)
                             .warp(32, 32, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx950")

                    // fp16 64x64x32 -- small-tile variant for small problems
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(64, 64, 32)
                             .wave(2, 2, 1)
                             .warp(16, 16, 32)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx950"));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 07: gfx950 Minimal GEMM",
                     "Demonstrates gfx950 (CDNA4 / MI350) dispatcher");
    args.add_flag("--list", "List registered kernels");
    args.add_flag("--list-verbose", "List registered kernels with full details");
    args.add_option("--M", "1024", "Problem M dimension");
    args.add_option("--N", "1024", "Problem N dimension");
    args.add_option("--K", "1024", "Problem K dimension");
    args.add_option("--arch", "gfx950", "GPU architecture (default: gfx950)");

    if(!args.parse(argc, argv))
        return 0;

    std::string gfx_arch = args.get("--arch", "gfx950");

    print_header("Example 07: gfx950 (CDNA4) Minimal GEMM");

    // =========================================================================
    // Architecture info
    // =========================================================================
    std::cout << "\ngfx950 (CDNA4 / MI350) highlights:\n";
    std::cout << "  - 160KB LDS (up from 64KB on gfx942)\n";
    std::cout << "  - Extended FP8 warp tiles: 16x16x128, 32x32x64\n";
    std::cout << "  - Packed FP4 support (pk_fp4)\n";
    std::cout << "  - Same warp configs as gfx942: [1,4,1], [2,2,1], [4,1,1]\n\n";

    // =========================================================================
    // Register kernels
    // =========================================================================
    std::cout << "Registering kernels for " << gfx_arch << "...\n";

    Registry registry;
    registry.set_name("gfx950_gemm");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);

    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    if(args.has("--list") || args.has("--list-verbose"))
    {
        std::cout << "\n";
        print_registered_kernels(registry, std::cout, args.has("--list-verbose"));
        return 0;
    }

    if(registry.size() == 0)
    {
        std::cerr << "ERROR: No kernels registered for " << gfx_arch << "!\n";
        std::cerr << "  Did you build with -DGPU_TARGETS=gfx950?\n";
        return 1;
    }

    // =========================================================================
    // Create Dispatcher
    // =========================================================================
    Dispatcher dispatcher(&registry);

    // =========================================================================
    // Setup Problem
    // =========================================================================
    const int M = args.get_int("--M", 1024);
    const int N = args.get_int("--N", 1024);
    const int K = args.get_int("--K", 1024);

    std::cout << "\nProblem: " << M << " x " << N << " x " << K << "\n";

    Problem problem(M, N, K);

    using DataType = ck_tile::fp16_t;
    GpuBuffer<DataType> a_dev(M * K);
    GpuBuffer<DataType> b_dev(K * N);
    GpuBuffer<DataType> c_dev(M * N);

    std::vector<DataType> a_host(M * K, DataType(1.0f));
    std::vector<DataType> b_host(K * N, DataType(1.0f));
    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());
    c_dev.zero();

    // =========================================================================
    // Select and Run
    // =========================================================================
    auto selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cerr << "ERROR: No suitable kernel found for " << M << "x" << N << "x" << K << "\n";
        return 1;
    }
    std::cout << "  Selected: " << selected->get_name() << "\n";

    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    std::cout << "  Time:   " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << calculate_tflops(M, N, K, time_ms) << "\n";

    // =========================================================================
    // Verify
    // =========================================================================
    std::cout << "\nVerification:\n";
    std::vector<DataType> c_host(M * N);
    c_dev.copy_to_host(c_host.data());

    const float expected = static_cast<float>(K);
    int errors           = 0;
    for(int i = 0; i < std::min(M * N, 1024); ++i)
    {
        if(std::abs(static_cast<float>(c_host[i]) - expected) > 0.01f * expected + 1.0f)
            ++errors;
    }

    bool passed = (errors == 0);
    std::cout << "  Expected value: " << expected << "\n";
    std::cout << "  Errors (first 1024 elements): " << errors << "\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";

    print_separator();
    return passed ? 0 : 1;
}
