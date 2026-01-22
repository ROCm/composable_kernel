// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 01: Basic GEMM - Autofill, Autocorrect, and Full Declaration
 *
 * Demonstrates THREE declaration patterns:
 *
 * 1. AUTOFILL: Minimal declaration - missing params filled with defaults
 *    .add(Signature().dtype("fp16").layout("rcr"),
 *         Algorithm().tile(128,128,64).pipeline("compv3").scheduler("intrawave"),
 *         "gfx942")
 *    -> wave(2,2,1), warp(32,32,16), epilogue("cshuffle") added automatically
 *
 * 2. AUTOCORRECT: Invalid params corrected to valid values
 *    .add(..., Algorithm().wave(1,1,1)...)
 *    -> wave(1,1,1) is invalid for gfx942, corrected to wave(2,2,1)
 *
 * 3. FULL: All parameters explicitly specified
 *    .add(..., Algorithm().tile().wave().warp().pipeline().scheduler().epilogue()...)
 *
 * Build: cd dispatcher/build && cmake .. && make gemm_01_basic
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
// THREE KERNEL DECLARATION PATTERNS
// =============================================================================

DECL_KERNEL_SET(
    basic_gemm_kernels,
    // -------------------------------------------------------------------------
    // Pattern 1: AUTOFILL - Minimal declaration
    // Only specify: dtype, layout, tile, pipeline, scheduler
    // Auto-filled: wave(2,2,1), warp(32,32,16), epilogue("cshuffle"), pad(false,false,false)
    // -------------------------------------------------------------------------
    .add(Signature().dtype("fp16").layout("rcr"),
         Algorithm()
             .tile(128, 128, 64)      // Required
             .pipeline("compv3")      // Required
             .scheduler("intrawave"), // Required
         "gfx942")

        // -------------------------------------------------------------------------
        // Pattern 2: AUTOCORRECT - Invalid wave config
        // wave(1,1,1) is invalid for gfx942 WMMA, corrected to wave(2,2,1)
        // -------------------------------------------------------------------------
        .add(Signature().dtype("fp16").layout("rcr"),
             Algorithm()
                 .tile(128, 128, 32) // Different tile_k to make unique kernel
                 .wave(1, 1, 1)      // INVALID: autocorrected to (2,2,1)
                 .warp(32, 32, 16)   // Valid warp for 128x128 tile
                 .pipeline("compv3")
                 .scheduler("intrawave")
                 .epilogue("cshuffle"),
             "gfx942")

        // -------------------------------------------------------------------------
        // Pattern 3: FULL - All parameters explicitly specified
        // No autofill or autocorrect needed
        // -------------------------------------------------------------------------
        .add(Signature().dtype("fp16").layout("rcr"),
             Algorithm()
                 .tile(64, 64, 32)          // Explicit tile
                 .wave(2, 2, 1)             // Explicit wave (valid)
                 .warp(16, 16, 32)          // Explicit warp tile
                 .pipeline("compv3")        // Explicit pipeline
                 .scheduler("intrawave")    // Explicit scheduler
                 .epilogue("cshuffle")      // Explicit epilogue
                 .pad(false, false, false), // Explicit padding
             "gfx942"));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 01: GEMM Autofill/Autocorrect/Full",
                     "Three kernel declaration patterns");
    args.add_flag("--list", "List registered kernels");
    args.add_flag("--list-verbose", "List registered kernels with full details");
    args.add_option("--size", "1024", "Problem size MxNxK");
    args.add_option("--arch", "gfx942", "GPU architecture");

    if(!args.parse(argc, argv))
        return 0;

    print_header("Example 01: GEMM Declaration Patterns");

    // =========================================================================
    // Show the Three Patterns
    // =========================================================================
    std::cout << "\nTHREE DECLARATION PATTERNS:\n";
    std::cout << "============================\n\n";

    std::cout << "1. AUTOFILL (minimal declaration):\n";
    std::cout << "   .add(Signature().dtype(\"fp16\").layout(\"rcr\"),\n";
    std::cout
        << "        Algorithm().tile(128,128,64).pipeline(\"compv3\").scheduler(\"intrawave\"),\n";
    std::cout << "        \"gfx942\")\n";
    std::cout << "   -> Auto-filled: wave(2,2,1), warp(32,32,16), epilogue(\"cshuffle\")\n\n";

    std::cout << "2. AUTOCORRECT (invalid params fixed):\n";
    std::cout << "   .add(..., Algorithm().wave(1,1,1)...)\n";
    std::cout << "   -> wave(1,1,1) invalid for gfx942, corrected to wave(2,2,1)\n\n";

    std::cout << "3. FULL (all params explicit):\n";
    std::cout << "   .add(..., "
                 "Algorithm().tile().wave().warp().pipeline().scheduler().epilogue().pad()...)\n";
    std::cout << "   -> No changes needed\n\n";

    std::string gfx_arch = args.get("--arch", "gfx942");

    // =========================================================================
    // Step 1: Show Declared Kernel Sets
    // =========================================================================
    std::cout << "Step 1: Declared Kernel Sets\n";
    KernelSetRegistry::instance().print();

    const auto& decl_set = KernelSetRegistry::instance().get("basic_gemm_kernels");
    std::cout << "  'basic_gemm_kernels': " << decl_set.size() << " declaration(s)\n";

    // =========================================================================
    // Step 2: Create Registry and Register Kernels
    // =========================================================================
    std::cout << "\nStep 2: Register Kernels\n";

    Registry registry;
    // Use generic macro
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);

    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    // List kernels if requested
    if(args.has("--list") || args.has("--list-verbose"))
    {
        std::cout << "\n";
        print_registered_kernels(registry, std::cout, args.has("--list-verbose"));
        return 0;
    }

    // =========================================================================
    // Step 3: Create Dispatcher
    // =========================================================================
    std::cout << "\nStep 3: Create Dispatcher\n";
    Dispatcher dispatcher(&registry);

    // =========================================================================
    // Step 4: Setup Problem
    // =========================================================================
    int size    = args.get_int("--size", 1024);
    const int M = size, N = size, K = size;

    std::cout << "\nStep 4: Setup Problem (" << M << "x" << N << "x" << K << ")\n";

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
    // Step 5: Select and Run
    // =========================================================================
    std::cout << "\nStep 5: Select and Run\n";

    auto selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cerr << "ERROR: No kernel found!\n";
        return 1;
    }
    std::cout << "  Selected: " << selected->get_name() << "\n";

    float time_ms = dispatcher.run(a_dev.get(), b_dev.get(), c_dev.get(), problem, nullptr);
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::setprecision(2) << calculate_tflops(M, N, K, time_ms) << "\n";

    // =========================================================================
    // Step 6: Verify
    // =========================================================================
    std::cout << "\nStep 6: Verify\n";
    std::vector<DataType> c_host(M * N);
    c_dev.copy_to_host(c_host.data());

    const float expected = static_cast<float>(K);
    int errors           = 0;
    for(int i = 0; i < M * N; ++i)
    {
        if(std::abs(static_cast<float>(c_host[i]) - expected) > 0.01f * expected + 1.0f)
            ++errors;
    }

    bool passed = (errors == 0);
    std::cout << "  Expected: " << expected << ", Errors: " << errors << "\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";

    // =========================================================================
    // Summary
    // =========================================================================
    print_separator();
    std::cout << "DECLARATION PATTERNS SUMMARY:\n";
    print_separator();
    std::cout << R"(
  1. AUTOFILL: Specify only required params, system fills defaults
     - Useful for quick prototyping
     - Guarantees valid configuration

  2. AUTOCORRECT: System validates and fixes invalid params
     - wave(1,1,1) -> wave(2,2,1) on gfx942
     - Invalid pipeline/scheduler combos fixed
     - Logs corrections for debugging

  3. FULL: All params explicit - no changes made
     - Full control over configuration
     - Best for production/tuning
)";
    print_separator();

    return passed ? 0 : 1;
}
