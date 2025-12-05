// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example 04: GEMM Validation
 *
 * Validates GEMM output against CK Tile reference implementations.
 *
 * Verification modes:
 *   --verify 0  : No verification (benchmark only)
 *   --verify 1  : CPU reference (slower, but always works)
 *   --verify 2  : GPU reference (faster for large matrices)
 *
 * Build:
 *   cd dispatcher/build && make gemm_04_validation
 *
 * Usage:
 *   ./gemm_04_validation
 *   ./gemm_04_validation --help
 *   ./gemm_04_validation --size 1024 --verify 2
 *   ./gemm_04_validation --size 256 --verify 1 --rtol 0.01
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::utils;
using namespace ck_tile::literals;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// KERNEL SET: Multiple kernels for validation testing
// =============================================================================

DECL_KERNEL_SET(validation_kernels,
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(128, 128, 32)
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx942"));

// =============================================================================
// Helper: Determine if layout is row-major
// =============================================================================

template <typename Layout>
constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    // Parse command line arguments
    ExampleArgs args("Example 04: GEMM Validation",
                     "Validates GPU output against CK Tile reference (CPU or GPU)");
    args.add_option("--size", "512", "Problem size MxNxK");
    args.add_option("--verify", "1", "Verification mode: 0=none, 1=CPU ref, 2=GPU ref");
    args.add_option("--rtol", "0.01", "Relative tolerance");
    args.add_option("--atol", "0.01", "Absolute tolerance");
    args.add_option("--arch", "gfx942", "GPU architecture");

    if(!args.parse(argc, argv))
    {
        return 0; // --help was printed
    }

    int M                = args.get_int("--size", 512);
    int N                = M;
    int K                = M;
    int verify           = args.get_int("--verify", 1);
    float rtol           = args.get_float("--rtol", 0.01f);
    float atol           = args.get_float("--atol", 0.01f);
    std::string gfx_arch = args.get("--arch", "gfx942");

    print_header("Example 04: GEMM Validation with CK Tile Reference");

    std::cout << "\nConfiguration:\n";
    std::cout << "  Problem:      " << M << " x " << N << " x " << K << "\n";
    std::cout << "  Layout:       RCR (A=row, B=col, C=row)\n";
    std::cout << "  Verify mode:  " << verify;
    if(verify == 0)
        std::cout << " (none)";
    else if(verify == 1)
        std::cout << " (CPU reference)";
    else if(verify == 2)
        std::cout << " (GPU reference - faster)";
    std::cout << "\n";
    std::cout << "  Tolerance:    rtol=" << rtol << ", atol=" << atol << "\n";

    // =========================================================================
    // Setup Registry and Dispatcher
    // =========================================================================
    Registry registry;
    generated::register_04_validation_kernels(registry, gfx_arch);
    Dispatcher dispatcher(&registry);

    std::cout << "  Kernels:      " << registry.size() << " registered\n";

    // =========================================================================
    // Initialize data using proper tensor descriptors for RCR layout
    // =========================================================================
    std::cout << "\nStep 1: Initialize Data\n";
    std::cout << "-----------------------\n";

    // Define layouts (RCR = Row-Col-Row)
    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using ADataType   = ck_tile::fp16_t;
    using BDataType   = ck_tile::fp16_t;
    using CDataType   = ck_tile::fp16_t;
    using AccDataType = float;

    // Get default strides for each layout
    auto stride_a = ck_tile::get_default_stride(M, K, 0_uz, is_row_major(ALayout{}));
    auto stride_b = ck_tile::get_default_stride(K, N, 0_uz, is_row_major(BLayout{}));
    auto stride_c = ck_tile::get_default_stride(M, N, 0_uz, is_row_major(CLayout{}));

    // Create HostTensors with proper layout descriptors
    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_a, is_row_major(ALayout{})));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_b, is_row_major(BLayout{})));
    ck_tile::HostTensor<CDataType> c_m_n_dev(
        ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(CLayout{})));
    ck_tile::HostTensor<CDataType> c_m_n_ref(
        ck_tile::host_tensor_descriptor(M, N, stride_c, is_row_major(CLayout{})));

    // Initialize with random values
    ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);

    std::cout << "  A: " << M << " x " << K << " (fp16, row-major, stride=" << stride_a << ")\n";
    std::cout << "  B: " << K << " x " << N << " (fp16, col-major, stride=" << stride_b << ")\n";
    std::cout << "  C: " << M << " x " << N << " (fp16, row-major, stride=" << stride_c << ")\n";

    // =========================================================================
    // Allocate GPU memory
    // =========================================================================
    ck_tile::DeviceMem a_dev(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev(c_m_n_dev.get_element_space_size_in_bytes());

    a_dev.ToDevice(a_m_k.data());
    b_dev.ToDevice(b_k_n.data());
    c_dev.SetZero();

    // =========================================================================
    // Compute Reference (if verify > 0)
    // =========================================================================
    if(verify > 0)
    {
        std::cout << "\nStep 2: Compute Reference\n";
        std::cout << "-------------------------\n";

        c_m_n_ref.SetZero();

        if(verify == 1)
        {
            std::cout << "  Using CPU reference (ck_tile::reference_gemm)...\n";

            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_m_k, b_k_n, c_m_n_ref);

            std::cout << "  CPU reference complete.\n";
        }
        else if(verify == 2)
        {
            std::cout << "  Using GPU reference (ck_tile::reference_gemm_gpu)...\n";

            // Create a separate buffer for GPU reference output
            ck_tile::DeviceMem c_ref_dev(c_m_n_ref.get_element_space_size_in_bytes());
            c_ref_dev.SetZero();

            ck_tile::reference_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout>(
                static_cast<ADataType*>(a_dev.GetDeviceBuffer()),
                static_cast<BDataType*>(b_dev.GetDeviceBuffer()),
                static_cast<CDataType*>(c_ref_dev.GetDeviceBuffer()),
                M,
                N,
                K,
                stride_a,
                stride_b,
                stride_c);

            // Sync and copy back
            (void)hipDeviceSynchronize();
            c_ref_dev.FromDevice(c_m_n_ref.data());

            std::cout << "  GPU reference complete.\n";
        }
    }

    // =========================================================================
    // Run GPU kernel
    // =========================================================================
    std::cout << "\nStep 3: Run GPU Kernel\n";
    std::cout << "----------------------\n";

    Problem problem(M, N, K);

    // Show selected kernel
    auto selected = dispatcher.select_kernel(problem);
    if(selected)
    {
        std::cout << "  Selected: " << selected->get_name() << "\n";
    }

    float time_ms = dispatcher.run(static_cast<ADataType*>(a_dev.GetDeviceBuffer()),
                                   static_cast<BDataType*>(b_dev.GetDeviceBuffer()),
                                   static_cast<CDataType*>(c_dev.GetDeviceBuffer()),
                                   problem,
                                   nullptr);

    // Copy result back
    c_dev.FromDevice(c_m_n_dev.data());

    // Calculate performance
    double flops  = 2.0 * M * N * K;
    double tflops = flops / (time_ms * 1e9);

    std::cout << "  Time:      " << std::fixed << std::setprecision(4) << time_ms << " ms\n";
    std::cout << "  TFLOPS:    " << std::fixed << std::setprecision(2) << tflops << "\n";

    // =========================================================================
    // Validate
    // =========================================================================
    bool pass = true;

    if(verify > 0)
    {
        std::cout << "\nStep 4: Validation\n";
        std::cout << "------------------\n";
        std::cout << "  Tolerance: rtol=" << rtol << ", atol=" << atol << "\n";

        // Use CK Tile's check_err for validation
        pass = ck_tile::check_err(c_m_n_dev, c_m_n_ref, "Validation Error!", rtol, atol);

        // Calculate max differences for reporting
        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        for(size_t i = 0; i < c_m_n_dev.get_element_space_size(); ++i)
        {
            float dev_val  = static_cast<float>(c_m_n_dev.mData[i]);
            float ref_val  = static_cast<float>(c_m_n_ref.mData[i]);
            float abs_diff = std::abs(dev_val - ref_val);
            float rel_diff = (ref_val != 0.0f) ? abs_diff / std::abs(ref_val) : abs_diff;
            max_abs_diff   = std::max(max_abs_diff, abs_diff);
            max_rel_diff   = std::max(max_rel_diff, rel_diff);
        }

        std::cout << "  Max abs diff: " << max_abs_diff << "\n";
        std::cout << "  Max rel diff: " << max_rel_diff << "\n";
    }

    // =========================================================================
    // Summary
    // =========================================================================
    print_separator();
    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << "\n";
    print_separator();

    if(verify == 0)
    {
        std::cout << "\nNote: Verification was disabled (--verify 0)\n";
        std::cout << "Use --verify 1 for CPU reference or --verify 2 for GPU reference.\n";
    }

    return pass ? 0 : 1;
}
