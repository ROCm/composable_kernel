// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 11k: XOR Test with LDS Transpose - Manual Addressing
 *
 * This uses RAW __shared__ memory with manual addressing to show bank conflicts.
 *
 * Pattern:
 * 1. Load from global [M, K]
 * 2. Store to LDS [M, K] row-major (sequential, no conflicts)
 * 3. Read from LDS in TRANSPOSED column-major pattern (stride-M = CONFLICTS!)
 * 4. Store to global [K, M] transposed
 *
 * For plain mode: Direct addressing
 * For XOR mode: Manual XOR transformation on addresses
 *
 * This should FINALLY show the bank conflict difference!
 */

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct LDSTransposeKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    // Manual XOR address calculation
    CK_TILE_DEVICE static index_t xor_address(index_t m, index_t k)
    {
        if constexpr (UseXor)
        {
            // Apply XOR swizzle: m' = m XOR (k / KPack)
            index_t m_xor = m ^ (k / kKPack);
            return m_xor * kK + k;
        }
        else
        {
            // Plain row-major addressing
            return m * kK + k;
        }
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                    DataType* __restrict__ output,
                                    index_t M,
                                    index_t K) const
    {
        // Raw shared memory
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        const index_t tid = threadIdx.x;

        if(block_m >= M) return;

        constexpr index_t elements_per_thread = (kM * kK) / kBlockSize;  // 8

        // ========================================================================
        // Phase 1: Load from global and store to LDS in ROW-MAJOR order
        // ========================================================================

        for(index_t i = 0; i < elements_per_thread; ++i)
        {
            index_t flat_idx = tid + i * kBlockSize;
            if(flat_idx < kM * kK)
            {
                index_t m_local = flat_idx / kK;
                index_t k_local = flat_idx % kK;

                // Load from global
                index_t global_idx = (block_m + m_local) * K + k_local;
                DataType val = input[global_idx];

                // Store to LDS with XOR addressing (or plain if not UseXor)
                index_t lds_addr = xor_address(m_local, k_local);
                lds[lds_addr] = val;
            }
        }

        __syncthreads();

        // ========================================================================
        // Phase 2: Read from LDS in TRANSPOSED pattern (multiple times)
        // Reading column-wise from row-major creates STRIDE-kK access!
        // ========================================================================

        constexpr int num_iterations = 50;

        for(int iter = 0; iter < num_iterations; ++iter)
        {
            DataType local_vals[elements_per_thread];

            // TRANSPOSE READ: Read column-wise (swap m and k roles)
            for(index_t i = 0; i < elements_per_thread; ++i)
            {
                index_t flat_idx = tid + i * kBlockSize;
                if(flat_idx < kM * kK)
                {
                    // TRANSPOSE: Treat flat_idx as [k][m] instead of [m][k]
                    index_t k_local = flat_idx / kM;  // ← Swapped!
                    index_t m_local = flat_idx % kM;  // ← Swapped!

                    if(k_local < kK && m_local < kM)
                    {
                        // Read from LDS at position [m_local][k_local]
                        // This creates STRIDE pattern when threads read columns!
                        index_t lds_addr = xor_address(m_local, k_local);
                        local_vals[i] = lds[lds_addr];  // BANK CONFLICTS HERE (plain mode)!
                    }
                }
            }

            __syncthreads();

            // Write back to keep data moving
            for(index_t i = 0; i < elements_per_thread; ++i)
            {
                index_t flat_idx = tid + i * kBlockSize;
                if(flat_idx < kM * kK)
                {
                    index_t k_local = flat_idx / kM;
                    index_t m_local = flat_idx % kM;

                    if(k_local < kK && m_local < kM)
                    {
                        index_t lds_addr = xor_address(m_local, k_local);
                        lds[lds_addr] = local_vals[i];
                    }
                }
            }

            __syncthreads();
        }

        // ========================================================================
        // Phase 3: Write to global in TRANSPOSED layout [K, M]
        // ========================================================================

        for(index_t i = 0; i < elements_per_thread; ++i)
        {
            index_t flat_idx = tid + i * kBlockSize;
            if(flat_idx < kM * kK)
            {
                // Read as transposed
                index_t k_local = flat_idx / kM;
                index_t m_local = flat_idx % kM;

                if(k_local < kK && m_local < kM)
                {
                    index_t lds_addr = xor_address(m_local, k_local);
                    DataType val = lds[lds_addr];

                    // Write to transposed global position
                    index_t global_out_idx = k_local * M + (block_m + m_local);
                    output[global_out_idx] = val;
                }
            }
        }
    }
};

template<bool UseXor>
bool run_test(const std::string& test_name)
{
    std::cout << "\n========================================\n";
    std::cout << test_name << "\n";
    std::cout << "========================================\n\n";

    constexpr index_t M = 256;
    constexpr index_t K = 128;

    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(K * M);

    // Initialize input
    for(index_t m = 0; m < M; ++m)
        for(index_t k = 0; k < K; ++k)
            h_input[m * K + k] = static_cast<DataType>(m * 1000 + k);

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(K * M * sizeof(DataType));

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    std::cout << "Configuration:\n";
    std::cout << "  Input:  [" << M << ", " << K << "]\n";
    std::cout << "  Output: [" << K << ", " << M << "] (transposed)\n";
    std::cout << "  LDS tile: [64, 32]\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads\n";
    std::cout << "  LDS access: Raw __shared__ with manual addressing\n";
    std::cout << "  Read pattern: COLUMN-MAJOR from row-major (TRANSPOSE)\n";
    std::cout << "  Iterations: 50× transposed reads\n";
    std::cout << "  XOR mode: " << (UseXor ? "ENABLED (manual XOR transform)" : "DISABLED (plain addressing)") << "\n\n";

    stream_config stream;

    launch_kernel(stream,
                 make_kernel<block_size>(
                     LDSTransposeKernel<DataType, UseXor>{},
                     dim3(grid_size),
                     dim3(block_size),
                     0,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    d_output.FromDevice(h_output.data(), K * M * sizeof(DataType));

    // Verify: output[k][m] == input[m][k]
    bool passed = true;
    index_t error_count = 0;

    for(index_t k = 0; k < K && error_count < 10; ++k)
    {
        for(index_t m = 0; m < M && error_count < 10; ++m)
        {
            DataType expected = h_input[m * K + k];
            DataType actual = h_output[k * M + m];

            if(bit_cast<uint16_t>(expected) != bit_cast<uint16_t>(actual))
            {
                std::cout << "Error at [" << k << "," << m << "]: "
                          << "expected " << static_cast<float>(expected)
                          << ", got " << static_cast<float>(actual) << "\n";
                error_count++;
                passed = false;
            }
        }
    }

    std::cout << "Results:\n";
    std::cout << "  Correctness: " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    if(!passed)
        std::cout << "  Errors: " << error_count << "+ (showing first 10)\n";

    return passed;
}

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Tutorial 11k: LDS Transpose with Manual XOR           ║\n";
    std::cout << "║ Raw __shared__ memory + manual addressing             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bool plain_passed = run_test<false>("Test 1: Plain LDS (Stride-32 transpose = CONFLICTS!)");
    bool xor_passed = run_test<true>("Test 2: XOR LDS (Manual XOR swizzle)");

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Summary                                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Plain LDS: " << (plain_passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "XOR LDS:   " << (xor_passed ? "✓ PASSED" : "✗ FAILED") << "\n\n";

    if(plain_passed && xor_passed)
    {
        std::cout << "Both correct! Now profile:\n\n";
        std::cout << "  rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \\\n";
        std::cout << "            -d /tmp/lds_transpose -- ./bin/aa_tutorial_11_xor_transpose_lds\n\n";
        std::cout << "This uses RAW __shared__ memory with:\n";
        std::cout << "  - Plain mode: Direct addressing\n";
        std::cout << "  - XOR mode:   m' = m XOR (k/8) for addressing\n";
        std::cout << "  - Transpose reads create stride-32 pattern\n\n";
        std::cout << "Expected: Plain mode HIGH conflicts, XOR mode REDUCED conflicts!\n";
    }

    return (plain_passed && xor_passed) ? 0 : 1;
}
