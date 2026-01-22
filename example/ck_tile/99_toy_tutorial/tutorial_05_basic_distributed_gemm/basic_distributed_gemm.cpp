// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Stage 10b: Distributed HGEMM - WORK IN PROGRESS
 *
 * This is saved work in progress. The structure is good but tile distributions
 * need to be fixed to properly match the MLSE example patterns.
 *
 * TODO: Fix tile_distribution_encoding for A and B
 * - Y dimensions should map to vector position
 * - Need 2x fp16 per register (32 bits)
 * - P0 = threads, P1 = warps typically
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <limits>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

using namespace ck_tile;

// Distributed HGEMM kernel using proper tile_distribution
template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
struct DistributedHgemmKernel
{
    static constexpr index_t kWaveSize = 64; // AMD wave size
    static constexpr index_t kBlockM   = 16; // MFMA M dimension
    static constexpr index_t kBlockN   = 16; // MFMA N dimension
    static constexpr index_t kBlockK   = 16; // MFMA K dimension per instruction

    // Use ck_tile's WarpGemm for MFMA
    using WarpGemm                      = WarpGemmMfmaF16F16F32M16N16K16;
    static constexpr index_t kBlockSize = kWaveSize;

    CK_TILE_DEVICE void operator()(const ADataType* a,
                                   const BDataType* b,
                                   const CDataType* c,
                                   CDataType* d,
                                   index_t M,
                                   index_t N,
                                   index_t K,
                                   index_t lda, // Leading dimension of A (column-major)
                                   index_t ldb, // Leading dimension of B (row-major)
                                   index_t ldc, // Leading dimension of C (column-major)
                                   index_t ldd, // Leading dimension of D (column-major)
                                   AccDataType alpha,
                                   AccDataType beta) const
    {
        // Calculate which 16×16 block this wave computes
        // const index_t wave_id = get_block_id() * get_block_size() / kWaveSize + threadIdx.x /
        // kWaveSize;
        const index_t wave_id = get_warp_id();
        const index_t wave_m  = wave_id / (N / kBlockN);
        const index_t wave_n  = wave_id % (N / kBlockN);

        const index_t m_offset = wave_m * kBlockM;
        const index_t n_offset = wave_n * kBlockN;

        // Bounds check
        if(m_offset >= M || n_offset >= N)
            return;

        // Only threads in the first wave of each block do work (simplified)
        if(threadIdx.x >= kWaveSize)
            return;

        // Create tensor views for matrices
        // A is column-major: M×K with stride lda between columns
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            a,
            make_tuple(M, K),   // Shape: M×K
            make_tuple(1, lda), // Strides: column-major
            number<1>{},
            number<1>{});

        // B is row-major: K×N with stride ldb between rows
        const auto b_tensor = make_naive_tensor_view<address_space_enum::global>(
            b,
            make_tuple(K, N),   // Shape: K×N
            make_tuple(ldb, 1), // Strides: row-major
            number<4>{},
            number<1>{});

        // C is column-major: M×N with stride ldc between columns
        const auto c_tensor = make_naive_tensor_view<address_space_enum::global>(
            c,
            make_tuple(M, N),   // Shape: M×N
            make_tuple(1, ldc), // Strides: column-major
            number<1>{},
            number<1>{});

        // D is column-major: M×N with stride ldd between columns
        auto d_tensor = make_naive_tensor_view<address_space_enum::global>(
            d,
            make_tuple(M, N),   // Shape: M×N
            make_tuple(1, ldd), // Strides: column-major
            number<1>{},
            number<1>{});

        // Use our tested custom distributions from test_a_distribution.cpp and
        // test_b_distribution.cpp A: Column-major M×K with each thread loading 4 consecutive K
        // values from one M position
        constexpr auto a_distribution = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,            // No replication
                                       tuple<sequence<16>,    // H0 (M): 16 lanes for M
                                             sequence<4, 4>>, // H1 (K): 4 lanes × 4 per lane
                                       tuple<sequence<2, 1>>, // P-dims map to H-dims
                                       tuple<sequence<0, 0>>, // P positions in H-dims
                                       sequence<2>,           // Y maps to K dimension only
                                       sequence<1>>{}         // Y at position 1
        );

        // B: Row-major K×N with each thread loading 4 consecutive K values from one N position
        constexpr auto b_distribution = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,            // No replication
                tuple<sequence<4, 4>,  // H0 (K): 4 groups of 4 consecutive K values
                      sequence<16>>,   // H1 (N): 16 N positions
                tuple<sequence<1, 2>>, // P-dims map to H-dims (P0->H1, P1->H0)
                tuple<sequence<0, 0>>, // P positions in H-dims
                sequence<1>,           // Y maps to K dimension (H0)
                sequence<1>>{}         // Y at position 1 in H0 (the second 4 in sequence<4,4>)
        );

        // Create windows for A and B that we'll move along K
        auto a_window = make_tile_window(a_tensor,
                                         make_tuple(number<kBlockM>{}, number<kBlockK>{}),
                                         {m_offset, 0},
                                         a_distribution);

        auto b_window = make_tile_window(b_tensor,
                                         make_tuple(number<kBlockK>{}, number<kBlockN>{}),
                                         {0, n_offset},
                                         b_distribution);

        // C distribution (column-major M×N output) - tested in test_c_distribution.cpp
        constexpr auto c_distribution = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,            // No replication
                tuple<sequence<4, 4>,  // H0 (M): 4 groups of 4 consecutive M values
                      sequence<16>>,   // H1 (N): 16 N positions
                tuple<sequence<1, 2>>, // P-dims map to H-dims (P0->H1, P1->H0)
                tuple<sequence<0, 0>>, // P positions in H-dims
                sequence<1>,           // Y maps to M dimension (H0)
                sequence<1>>{}         // Y at position 1 in H0 (the second 4 in sequence<4,4>)
        );

        // Create accumulator using our tested C distribution
        auto acc_tile = make_static_distributed_tensor<AccDataType>(c_distribution);

        // Initialize accumulator to zero using set_tile
        set_tile(acc_tile, AccDataType{0});

        // Main K-loop with MFMA accumulation
        const index_t num_k_loops = K / kBlockK;
        for(index_t k_iter = 0; k_iter < num_k_loops; ++k_iter)
        {
            // Load tiles
            const auto a_tile = load_tile(a_window);
            const auto b_tile = load_tile(b_window);

            // Use WarpGemm to perform MFMA
            // This properly calls the MFMA instruction with the right distributions
            WarpGemm{}(acc_tile, a_tile, b_tile);

            // Move windows to next K chunk using the move API
            // This efficiently updates window_origin_ without recreating the window
            if(k_iter < num_k_loops - 1)
            {
                a_window.move({0, kBlockK}); // Move K forward for A
                b_window.move({kBlockK, 0}); // Move K forward for B
            }
        }

        // Scale by alpha using ck_tile's elementwise API
        // This is more idiomatic than manual buffer manipulation
        tile_elementwise_inout([alpha](auto& acc_val) { acc_val *= alpha; }, acc_tile);

        // Load C, apply beta, and add to result
        if(std::abs(beta) > 1e-6f)
        {
            auto c_window = make_tile_window(c_tensor,
                                             make_tuple(number<kBlockM>{}, number<kBlockN>{}),
                                             {m_offset, n_offset},
                                             c_distribution);

            const auto c_tile = load_tile(c_window);

            // Apply beta * C + acc using ck_tile's elementwise API
            // This combines two tiles with a lambda function
            tile_elementwise_inout(
                [beta](const auto& c_val, auto& acc_val) { acc_val += beta * c_val; },
                c_tile,
                acc_tile);
        }

        // Store final result to D
        auto d_window = make_tile_window(d_tensor,
                                         make_tuple(number<kBlockM>{}, number<kBlockN>{}),
                                         {m_offset, n_offset},
                                         c_distribution);

        store_tile(d_window, acc_tile);
    }
};

// CPU reference for verification
template <typename InType, typename AccType>
void reference_gemm_mixed(const std::vector<InType>& a,  // Column-major
                          const std::vector<InType>& b,  // Row-major
                          const std::vector<AccType>& c, // Column-major
                          std::vector<AccType>& d,       // Column-major
                          index_t M,
                          index_t N,
                          index_t K,
                          index_t lda,
                          index_t ldb,
                          index_t ldc,
                          index_t ldd,
                          AccType alpha,
                          AccType beta)
{
    // D = alpha * A * B + beta * C
    for(index_t n = 0; n < N; ++n)
    {
        for(index_t m = 0; m < M; ++m)
        {
            AccType sum = 0;

            // Compute A * B
            for(index_t k = 0; k < K; ++k)
            {
                // A is column-major: A[m,k] = a[m + k*lda]
                // B is row-major: B[k,n] = b[k*ldb + n]
                sum += static_cast<AccType>(a[m + k * lda]) * static_cast<AccType>(b[k * ldb + n]);
            }

            // D[m,n] = alpha * sum + beta * C[m,n]
            // Both C and D are column-major
            d[m + n * ldd] = alpha * sum + beta * c[m + n * ldc];
        }
    }
}

// Helper to fill matrix with random values
template <typename T>
void fill_random(std::vector<T>& data, T min_val = -1, T max_val = 1)
{
    for(auto& val : data)
    {
        val = static_cast<T>(min_val + (max_val - min_val) * static_cast<float>(rand()) / RAND_MAX);
    }
}

// Helper to print matrix (for debugging)
template <typename T>
void print_matrix(const std::vector<T>& mat,
                  index_t rows,
                  index_t cols,
                  index_t ld,
                  bool col_major          = true,
                  const std::string& name = "Matrix")
{
    std::cout << name << " (" << rows << "×" << cols << "):\n";
    for(index_t i = 0; i < std::min(rows, index_t(8)); ++i)
    {
        for(index_t j = 0; j < std::min(cols, index_t(8)); ++j)
        {
            index_t idx = col_major ? (i + j * ld) : (i * ld + j);
            std::cout << std::setw(8) << std::setprecision(3) << mat[idx] << " ";
        }
        if(cols > 8)
            std::cout << "...";
        std::cout << "\n";
    }
    if(rows > 8)
        std::cout << "...\n";
    std::cout << "\n";
}

int main()
{
    std::cout << "\n==================================================\n";
    std::cout << "Stage 10b: Distributed HGEMM with ck_tile\n";
    std::cout << "==================================================\n\n";

    std::cout << "Key features:\n";
    std::cout << "• Uses tile_distribution for both A and B matrices\n";
    std::cout << "• A is column-major, B is row-major (like MLSE example)\n";
    std::cout << "• Half-precision inputs (fp16) with fp32 accumulation\n";
    std::cout << "• Non-contiguous loads for A and B\n";
    std::cout << "• Uses move_tile_window to advance along K dimension\n\n";

    // Test configuration - must be multiples of 16
    constexpr index_t M = 64;
    constexpr index_t N = 64;
    constexpr index_t K = 64;

    // Leading dimensions
    constexpr index_t lda = M; // Column-major
    constexpr index_t ldb = N; // Row-major
    constexpr index_t ldc = M; // Column-major
    constexpr index_t ldd = M; // Column-major

    using InputType = half_t; // fp16
    using AccumType = float;  // fp32

    constexpr AccumType alpha = 2.0f;
    constexpr AccumType beta  = 1.5f;

    std::cout << "Problem configuration:\n";
    std::cout << "  M×N×K: " << M << "×" << N << "×" << K << "\n";
    std::cout << "  A: column-major, lda=" << lda << " (fp16)\n";
    std::cout << "  B: row-major, ldb=" << ldb << " (fp16)\n";
    std::cout << "  C/D: column-major, ldc=" << ldc << ", ldd=" << ldd << " (fp32)\n";
    std::cout << "  alpha=" << alpha << ", beta=" << beta << "\n";
    std::cout << "  Total FLOPs: " << 2 * M * N * K << "\n\n";

    // Host memory
    std::vector<InputType> h_a(M * K);
    std::vector<InputType> h_b(K * N);
    std::vector<AccumType> h_c(M * N);
    std::vector<AccumType> h_d(M * N, std::numeric_limits<AccumType>::quiet_NaN());
    std::vector<AccumType> h_d_ref(M * N);

    // Initialize matrices
    srand(42);
    fill_random(h_a, InputType(-1), InputType(1));
    fill_random(h_b, InputType(-1), InputType(1));
    fill_random(h_c, AccumType(-1), AccumType(1));

    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    reference_gemm_mixed(h_a, h_b, h_c, h_d_ref, M, N, K, lda, ldb, ldc, ldd, alpha, beta);
    auto cpu_end       = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Device memory
    DeviceMem d_a(M * K * sizeof(InputType));
    DeviceMem d_b(K * N * sizeof(InputType));
    DeviceMem d_c(M * N * sizeof(AccumType));
    DeviceMem d_d(M * N * sizeof(AccumType));

    d_a.ToDevice(h_a.data(), M * K * sizeof(InputType));
    d_b.ToDevice(h_b.data(), K * N * sizeof(InputType));
    d_c.ToDevice(h_c.data(), M * N * sizeof(AccumType));
    d_d.ToDevice(h_d.data(), M * N * sizeof(AccumType));

    // Launch kernel
    constexpr index_t block_size = 64;                  // One wave
    const index_t grid_size      = (M / 16) * (N / 16); // One wave per 16×16 output block

    std::cout << "Launching kernel:\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads (1 wave)\n";
    std::cout << "  Output blocks: " << (M / 16) << "×" << (N / 16) << " = " << grid_size << "\n";
    std::cout << "  MFMA instructions per block: " << K / 16 << "\n\n";

    stream_config stream;

    // Warmup
    for(int i = 0; i < 5; ++i)
    {
        launch_kernel(stream,
                      make_kernel<block_size>(
                          DistributedHgemmKernel<InputType, InputType, AccumType, AccumType>{},
                          dim3(grid_size),
                          dim3(block_size),
                          0,
                          static_cast<const InputType*>(d_a.GetDeviceBuffer()),
                          static_cast<const InputType*>(d_b.GetDeviceBuffer()),
                          static_cast<const AccumType*>(d_c.GetDeviceBuffer()),
                          static_cast<AccumType*>(d_d.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          alpha,
                          beta));
    }
    hip_check_error(hipDeviceSynchronize());

    // Timed run
    auto gpu_start = std::chrono::high_resolution_clock::now();

    launch_kernel(stream,
                  make_kernel<block_size>(
                      DistributedHgemmKernel<InputType, InputType, AccumType, AccumType>{},
                      dim3(grid_size),
                      dim3(block_size),
                      0,
                      static_cast<const InputType*>(d_a.GetDeviceBuffer()),
                      static_cast<const InputType*>(d_b.GetDeviceBuffer()),
                      static_cast<const AccumType*>(d_c.GetDeviceBuffer()),
                      static_cast<AccumType*>(d_d.GetDeviceBuffer()),
                      M,
                      N,
                      K,
                      lda,
                      ldb,
                      ldc,
                      ldd,
                      alpha,
                      beta));

    hip_check_error(hipDeviceSynchronize());

    auto gpu_end       = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    // Get result
    d_d.FromDevice(h_d.data(), M * N * sizeof(AccumType));

    // Verify
    bool passed         = true;
    float max_error     = 0;
    index_t error_count = 0;

    for(index_t i = 0; i < M * N; ++i)
    {
        float error = std::abs(h_d[i] - h_d_ref[i]);
        max_error   = std::max(max_error, error);
        if(error > 1e-2f)
        { // Relaxed tolerance for fp16
            if(error_count < 5)
            {
                index_t m = i % M;
                index_t n = i / M;
                std::cout << "Error at [" << m << "," << n << "]: " << h_d[i] << " vs "
                          << h_d_ref[i] << " (diff=" << error << ")\n";
            }
            error_count++;
        }
    }

    passed = (error_count == 0);

    // Calculate performance
    double gflops     = 2.0 * M * N * K / 1e9;
    double gpu_tflops = gflops / (gpu_time_ms / 1000);
    double cpu_gflops = gflops / (cpu_time_ms / 1000);

    std::cout << "Results:\n";
    std::cout << "  Correctness: " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "  Max error: " << max_error << "\n";
    if(!passed)
        std::cout << "  Error count: " << error_count << "/" << M * N << "\n";
    std::cout << "\n";

    std::cout << "Performance:\n";
    std::cout << "  CPU time: " << cpu_time_ms << " ms (" << cpu_gflops << " GFLOPS)\n";
    std::cout << "  GPU time: " << gpu_time_ms << " ms (" << gpu_tflops << " TFLOPS)\n";
    std::cout << "  Speedup: " << cpu_time_ms / gpu_time_ms << "x\n\n";

#ifdef DEBUG_OUTPUT
    // Print sample outputs for debugging
    print_matrix(h_a, M, K, lda, true, "A (col-major)");
    print_matrix(h_b, K, N, ldb, false, "B (row-major)");
    print_matrix(h_c, M, N, ldc, true, "C (col-major)");
    print_matrix(h_d_ref, M, N, ldd, true, "D_ref (col-major)");
    print_matrix(h_d, M, N, ldd, true, "D_gpu (col-major)");
#endif

    std::cout << "=== Key Insights ===\n";
    std::cout << "• Y dimensions should map to vector position in hierarchical factorization\n";
    std::cout << "• move_tile_window efficiently advances along K dimension\n";
    std::cout << "• Column-major A and row-major B require different distributions\n";
    std::cout << "• Each thread loads 4 elements (2x fp16 = 32 bits per load)\n";
    std::cout << "• MFMA efficiently computes 16×16×16 in one instruction\n";
    std::cout << "• This pattern extends to production GEMM kernels\n\n";

    return passed ? 0 : 1;
}
