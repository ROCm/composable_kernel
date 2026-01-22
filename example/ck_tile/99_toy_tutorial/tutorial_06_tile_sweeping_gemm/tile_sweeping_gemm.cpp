// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 06: Tile Sweeping GEMM
 *
 * Demonstrates tile sweeping pattern with multiple warps cooperating
 * to compute larger output blocks. This tutorial shows how warps sweep
 * over multiple tiles using static_for loops and move_tile_window.
 *
 * Key concepts:
 * - Multiple warps per block (2×2 warp configuration)
 * - Each warp computes multiple output tiles (tile sweeping)
 * - Tile distributions with replication (B matrix)
 * - Using static_for to iterate over tiles
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

// Tile Sweeping HGEMM kernel with multiple warps
template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
struct TileSweepingHgemmKernel
{
    static constexpr index_t kWaveSize = 64; // AMD wave size
    static constexpr index_t kWarpM    = 16; // MFMA M dimension per warp
    static constexpr index_t kWarpN    = 16; // MFMA N dimension per warp
    static constexpr index_t kWarpK    = 16; // MFMA K dimension per instruction

    // Warp configuration: 2×2 warps per block
    static constexpr index_t MWarp      = 2;                         // 2 warps in M dimension
    static constexpr index_t NWarp      = 2;                         // 2 warps in N dimension
    static constexpr index_t kBlockSize = MWarp * NWarp * kWaveSize; // 256 threads

    // No iterations - each warp computes exactly one 16×16 output tile

    // Use ck_tile's WarpGemm for MFMA
    using WarpGemm = WarpGemmMfmaF16F16F32M16N16K16;

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
        // Calculate which warp this thread belongs to within the block
        const index_t warp_id  = get_warp_id();
        const index_t iMWarp   = warp_id / NWarp; // M-warp index (0 or 1)
        const index_t iNWarp   = warp_id % NWarp; // N-warp index (0 or 1)
        const index_t block_id = get_block_id();

        // Convert linear block_id to 2D grid coordinates
        const index_t num_blocks_n = N / (NWarp * kWarpN);    // Number of blocks in N dimension
        const index_t block_m      = block_id / num_blocks_n; // M-block index
        const index_t block_n      = block_id % num_blocks_n; // N-block index

        // printf("Block %d (grid [%d,%d]), Warp %d (M-warp %d, N-warp %d)\n",
        //        block_id, block_m, block_n, warp_id, iMWarp, iNWarp);

        // Calculate base offset for this warp's single tile
        const index_t m_warp_base = block_m * (MWarp * kWarpM) + iMWarp * kWarpM;
        const index_t n_warp_base = block_n * (NWarp * kWarpN) + iNWarp * kWarpN;

        // Bounds check for the warp's entire region
        if(m_warp_base >= M || n_warp_base >= N)
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

        // ============================================================================
        // TILE DISTRIBUTIONS using EMBED API (from verified tests)
        // ============================================================================

        // Step 1: Warp-level distribution (64 threads within one warp)
        constexpr auto a_warp_dstr_encode =
            tile_distribution_encoding<sequence<>,            // No replication at warp level
                                       tuple<sequence<16>,    // H0 (M): 16 M positions
                                             sequence<4, 4>>, // H1 (K): 4×4 = 16 K elements
                                       tuple<sequence<2, 1>>, // Ps_to_Hs: 2D P-space (64 threads)
                                       tuple<sequence<0, 0>>, // Ps_in_Hs
                                       sequence<2>,           // Ys_to_Hs: Y maps to K
                                       sequence<1>>{};        // Ys_in_Hs

        // Step 2: Block-level outer distribution (warp organization)
        // Must have same NDimX as inner (2 dimensions: M and K)
        constexpr auto a_block_outer_dstr_encode =
            tile_distribution_encoding<sequence<NWarp>, // R: Replicate across N-warps
                                       tuple<sequence<MWarp>, sequence<>>, // H: MWarp in M-dim, 1
                                                                           // in K-dim
                                       tuple<sequence<0, 1>>,              // Ps_to_Hs
                                       tuple<sequence<0, 0>>,              // Ps_in_Hs
                                       sequence<>,    // Ys_to_Hs: Y maps to both M and K
                                       sequence<>>{}; // Ys_in_Hs

        // B warp-level distribution
        constexpr auto b_warp_dstr_encode =
            tile_distribution_encoding<sequence<>,            // No replication at warp level
                                       tuple<sequence<4, 4>,  // H0 (K): 4×4 = 16 K elements
                                             sequence<16>>,   // H1 (N): 16 N positions
                                       tuple<sequence<1, 2>>, // Ps_to_Hs: 1 sequence with 2 values
                                                              // (2D P-space)
                                       tuple<sequence<0, 0>>, // Ps_in_Hs
                                       sequence<1>,           // Ys_to_Hs: Y maps to K
                                       sequence<1>>{};        // Ys_in_Hs

        // Step 2: Block-level outer distribution (warp organization)
        // Must have same NDimX as inner (2 dimensions: K and N)
        constexpr auto b_block_outer_dstr_encode =
            tile_distribution_encoding<sequence<MWarp>, // R: Replicate across M-warps
                                       tuple<sequence<>, sequence<NWarp>>, // H: NWarp in N-dim, 1
                                                                           // in K-dim
                                       tuple<sequence<2, 0>>,              // Ps_to_Hs
                                       tuple<sequence<0, 0>>,              // Ps_in_Hs
                                       sequence<>,    // Ys_to_Hs: Y maps to both K and N
                                       sequence<>>{}; // Ys_in_Hs

        // Embed to create block-level distributions with replication
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encode, a_warp_dstr_encode);

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encode, b_warp_dstr_encode);

        // /*direct approach*/
        // constexpr auto a_block_dstr_encode =
        //     tile_distribution_encoding<
        //         sequence<NWarp>,                      // R: REPLICATE across 2 N-warps
        //         tuple<sequence<MWarp, 16>,            // H0 (M): 2 M-warps × 16 threads = 32 M
        //               sequence<4, 4>>,                // H1 (K): 4×4 = 16 K elements
        //         tuple<sequence<0, 1>, sequence<2, 1>>,  // Ps_to_Hs: P0→(R,M), P1→(M,K)
        //         tuple<sequence<0, 0>, sequence<0, 1>>,  // Ps_in_Hs: positions
        //         sequence<2>,                          // Ys_to_Hs: Y maps to K (dimension 2)
        //         sequence<1>>{};                        // Ys_in_Hs: Y at position 1 in K

        // // Direct approach (like test_b_distribution_with_replication.cpp)
        // constexpr auto b_block_dstr_encode =
        //     tile_distribution_encoding<
        //         sequence<MWarp>,                      // R: dimension 0, REPLICATE across 2
        //         M-warps tuple<sequence<4, 4>,                 // H: dimension 1 (K): 4×4 = 16 K
        //         elements
        //               sequence<2, 16>>,                  // H: dimension 2 (N): 16 N positions
        //         tuple<sequence<2, 0>, sequence<1, 2>>,  // Ps_to_Hs: P0→R(dim 0), P1→K(dim 1),
        //         P2→N(dim 2) tuple<sequence<0, 0>, sequence<0, 1>>,  // Ps_in_Hs: positions
        //         sequence<1>,                          // Ys_to_Hs: Y maps to K (dimension 1)
        //         sequence<1>>{};                        // Ys_in_Hs: Y at position 1 in K
        // /*direct approach*/

        // Use block-level distributions for loading (includes replication)
        constexpr auto a_block_distribution = make_static_tile_distribution(a_block_dstr_encode);
        constexpr auto b_block_distribution = make_static_tile_distribution(b_block_dstr_encode);

        // C Distribution: Create block-level distribution for 32×32 output
        // No replication needed - each warp computes its own unique output region
        // 2D P-space for 4 warps: P[0] for M-warp, P[1] for N-warp
        // constexpr auto c_block_dstr_encode = tile_distribution_encoding<
        //     sequence<>,                                 // No replication for output
        //     tuple<sequence<MWarp, 4, 4>,                // H0 (M): 2 M-warps × 16 threads = 32
        //           sequence<NWarp, 16>>,                 // H1 (N): 2 N-warps × 16 threads = 32
        //     tuple<sequence<2, 1>, sequence<1, 2>>,      // Ps_to_Hs: P[0]→M, P[1]→N (2D P-space)
        //     tuple<sequence<0, 0>, sequence<1, 1>>,      // Ps_in_Hs
        //     sequence<1>,                                // No Y dimension for output
        //     sequence<2>>{};

        constexpr auto c_warp_dstr_encode =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<4, 4>, sequence<16>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<0, 0>>,
                                       sequence<1>,
                                       sequence<1>>{};

        constexpr auto c_block_outer_dstr_encode =
            tile_distribution_encoding<sequence<>,             // No replication for output
                                       tuple<sequence<MWarp>,  // H0: M iterations
                                             sequence<NWarp>>, // H1: N iterations
                                       tuple<sequence<2, 1>>,  // Ps_to_Hs
                                       tuple<sequence<0, 0>>,  // Ps_in_Hs
                                       sequence<>,             // Ys_to_Hs: Y maps to BOTH M and N
                                       sequence<>>{};          // Ys_in_Hs

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encode, c_warp_dstr_encode);

        constexpr auto c_block_distribution = make_static_tile_distribution(c_block_dstr_encode);

        // Create block-level windows (one K-chunk at a time)
        // A: 32×16 (all M-rows × one K-chunk)
        // B: 16×32 (one K-chunk × all N-columns)
        auto a_block_window =
            make_tile_window(a_tensor,
                             make_tuple(number<MWarp * kWarpM>{}, number<kWarpK>{}), // 32×16
                             {block_m * (MWarp * kWarpM), 0},
                             a_block_distribution);

        auto b_block_window =
            make_tile_window(b_tensor,
                             make_tuple(number<kWarpK>{}, number<NWarp * kWarpN>{}), // 16×32
                             {0, block_n * (NWarp * kWarpN)},
                             b_block_distribution);

        // Create block-level accumulator tile (covers all 4 warps)
        auto c_block_tile = make_static_distributed_tensor<AccDataType>(c_block_distribution);
        set_tile(c_block_tile, AccDataType{0});

        // Main K-loop
        const index_t num_k_loops = K / kWarpK;
        for(index_t k_iter = 0; k_iter < num_k_loops; ++k_iter)
        {
            // Load block tiles - distribution handles replication automatically
            // Each warp gets its correct portion based on the distribution encoding
            const auto a_tile = load_tile(a_block_window);
            const auto b_tile = load_tile(b_block_window);

            // Perform MFMA: C += A * B
            // Each warp updates its portion of the block tile
            WarpGemm{}(c_block_tile, a_tile, b_tile);

            // // Move windows to next K chunk
            if(k_iter < num_k_loops - 1)
            {
                move_tile_window(a_block_window, {0, kWarpK});
                move_tile_window(b_block_window, {kWarpK, 0});
            }
        }

        // Scale by alpha
        tile_elementwise_inout([alpha](auto& acc_val) { acc_val *= alpha; }, c_block_tile);

        // Add beta * C if needed (load entire block C)
        if(std::abs(beta) > 1e-6f)
        {
            auto c_block_window = make_tile_window(
                c_tensor,
                make_tuple(number<MWarp * kWarpM>{}, number<NWarp * kWarpN>{}), // 32×32
                {block_m * (MWarp * kWarpM), block_n * (NWarp * kWarpN)},
                c_block_distribution);

            const auto c_input_block_tile = load_tile(c_block_window);

            tile_elementwise_inout(
                [beta](const auto& c_val, auto& acc_val) { acc_val += beta * c_val; },
                c_input_block_tile,
                c_block_tile);
        }

        // Store final result to D (entire block)
        auto d_block_window = make_tile_window(
            d_tensor,
            make_tuple(number<MWarp * kWarpM>{}, number<NWarp * kWarpN>{}), // 32×32
            {block_m * (MWarp * kWarpM), block_n * (NWarp * kWarpN)},
            c_block_distribution);

        store_tile(d_block_window, c_block_tile);
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
    std::cout << "Tutorial 06: Tile Sweeping GEMM\n";
    std::cout << "==================================================\n\n";

    std::cout << "Key features:\n";
    std::cout << "• Multiple warps per block (2×2 warp configuration)\n";
    std::cout << "• Each warp sweeps over 4×4 output tiles\n";
    std::cout << "• Tile distribution with replication (B matrix)\n";
    std::cout << "• Uses static_for loops for tile iteration\n";
    std::cout << "• Uses move_tile_window to position windows\n\n";

    // Test configuration - simple 4-warp example
    constexpr index_t M = 64;
    constexpr index_t N = 64;
    constexpr index_t K = 32;

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
    constexpr index_t block_size        = 256; // 4 warps (2×2 configuration)
    constexpr index_t tiles_per_block_m = 2;   // MWarp (no iterations)
    constexpr index_t tiles_per_block_n = 2;   // NWarp (no iterations)
    const index_t grid_size = (M / (tiles_per_block_m * 16)) * (N / (tiles_per_block_n * 16));

    std::cout << "Launching kernel:\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads (4 warps in 2×2 config)\n";
    std::cout << "  Each warp computes: ONE 16×16 output tile\n";
    std::cout << "  Each block computes: " << tiles_per_block_m * 16 << "×"
              << tiles_per_block_n * 16 << " output\n";
    std::cout << "  Total output tiles: " << (M / 16) << "×" << (N / 16) << "\n";
    std::cout << "  MFMA instructions per warp: " << (K / 16) << "\n\n";

    stream_config stream;

    // Warmup
    for(int i = 0; i < 5; ++i)
    {
        launch_kernel(stream,
                      make_kernel<block_size>(
                          TileSweepingHgemmKernel<InputType, InputType, AccumType, AccumType>{},
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
                      TileSweepingHgemmKernel<InputType, InputType, AccumType, AccumType>{},
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
    std::cout << "• Tile sweeping allows warps to compute multiple output tiles\n";
    std::cout << "• static_for loops iterate over tiles at compile time\n";
    std::cout << "• move_tile_window positions windows at different tiles\n";
    std::cout << "• A matrix REPLICATES across N-warps (warps in same M-row need same A)\n";
    std::cout << "• B matrix REPLICATES across M-warps (warps in same N-column need same B)\n";
    std::cout << "• Replication in R parameter: A has sequence<NWarp>, B has sequence<MWarp>\n";
    std::cout << "• This pattern scales to production GEMM kernels\n";
    std::cout << "• 2×2 warp config with 4×4 tiles per warp = 128×128 output per block\n\n";

    return passed ? 0 : 1;
}
