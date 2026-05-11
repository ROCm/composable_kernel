// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 08: Simple LDS Staging
 *
 * Adds LDS (Local Data Share / shared memory) staging to demonstrate data reuse.
 * This is the SIMPLE version - uses the same distributions for all operations.
 * Tutorial 09 will add optimizations like separate copy distributions.
 *
 * Key concepts:
 * - Global Memory → LDS → Registers → MFMA (memory hierarchy)
 * - kKPerBlock = 32 for temporal reuse (vs kWarpK = 16)
 * - KIterPerWarp = 2: iterate over K-chunks within LDS
 * - block_sync_lds() for synchronization
 * - Same distributions used for all operations (simple!)
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

// Simple LDS Staging HGEMM kernel
template<typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
struct SimpleLdsStagingHgemmKernel
{
    static constexpr index_t kWaveSize = 64;   // AMD wave size
    static constexpr index_t kWarpM = 16;      // MFMA M dimension per warp
    static constexpr index_t kWarpN = 16;      // MFMA N dimension per warp
    static constexpr index_t kWarpK = 16;      // MFMA K dimension per instruction

    // Warp configuration: 2×2 warps per block
    static constexpr index_t MWarp = 2;        // 2 warps in M dimension
    static constexpr index_t NWarp = 2;        // 2 warps in N dimension
    static constexpr index_t kBlockSize = MWarp * NWarp * kWaveSize;  // 256 threads

    // Tile iterations per warp (Y-dimension repetition)
    static constexpr index_t MIterPerWarp = 2;  // Each warp sweeps 2 tiles in M
    static constexpr index_t NIterPerWarp = 2;  // Each warp sweeps 2 tiles in N

    // NEW: Larger K-tile for temporal reuse!
    static constexpr index_t kKPerBlock = 32;   // K-tile loaded to LDS (was 16)
    static constexpr index_t KIterPerWarp = kKPerBlock / kWarpK;  // = 2 (was 1)

    // Use ck_tile's WarpGemm for MFMA
    using WarpGemm = WarpGemmMfmaF16F16F32M16N16K16;

    // LDS size calculation
    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        constexpr index_t kMPerBlock = MWarp * MIterPerWarp * kWarpM;  // 64
        constexpr index_t kNPerBlock = NWarp * NIterPerWarp * kWarpN;  // 64

        constexpr index_t a_lds_size = kMPerBlock * kKPerBlock * sizeof(ADataType);  // 64*32*2 = 4096
        constexpr index_t b_lds_size = kNPerBlock * kKPerBlock * sizeof(BDataType);  // 64*32*2 = 4096

        // Align A to 16 bytes
        constexpr index_t a_lds_aligned = ((a_lds_size + 15) / 16) * 16;
        return a_lds_aligned + b_lds_size;  // ~8KB total
    }

    CK_TILE_DEVICE void operator()(const ADataType* a,
                                   const BDataType* b,
                                   const CDataType* c,
                                   CDataType* d,
                                   index_t M,
                                   index_t N,
                                   index_t K,
                                   index_t lda,  // Leading dimension of A (column-major)
                                   index_t ldb,  // Leading dimension of B (row-major)
                                   index_t ldc,  // Leading dimension of C (column-major)
                                   index_t ldd,  // Leading dimension of D (column-major)
                                   AccDataType alpha,
                                   AccDataType beta) const
    {
        // Get dynamic shared memory
        extern __shared__ char smem[];
        void* p_smem = static_cast<void*>(smem);
        // Calculate which warp this thread belongs to within the block
        [[maybe_unused]] const index_t warp_id = get_warp_id();
        [[maybe_unused]] const index_t iMWarp = warp_id / NWarp;  // M-warp index (0 or 1)
        [[maybe_unused]] const index_t iNWarp = warp_id % NWarp;  // N-warp index (0 or 1)

        // Block dimensions
        constexpr index_t kMPerBlock = MWarp * MIterPerWarp * kWarpM;  // 64
        constexpr index_t kNPerBlock = NWarp * NIterPerWarp * kWarpN;  // 64

        // Calculate block position in 2D grid
        const index_t num_blocks_n = N / kNPerBlock;
        const index_t block_m = get_block_id() / num_blocks_n;
        const index_t block_n = get_block_id() % num_blocks_n;
        
        const index_t m_block_base = block_m * kMPerBlock;
        const index_t n_block_base = block_n * kNPerBlock;

        // Bounds check
        if(m_block_base >= M || n_block_base >= N)
            return;

        // Create tensor views for matrices
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            a,
            make_tuple(M, K),
            make_tuple(1, lda),
            number<1>{},
            number<1>{}
        );

        const auto b_tensor = make_naive_tensor_view<address_space_enum::global>(
            b,
            make_tuple(K, N),
            make_tuple(ldb, 1),
            number<4>{},
            number<1>{}
        );

        const auto c_tensor = make_naive_tensor_view<address_space_enum::global>(
            c,
            make_tuple(M, N),
            make_tuple(1, ldc),
            number<1>{},
            number<1>{}
        );

        auto d_tensor = make_naive_tensor_view<address_space_enum::global>(
            d,
            make_tuple(M, N),
            make_tuple(1, ldd),
            number<1>{},
            number<1>{}
        );

        // ============================================================================
        // LDS SETUP (Tutorial 08 Addition)
        // ============================================================================

        // Create LDS descriptors using packed layout (row-major by default)
        constexpr auto a_lds_desc = make_naive_tensor_descriptor_packed(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}));

        constexpr auto b_lds_desc = make_naive_tensor_descriptor_packed(
            make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}));

        // Allocate LDS space
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);
        constexpr index_t a_lds_size_aligned =
            ((kMPerBlock * kKPerBlock * sizeof(ADataType) + 15) / 16) * 16;
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_size_aligned));

        // Create LDS tensor views
        auto a_lds_view = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_desc);
        auto b_lds_view = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_desc);

        // ============================================================================
        // TILE DISTRIBUTIONS with Y-DIMENSION REPETITION (following 02_gemm pattern)
        // ============================================================================
        
        // A Distribution: Block-level with Y-repetition
        // Warp-level distribution (same as Tutorial 06)
        constexpr auto a_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<16>, sequence<4, 4>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>{};

        // constexpr auto a_block_outer_dstr_encoding =
        //     tile_distribution_encoding<sequence<NWarp>,
        //                                tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
        //                                tuple<sequence<1, 0>>,
        //                                tuple<sequence<1, 0>>,
        //                                sequence<1, 2>,
        //                                sequence<0, 0>>{};
        
        // Block-level outer distribution with Y-repetition
        constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<NWarp>,                                                        // Replicate across N-warps
            tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,               // H0: 2 iters × 2 warps in M
            tuple<sequence<0, 1>>,                              // Ps_to_Hs
            tuple<sequence<0, 1>>,                              // Ps_in_Hs
            sequence<1, 2>,                                     // Ys_to_Hs: Y maps to BOTH M and K
            sequence<0, 0>>{};                                  // Ys_in_Hs

        constexpr auto a_block_dstr_encode = 
            detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encode, a_warp_dstr_encode);


        // B Distribution: Block-level with Y-repetition
        constexpr auto b_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<4, 4>, sequence<16>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<0, 0>>,
            sequence<1>,
            sequence<1>>{};
        
        // constexpr auto b_block_outer_dstr_encode =
        //     tile_distribution_encoding<sequence<MWarp>,
        //                                tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
        //                                tuple<sequence<0, 1>>,
        //                                tuple<sequence<0, 1>>,
        //                                sequence<1, 2>,
        //                                sequence<0, 0>>{};

        constexpr auto b_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<MWarp>,                                    // Replicate across M-warps
            tuple<sequence<KIterPerWarp>,                // H0: 2 iters × 2 warps in N
                  sequence<NIterPerWarp, NWarp>>,                      // H1: 1 K-chunk
            tuple<sequence<2, 0>>,                              // Ps_to_Hs
            tuple<sequence<1, 0>>,                              // Ps_in_Hs
            sequence<1, 2>,                                     // Ys_to_Hs: Y maps to BOTH N and K
            sequence<0, 0>>{};                                  // Ys_in_Hs

        constexpr auto b_block_dstr_encode = 
            detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encode, b_warp_dstr_encode);

        // // C Distribution: Block-level with Y-repetition for output
        constexpr auto c_warp_dstr_encode = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<4, 4>, sequence<16>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<0, 0>>,
            sequence<1>,
            sequence<1>>{};

        // constexpr auto c_block_outer_dstr_encode = tile_distribution_encoding<
        //     sequence<>,
        //     tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
        //     tuple<sequence<1, 2>>,
        //     tuple<sequence<1, 1>>,
        //     sequence<1, 2>,
        //     sequence<0, 0>>{};

        constexpr auto c_block_outer_dstr_encode = tile_distribution_encoding<
            sequence<>,                                         // No replication for output
            tuple<sequence<MIterPerWarp, MWarp>,                // H0: M iterations
                  sequence<NIterPerWarp, NWarp>>,               // H1: N iterations
            tuple<sequence<2, 1>>,                              // Ps_to_Hs
            tuple<sequence<1, 1>>,                              // Ps_in_Hs
            sequence<1, 2>,                                     // Ys_to_Hs: Y maps to BOTH M and N
            sequence<0, 0>>{};                                  // Ys_in_Hs

        constexpr auto c_block_dstr_encode = 
            detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encode, c_warp_dstr_encode);

        // Create distributions
        constexpr auto a_block_distribution = make_static_tile_distribution(a_block_dstr_encode);
        constexpr auto b_block_distribution = make_static_tile_distribution(b_block_dstr_encode);
        constexpr auto c_block_distribution = make_static_tile_distribution(c_block_dstr_encode);

        // Get Y-dimension information for slicing
        using AWarpDstr = decltype(make_static_tile_distribution(a_warp_dstr_encode));
        using BWarpDstr = decltype(make_static_tile_distribution(b_warp_dstr_encode));
        using CWarpDstr = decltype(make_static_tile_distribution(c_warp_dstr_encode));

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // Create global memory windows (size changed to kKPerBlock!)
        auto a_global_window = make_tile_window(
            a_tensor,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),  // 64x32 (was 64x16)
            {m_block_base, 0},
            a_block_distribution
        );

        auto b_global_window = make_tile_window(
            b_tensor,
            make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),  // 32x64 (was 16x64)
            {0, n_block_base},
            b_block_distribution
        );

        // Create LDS windows (same distribution - simple!)
        auto a_lds_window = make_tile_window(
            a_lds_view,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),  // 64x32
            {0, 0},
            a_block_distribution  // Reuse same distribution
        );

        auto b_lds_window = make_tile_window(
            b_lds_view,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),  // 64x32
            {0, 0},
            b_block_distribution  // Reuse same distribution
        );

        // Create block-level accumulator tile
        auto c_block_tile = make_static_distributed_tensor<AccDataType>(c_block_distribution);
        set_tile(c_block_tile, AccDataType{0});

        // Main K-loop with LDS staging
        const index_t num_k_loops = K / kKPerBlock;  // K/32 (was K/16)
        for(index_t k_iter = 0; k_iter < num_k_loops; ++k_iter)
        {
            // Phase 1: Global -> Registers
            const auto a_global_tile = load_tile(a_global_window);
            const auto b_global_tile = load_tile(b_global_window);

            // Phase 2: Registers -> LDS
            store_tile(a_lds_window, a_global_tile);
            store_tile(b_lds_window, b_global_tile);

            // Phase 3: Synchronize
            block_sync_lds();

            // Phase 4: LDS -> Registers (for GEMM)
            const auto a_block_tile = load_tile(a_lds_window);
            const auto b_block_tile = load_tile(b_lds_window);
            
            // Nested loops over tile iterations using Y-slicing (like 02_gemm)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // Extract A warp tensor for this M-iteration using Y-slicing
                    auto a_warp_tensor = make_static_distributed_tensor<ADataType>(
                        make_static_tile_distribution(a_warp_dstr_encode));
                    
                    a_warp_tensor.get_thread_buffer() = a_block_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                    
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // Extract B warp tensor for this N-iteration using Y-slicing
                        auto b_warp_tensor = make_static_distributed_tensor<BDataType>(
                            make_static_tile_distribution(b_warp_dstr_encode));
                        
                        b_warp_tensor.get_thread_buffer() = b_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<kIter, nIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                        
                        // Extract C warp tensor for this M,N iteration
                        auto c_warp_tensor = make_static_distributed_tensor<AccDataType>(
                            make_static_tile_distribution(c_warp_dstr_encode));
                        
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                        
                        // Warp GEMM: C[mIter, nIter] += A[mIter, kIter] * B[nIter, kIter]
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                        
                        // Write C warp tensor back to block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });

            // Move global windows to next K chunk
            if(k_iter < num_k_loops - 1) {
                move_tile_window(a_global_window, {0, kKPerBlock});  // Move by 32 (was 16)
                move_tile_window(b_global_window, {kKPerBlock, 0});
            }
        }

        // Scale by alpha
        tile_elementwise_inout([alpha](auto& acc_val) { acc_val *= alpha; }, c_block_tile);

        // Add beta * C if needed
        if(std::abs(beta) > 1e-6f)
        {
            auto c_block_window = make_tile_window(
                c_tensor,
                make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                {m_block_base, n_block_base},
                c_block_distribution
            );

            const auto c_input_block_tile = load_tile(c_block_window);

            tile_elementwise_inout(
                [beta](const auto& c_val, auto& acc_val) {
                    acc_val += beta * c_val;
                },
                c_input_block_tile, c_block_tile);
        }

        // Store final result to D
        auto d_block_window = make_tile_window(
            d_tensor,
            make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
            {m_block_base, n_block_base},
            c_block_distribution
        );

        store_tile(d_block_window, c_block_tile);
    }
};

// CPU reference for verification
template<typename InType, typename AccType>
void reference_gemm_mixed(const std::vector<InType>& a,
                          const std::vector<InType>& b,
                          const std::vector<AccType>& c,
                          std::vector<AccType>& d,
                          index_t M, index_t N, index_t K,
                          index_t lda, index_t ldb, index_t ldc, index_t ldd,
                          AccType alpha, AccType beta)
{
    for(index_t n = 0; n < N; ++n) {
        for(index_t m = 0; m < M; ++m) {
            AccType sum = 0;
            for(index_t k = 0; k < K; ++k) {
                sum += static_cast<AccType>(a[m + k * lda]) *
                       static_cast<AccType>(b[k * ldb + n]);
            }
            d[m + n * ldd] = alpha * sum + beta * c[m + n * ldc];
        }
    }
}

template<typename T>
void fill_random(std::vector<T>& data, T min_val = -1, T max_val = 1)
{
    for(auto& val : data) {
        val = static_cast<T>(min_val + (max_val - min_val) *
                             static_cast<float>(rand()) / RAND_MAX);
    }
}

int main()
{
    std::cout << "\n==================================================\n";
    std::cout << "Tutorial 08: Simple LDS Staging\n";
    std::cout << "==================================================\n\n";

    std::cout << "Key features:\n";
    std::cout << "• Adds LDS (shared memory) for data reuse\n";
    std::cout << "• kKPerBlock=32 for temporal reuse (vs kWarpK=16)\n";
    std::cout << "• KIterPerWarp=2: iterate over K-chunks within LDS\n";
    std::cout << "• Global → LDS → Registers → MFMA data flow\n";
    std::cout << "• Same distributions for all operations (simple!)\n";
    std::cout << "• block_sync_lds() for synchronization\n\n";

    // Problem size: Each block computes 64×64 (2×2 warps × 2×2 iters × 16×16)
    // constexpr index_t M = 128;
    // constexpr index_t N = 128;
    // constexpr index_t K = 64;
    
    constexpr index_t M = 128;
    constexpr index_t N = 128;
    constexpr index_t K = 64;

    constexpr index_t lda = M;
    constexpr index_t ldb = N;
    constexpr index_t ldc = M;
    constexpr index_t ldd = M;

    using InputType = half_t;
    using AccumType = float;

    constexpr AccumType alpha = 2.0f;
    constexpr AccumType beta = 1.5f;

    std::cout << "Problem configuration:\n";
    std::cout << "  M×N×K: " << M << "×" << N << "×" << K << "\n";
    std::cout << "  Block output: 64×64 (2 warps × 2 iters × 16)\n";
    std::cout << "  Warp output: 32×32 (2 iters × 16 in each dim)\n";
    std::cout << "  Total blocks: " << (M/64) << "×" << (N/64) << "\n\n";

    // Host memory
    std::vector<InputType> h_a(M * K);
    std::vector<InputType> h_b(K * N);
    std::vector<AccumType> h_c(M * N);
    std::vector<AccumType> h_d(M * N, std::numeric_limits<AccumType>::quiet_NaN());
    std::vector<AccumType> h_d_ref(M * N);

    srand(42);
    fill_random(h_a, InputType(-1), InputType(1));
    fill_random(h_b, InputType(-1), InputType(1));
    fill_random(h_c, AccumType(-1), AccumType(1));

    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    reference_gemm_mixed(h_a, h_b, h_c, h_d_ref, M, N, K, lda, ldb, ldc, ldd, alpha, beta);
    auto cpu_end = std::chrono::high_resolution_clock::now();
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
    constexpr index_t block_size = 256;
    const index_t grid_size = (M / 64) * (N / 64);  // 64×64 per block

    std::cout << "Launching kernel:\n";
    std::cout << "  Grid: " << grid_size << " blocks\n";
    std::cout << "  Block: " << block_size << " threads (4 warps in 2×2 config)\n";
    std::cout << "  LDS size: " << SimpleLdsStagingHgemmKernel<InputType, InputType, AccumType, AccumType>::GetStaticLdsSize() << " bytes (~8KB)\n";
    std::cout << "  K-chunk: 32 elements (was 16), KIterPerWarp=2\n";
    std::cout << "  Each block: 64×64 output\n\n";

    stream_config stream;

    constexpr index_t lds_size = SimpleLdsStagingHgemmKernel<InputType, InputType, AccumType, AccumType>::GetStaticLdsSize();

    // Warmup
    for(int i = 0; i < 5; ++i) {
        launch_kernel(stream,
                     make_kernel<block_size>(
                         SimpleLdsStagingHgemmKernel<InputType, InputType, AccumType, AccumType>{},
                         dim3(grid_size),
                         dim3(block_size),
                         lds_size,  // LDS size in bytes!
                         static_cast<const InputType*>(d_a.GetDeviceBuffer()),
                         static_cast<const InputType*>(d_b.GetDeviceBuffer()),
                         static_cast<const AccumType*>(d_c.GetDeviceBuffer()),
                         static_cast<AccumType*>(d_d.GetDeviceBuffer()),
                         M, N, K, lda, ldb, ldc, ldd, alpha, beta));
    }
    hip_check_error(hipDeviceSynchronize());

    // Timed run
    auto gpu_start = std::chrono::high_resolution_clock::now();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     SimpleLdsStagingHgemmKernel<InputType, InputType, AccumType, AccumType>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const InputType*>(d_a.GetDeviceBuffer()),
                     static_cast<const InputType*>(d_b.GetDeviceBuffer()),
                     static_cast<const AccumType*>(d_c.GetDeviceBuffer()),
                     static_cast<AccumType*>(d_d.GetDeviceBuffer()),
                     M, N, K, lda, ldb, ldc, ldd, alpha, beta));

    hip_check_error(hipDeviceSynchronize());

    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    // Get result
    d_d.FromDevice(h_d.data(), M * N * sizeof(AccumType));

    // Verify
    bool passed = true;
    float max_error = 0;
    index_t error_count = 0;

    for(index_t i = 0; i < M * N; ++i) {
        float error = std::abs(h_d[i] - h_d_ref[i]);
        max_error = std::max(max_error, error);
        if(error > 1e-2f) {
            if(error_count < 5) {
                index_t m = i % M;
                index_t n = i / M;
                std::cout << "Error at [" << m << "," << n << "]: "
                          << h_d[i] << " vs " << h_d_ref[i]
                          << " (diff=" << error << ")\n";
            }
            error_count++;
        }
    }

    passed = (error_count == 0);

    double gflops = 2.0 * M * N * K / 1e9;
    double gpu_tflops = gflops / (gpu_time_ms / 1000);
    double cpu_gflops = gflops / (cpu_time_ms / 1000);

    std::cout << "Results:\n";
    std::cout << "  Correctness: " << (passed ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "  Max error: " << max_error << "\n";
    if(!passed) std::cout << "  Error count: " << error_count << "/" << M*N << "\n";
    std::cout << "\n";

    std::cout << "Performance:\n";
    std::cout << "  CPU time: " << cpu_time_ms << " ms (" << cpu_gflops << " GFLOPS)\n";
    std::cout << "  GPU time: " << gpu_time_ms << " ms (" << gpu_tflops << " TFLOPS)\n";
    std::cout << "  Speedup: " << cpu_time_ms / gpu_time_ms << "x\n\n";

    std::cout << "=== Key Insights ===\n";
    std::cout << "• Y-dimension repetition enables tile sweeping within distributions\n";
    std::cout << "• MIterPerWarp and NIterPerWarp control how many tiles each warp processes\n";
    std::cout << "• get_y_sliced_thread_data extracts specific tiles from block tensor\n";
    std::cout << "• static_for loops iterate over tile indices at compile time\n";
    std::cout << "• Replication still works: A replicates across NWarp, B across MWarp\n";
    std::cout << "• This pattern scales to production kernels (see 02_gemm)\n";
    std::cout << "• Each warp: 2×2 iters × 16×16 per tile = 32×32 output\n";
    std::cout << "• Each block: 2×2 warps × 32×32 per warp = 64×64 output\n\n";

    return passed ? 0 : 1;
}
