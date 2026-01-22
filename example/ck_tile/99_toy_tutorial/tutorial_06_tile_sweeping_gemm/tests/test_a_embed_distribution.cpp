// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Test A matrix distribution using EMBED API
 *
 * Goal: Load a 16x16 A matrix with 256 threads (4 warps in 2x2 config)
 * Uses detail::make_embed_tile_distribution_encoding to separate:
 * - Block-level: Warp organization with replication
 * - Warp-level: Thread organization within each warp
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct TestADistributionKernel
{
    static constexpr index_t kBlockSize = 256; // 4 warps
    static constexpr index_t MWarp      = 2;
    static constexpr index_t NWarp      = 2;
    static constexpr index_t kM         = 16;
    static constexpr index_t kK         = 16;

    CK_TILE_DEVICE void operator()(const DataType* a, DataType* debug_output, index_t lda) const
    {
        if(get_block_id() != 0)
            return;

        const index_t tid     = threadIdx.x;
        const index_t warp_id = tid / 64;
        const index_t lane_id = tid % 64;

        // Create tensor view for A (column-major)
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            a, make_tuple(kM, kK), make_tuple(1, lda), number<1>{}, number<1>{});

        // A distribution using EMBED API (like 02_gemm)
        // Separate block-level and warp-level distributions

        // constexpr auto a_distribution = make_static_tile_distribution(
        //     tile_distribution_encoding<
        //         sequence<NWarp>,                      // R: REPLICATE across 2 N-warps
        //         tuple<sequence<MWarp, 16>,            // H0 (M): 2 M-warps × 16 threads = 32 M
        //               sequence<4, 4>>,                // H1 (K): 4×4 = 16 K elements
        //         tuple<sequence<0, 1>, sequence<2, 1>>,  // Ps_to_Hs: P0→(R,M), P1→(M,K)
        //         tuple<sequence<0, 0>, sequence<0, 1>>,  // Ps_in_Hs: positions
        //         sequence<2>,                          // Ys_to_Hs: Y maps to K (dimension 2)
        //         sequence<1>>{}                        // Ys_in_Hs: Y at position 1 in K
        // );

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

        // Step 3: Embed warp-level into block-level
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encode, a_warp_dstr_encode);

        // Step 4: Create final distribution
        constexpr auto a_distribution = make_static_tile_distribution(a_block_dstr_encode);

        auto a_window = make_tile_window(
            a_tensor, make_tuple(number<kM>{}, number<kK>{}), {0, 0}, a_distribution);

        const auto a_tile         = load_tile(a_window);
        const auto& thread_buffer = a_tile.get_thread_buffer();

        // Calculate matrix coordinates using make_tensor_coordinate
        // This shows which matrix positions each thread accesses

        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== Matrix Coverage (Tiled by Warp) ===\n");
            printf("Distribution covers 32×16 matrix (MWarp×16 threads × K)\n");
            printf("With NWarp=2 replication, pattern repeats\n");
            printf("Showing first 16 threads of each warp:\n\n");
        }

        __syncthreads();

        // Print warp by warp with calculated coordinates
        for(int w = 0; w < 4; ++w)
        {
            __syncthreads();

            if(warp_id == w && lane_id == 0)
            {
                printf("\n--- Warp %d (M-warp %d, N-warp %d) ---\n", w, w / NWarp, w % NWarp);
            }

            __syncthreads();

            // Print lanes sequentially within each warp
            for(int lane = 0; lane < 16; ++lane)
            {
                __syncthreads();

                if(warp_id == w && lane_id == lane)
                {
                    printf("W%d L%02d: ", w, lane);

                    // For each Y element, just print the loaded value
                    // The distribution handles the coordinate mapping internally
                    for(int y_idx = 0; y_idx < thread_buffer.size(); ++y_idx)
                    {
                        float val = static_cast<float>(thread_buffer[y_idx]);
                        int m     = static_cast<int>(val) % 100;
                        int k     = static_cast<int>(val) / 100;

                        printf("A[%2d,%2d] ", m, k);
                    }
                    printf("\n");
                }
            }
        }

        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== Expected Pattern with NWarp Replication ===\n");
            printf("sequence<NWarp> replicates across N-warp dimension:\n");
            printf("Warp 0 (M-warp 0, N-warp 0): Loads some M-rows, K[0-15]\n");
            printf("Warp 1 (M-warp 0, N-warp 1): Loads DIFFERENT M-rows (different N-warp)\n");
            printf("Warp 2 (M-warp 1, N-warp 0): Loads SAME as Warp 0 (NWarp replication!)\n");
            printf("Warp 3 (M-warp 1, N-warp 1): Loads SAME as Warp 1 (NWarp replication!)\n");
            printf("\nReplication pairs:\n");
            printf(
                "  Warps 0 & 2 should be identical (same N-warp 0, replicated across M-warps)\n");
            printf(
                "  Warps 1 & 3 should be identical (same N-warp 1, replicated across M-warps)\n");
            printf("  Warps 0 & 1 should be DIFFERENT (different N-warps)\n");
        }

        // Store for verification
        for(int i = 0; i < thread_buffer.size(); ++i)
        {
            debug_output[tid * 4 + i] = thread_buffer[i];
        }
    }
};

int main()
{
    std::cout << "\n==================================================\n";
    std::cout << "Test A Distribution using EMBED API\n";
    std::cout << "==================================================\n\n";
    std::cout
        << "Separates block-level (warp organization) from warp-level (thread organization)\n\n";

    constexpr index_t M   = 16;
    constexpr index_t K   = 16;
    constexpr index_t lda = M;

    using DataType = half_t;

    // Create test matrix
    std::vector<DataType> h_a(M * K);
    std::vector<DataType> h_debug(256 * 4, -1);

    // Initialize A[m,k] = m + k*100
    for(index_t k = 0; k < K; ++k)
    {
        for(index_t m = 0; m < M; ++m)
        {
            h_a[m + k * lda] = static_cast<DataType>(m + k * 100);
        }
    }

    DeviceMem d_a(M * K * sizeof(DataType));
    DeviceMem d_debug(256 * 4 * sizeof(DataType));

    d_a.ToDevice(h_a.data(), M * K * sizeof(DataType));
    d_debug.ToDevice(h_debug.data(), 256 * 4 * sizeof(DataType));

    stream_config stream;
    launch_kernel(stream,
                  make_kernel<256>(TestADistributionKernel<DataType>{},
                                   dim3(1),
                                   dim3(256),
                                   0,
                                   static_cast<const DataType*>(d_a.GetDeviceBuffer()),
                                   static_cast<DataType*>(d_debug.GetDeviceBuffer()),
                                   lda));

    hip_check_error(hipDeviceSynchronize());

    d_debug.FromDevice(h_debug.data(), 256 * 4 * sizeof(DataType));

    // Verify NWarp replication: warps 0&2 identical, warps 1&3 identical
    bool passed = true;

    // Check warps 0 and 2 (same N-warp 0, replicated across M-warps)
    for(int lane = 0; lane < 64; ++lane)
    {
        for(int i = 0; i < 4; ++i)
        {
            float warp0_val = h_debug[lane * 4 + i];
            float warp2_val = h_debug[(128 + lane) * 4 + i];
            if(std::abs(warp0_val - warp2_val) > 0.01f)
            {
                std::cout << "ERROR: Warp 0 and Warp 2 differ at lane " << lane << " element " << i
                          << "\n";
                std::cout << "  Warp 0: " << warp0_val << ", Warp 2: " << warp2_val << "\n";
                passed = false;
                break;
            }
        }
        if(!passed)
            break;
    }

    // Check warps 1 and 3 (same N-warp 1, replicated across M-warps)
    if(passed)
    {
        for(int lane = 0; lane < 64; ++lane)
        {
            for(int i = 0; i < 4; ++i)
            {
                float warp1_val = h_debug[(64 + lane) * 4 + i];
                float warp3_val = h_debug[(192 + lane) * 4 + i];
                if(std::abs(warp1_val - warp3_val) > 0.01f)
                {
                    std::cout << "ERROR: Warp 1 and Warp 3 differ at lane " << lane << " element "
                              << i << "\n";
                    std::cout << "  Warp 1: " << warp1_val << ", Warp 3: " << warp3_val << "\n";
                    passed = false;
                    break;
                }
            }
            if(!passed)
                break;
        }
    }

    if(passed)
    {
        std::cout << "\n✓ NWarp Replication verified:\n";
        std::cout << "  Warps 0 & 2 load identical data (N-warp 0, replicated across M-warps)\n";
        std::cout << "  Warps 1 & 3 load identical data (N-warp 1, replicated across M-warps)\n";
    }

    return passed ? 0 : 1;
}
