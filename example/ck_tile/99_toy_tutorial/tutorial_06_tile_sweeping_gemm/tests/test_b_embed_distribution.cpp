// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Test B matrix distribution using EMBED API
 *
 * Goal: Load a 16x16 B matrix with 256 threads (4 warps in 2x2 config)
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
struct TestBDistributionKernel
{
    static constexpr index_t kBlockSize = 256; // 4 warps
    static constexpr index_t MWarp      = 2;
    static constexpr index_t NWarp      = 2;
    static constexpr index_t kK         = 16;
    static constexpr index_t kN         = 16;

    CK_TILE_DEVICE void operator()(const DataType* b, DataType* debug_output, index_t ldb) const
    {
        if(get_block_id() != 0)
            return;

        const index_t tid     = threadIdx.x;
        const index_t warp_id = tid / 64;
        const index_t lane_id = tid % 64;

        // Create tensor view for B (row-major)
        const auto b_tensor = make_naive_tensor_view<address_space_enum::global>(
            b, make_tuple(kK, kN), make_tuple(ldb, 1), number<4>{}, number<1>{});

        // B distribution using EMBED API (like 02_gemm)
        // Separate block-level and warp-level distributions

        // Step 1: Warp-level distribution (64 threads within one warp)
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

        // constexpr auto b_distribution = make_static_tile_distribution(
        //     tile_distribution_encoding<
        //         sequence<MWarp>,                      // R: dimension 0, REPLICATE across 2
        //         M-warps tuple<sequence<4, 4>,                 // H: dimension 1 (K): 4×4 = 16 K
        //         elements
        //               sequence<2, 16>>,                  // H: dimension 2 (N): 16 N positions
        //         tuple<sequence<2, 0>, sequence<1, 2>>,  // Ps_to_Hs: P0→R(dim 0), P1→K(dim 1),
        //         P2→N(dim 2) tuple<sequence<0, 0>, sequence<0, 1>>,  // Ps_in_Hs: positions
        //         sequence<1>,                          // Ys_to_Hs: Y maps to K (dimension 1)
        //         sequence<1>>{}                        // Ys_in_Hs: Y at position 1 in K
        // );

        // Step 3: Embed warp-level into block-level
        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encode, b_warp_dstr_encode);

        // Step 4: Create final distribution
        constexpr auto b_distribution = make_static_tile_distribution(b_block_dstr_encode);

        auto b_window = make_tile_window(
            b_tensor, make_tuple(number<kK>{}, number<kN>{}), {0, 0}, b_distribution);

        const auto b_tile         = load_tile(b_window);
        const auto& thread_buffer = b_tile.get_thread_buffer();

        // Sequential printing with synchronizations (copied from test_a)
        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== Matrix Coverage (Tiled by Warp) ===\n");
            printf("Distribution covers K×32 matrix (K × NWarp×16 threads)\n");
            printf("With MWarp=2 replication, pattern repeats\n");
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

                    // Print loaded values
                    for(int y_idx = 0; y_idx < thread_buffer.size(); ++y_idx)
                    {
                        float val = static_cast<float>(thread_buffer[y_idx]);
                        int k     = static_cast<int>(val) % 100;
                        int n     = static_cast<int>(val) / 100;

                        printf("B[%2d,%2d] ", k, n);
                    }
                    printf("\n");
                }
            }
        }

        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== Observed Pattern ===\n");
            printf("Warps 0 & 1 are identical\n");
            printf("Warps 2 & 3 are identical\n");
            printf("Warps 0 & 2 are different\n");
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
    std::cout << "Test B Distribution using EMBED API\n";
    std::cout << "==================================================\n\n";
    std::cout
        << "Separates block-level (warp organization) from warp-level (thread organization)\n\n";

    constexpr index_t K   = 32;
    constexpr index_t N   = 32;
    constexpr index_t ldb = N;

    using DataType = half_t;

    // Create test matrix
    std::vector<DataType> h_b(K * N);
    std::vector<DataType> h_debug(256 * 4, -1);

    // Initialize B[k,n] = k + n*100 (row-major)
    for(index_t k = 0; k < K; ++k)
    {
        for(index_t n = 0; n < N; ++n)
        {
            h_b[k * ldb + n] = static_cast<DataType>(k + n * 100);
        }
    }

    DeviceMem d_b(K * N * sizeof(DataType));
    DeviceMem d_debug(256 * 4 * sizeof(DataType));

    d_b.ToDevice(h_b.data(), K * N * sizeof(DataType));
    d_debug.ToDevice(h_debug.data(), 256 * 4 * sizeof(DataType));

    stream_config stream;
    launch_kernel(stream,
                  make_kernel<256>(TestBDistributionKernel<DataType>{},
                                   dim3(1),
                                   dim3(256),
                                   0,
                                   static_cast<const DataType*>(d_b.GetDeviceBuffer()),
                                   static_cast<DataType*>(d_debug.GetDeviceBuffer()),
                                   ldb));

    hip_check_error(hipDeviceSynchronize());

    d_debug.FromDevice(h_debug.data(), 256 * 4 * sizeof(DataType));

    // Verify: Based on your observation, warps 0&1 identical, warps 2&3 identical
    bool passed = true;

    // Check warps 0 and 1
    for(int lane = 0; lane < 64; ++lane)
    {
        for(int i = 0; i < 4; ++i)
        {
            float warp0_val = h_debug[lane * 4 + i];
            float warp1_val = h_debug[(64 + lane) * 4 + i];
            if(std::abs(warp0_val - warp1_val) > 0.01f)
            {
                std::cout << "ERROR: Warp 0 and Warp 1 differ at lane " << lane << " element " << i
                          << "\n";
                std::cout << "  Warp 0: " << warp0_val << ", Warp 1: " << warp1_val << "\n";
                passed = false;
                break;
            }
        }
        if(!passed)
            break;
    }

    // Check warps 2 and 3
    if(passed)
    {
        for(int lane = 0; lane < 64; ++lane)
        {
            for(int i = 0; i < 4; ++i)
            {
                float warp2_val = h_debug[(128 + lane) * 4 + i];
                float warp3_val = h_debug[(192 + lane) * 4 + i];
                if(std::abs(warp2_val - warp3_val) > 0.01f)
                {
                    std::cout << "ERROR: Warp 2 and Warp 3 differ at lane " << lane << " element "
                              << i << "\n";
                    std::cout << "  Warp 2: " << warp2_val << ", Warp 3: " << warp3_val << "\n";
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
        std::cout << "\n✓ Replication verified:\n";
        std::cout << "  Warps 0 & 1 load identical data\n";
        std::cout << "  Warps 2 & 3 load identical data\n";
    }

    return passed ? 0 : 1;
}
