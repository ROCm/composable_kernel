// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Test A matrix distribution with Y-dimension repetition
 *
 * Goal: Load a 64x16 A matrix with 256 threads (4 warps in 2x2 config)
 * With MIterPerWarp=2, each warp should load 2 tiles of 16x16 in M dimension
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct TestADistributionYRepetitionKernel
{
    static constexpr index_t kBlockSize   = 256;
    static constexpr index_t MWarp        = 2;
    static constexpr index_t NWarp        = 2;
    static constexpr index_t MIterPerWarp = 2;
    static constexpr index_t KIterPerWarp = 1;
    static constexpr index_t kM           = 64; // 2 warps × 2 iters × 16
    static constexpr index_t kK           = 64;

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

        // Step 2: Block-level outer distribution (warp organization)
        // Must have same NDimX as inner (2 dimensions: M and K)
        // constexpr auto a_block_outer_dstr_encode = tile_distribution_encoding<
        //     sequence<NWarp>,                      // R: Replicate across N-warps
        //     tuple<sequence<MWarp>, sequence<>>,  // H: MWarp in M-dim, 1 in K-dim
        //     tuple<sequence<0, 1>>,                // Ps_to_Hs
        //     tuple<sequence<0, 0>>,                // Ps_in_Hs
        //     sequence<>,                       // Ys_to_Hs: Y maps to both M and K
        //     sequence<>>{};                    // Ys_in_Hs

        // A distribution with Y-repetition (from tutorial_07)
        constexpr auto a_warp_dstr_encode =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<16>, sequence<4, 4>>,
                                       tuple<sequence<2, 1>>,
                                       tuple<sequence<0, 0>>,
                                       sequence<2>,
                                       sequence<1>>{};

        constexpr auto a_block_outer_dstr_encode =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encode, a_warp_dstr_encode);

        constexpr auto a_distribution = make_static_tile_distribution(a_block_dstr_encode);

        auto a_window = make_tile_window(
            a_tensor, make_tuple(number<kM>{}, number<kK>{}), {0, 0}, a_distribution);

        const auto a_tile         = load_tile(a_window);
        const auto& thread_buffer = a_tile.get_thread_buffer();

        // Print from all warps sequentially
        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== A Distribution with Y-Repetition Test ===\n");
            printf("Matrix: 64×16 (MWarp=2, MIterPerWarp=2, each warp loads 2×16 tiles)\n");
            printf("Input: A[m,k] = m + k*100 (unique values)\n\n");
        }

        __syncthreads();

        for(int w = 0; w < 4; ++w)
        {
            __syncthreads();
            if(warp_id == w && lane_id == 0)
            {
                printf("Warp %d (M-warp %d, N-warp %d):\n", w, w / NWarp, w % NWarp);
                printf("  Thread buffer size: %d\n", static_cast<int>(thread_buffer.size()));
                printf("  Values: ");
                for(int i = 0; i < thread_buffer.size(); ++i)
                {
                    printf("%.0f ", static_cast<float>(thread_buffer[i]));
                }
                printf("\n");
            }
        }

        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== Expected Pattern ===\n");
            printf("Each warp should load 8 elements (2 M-iters × 1 K-iter × 4 warp elements)\n");
            printf("Warp 0 (M-warp 0): Should have M-rows [0-15] and [16-31]\n");
            printf("Warp 2 (M-warp 1): Should have M-rows [32-47] and [48-63]\n");
            printf("Warps 0&2 should be identical (NWarp replication)\n");
            printf("Warps 1&3 should be identical (NWarp replication)\n");
        }

        // Store for verification
        for(int i = 0; i < thread_buffer.size(); ++i)
        {
            debug_output[tid * 8 + i] = thread_buffer[i];
        }
    }
};

int main()
{
    std::cout << "\n==================================================\n";
    std::cout << "Test A Distribution with Y-Dimension Repetition\n";
    std::cout << "==================================================\n\n";

    constexpr index_t M   = 64;
    constexpr index_t K   = 64;
    constexpr index_t lda = M;

    using DataType = half_t;

    std::vector<DataType> h_a(M * K);
    std::vector<DataType> h_debug(256 * 8, -1);

    // Initialize A[m,k] = m + k*100 (unique for each position)
    for(index_t k = 0; k < K; ++k)
    {
        for(index_t m = 0; m < M; ++m)
        {
            h_a[m + k * lda] = static_cast<DataType>(m + k * 100);
        }
    }

    DeviceMem d_a(M * K * sizeof(DataType));
    DeviceMem d_debug(256 * 8 * sizeof(DataType));

    d_a.ToDevice(h_a.data(), M * K * sizeof(DataType));
    d_debug.ToDevice(h_debug.data(), 256 * 8 * sizeof(DataType));

    stream_config stream;
    launch_kernel(stream,
                  make_kernel<256>(TestADistributionYRepetitionKernel<DataType>{},
                                   dim3(1),
                                   dim3(256),
                                   0,
                                   static_cast<const DataType*>(d_a.GetDeviceBuffer()),
                                   static_cast<DataType*>(d_debug.GetDeviceBuffer()),
                                   lda));

    hip_check_error(hipDeviceSynchronize());

    d_debug.FromDevice(h_debug.data(), 256 * 8 * sizeof(DataType));

    std::cout << "\n✓ Test completed - check GPU output above\n";

    return 0;
}
