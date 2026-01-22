// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Test B matrix distribution with Y-dimension repetition
 *
 * Goal: Load a 16x64 B matrix with 256 threads (4 warps in 2x2 config)
 * With NIterPerWarp=2, each warp should load 2 tiles of 16x16 in N dimension
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct TestBDistributionYRepetitionKernel
{
    static constexpr index_t kBlockSize   = 256;
    static constexpr index_t MWarp        = 2;
    static constexpr index_t NWarp        = 2;
    static constexpr index_t NIterPerWarp = 2;
    static constexpr index_t KIterPerWarp = 1;
    static constexpr index_t kK           = 64;
    static constexpr index_t kN           = 64; // 2 warps × 2 iters × 16

    CK_TILE_DEVICE void operator()(const DataType* b, DataType* debug_output, index_t ldb) const
    {
        if(get_block_id() != 0)
            return;

        // each warp is 64 x 4 items and 4 warps total and 2 iteration, so totally it becomes 64 x
        // 32 we don't cover the whole matrix
        const index_t tid     = threadIdx.x;
        const index_t warp_id = tid / 64;
        const index_t lane_id = tid % 64;

        // Create tensor view for B (row-major)
        const auto b_tensor = make_naive_tensor_view<address_space_enum::global>(
            b, make_tuple(kK, kN), make_tuple(ldb, 1), number<4>{}, number<1>{});

        // B distribution with Y-repetition (from tutorial_07)
        constexpr auto b_warp_dstr_encode =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<4, 4>, sequence<16>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<0, 0>>,
                                       sequence<1>,
                                       sequence<1>>{};

        constexpr auto b_block_outer_dstr_encode =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<KIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
                                       tuple<sequence<2, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encode, b_warp_dstr_encode);

        constexpr auto b_distribution = make_static_tile_distribution(b_block_dstr_encode);

        auto b_window = make_tile_window(
            b_tensor, make_tuple(number<kK>{}, number<kN>{}), {0, 0}, b_distribution);

        const auto b_tile         = load_tile(b_window);
        const auto& thread_buffer = b_tile.get_thread_buffer();

        // Print from all warps sequentially
        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== B Distribution with Y-Repetition Test ===\n");
            printf("Matrix: 16×64 (NWarp=2, NIterPerWarp=2, each warp loads 2×16 tiles)\n");
            printf("Input: B[k,n] = k + n*100 (unique values)\n\n");
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
            printf("Each warp should load 8 elements (2 N-iters × 1 K-iter × 4 warp elements)\n");
            printf("Warp 0 (N-warp 0): Should have N-cols [0-15] and [16-31]\n");
            printf("Warp 1 (N-warp 1): Should have N-cols [32-47] and [48-63]\n");
            printf("Warps 0&1 should be identical (MWarp replication)\n");
            printf("Warps 2&3 should be identical (MWarp replication)\n");
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
    std::cout << "Test B Distribution with Y-Dimension Repetition\n";
    std::cout << "==================================================\n\n";

    constexpr index_t K   = 64;
    constexpr index_t N   = 64;
    constexpr index_t ldb = N;

    using DataType = half_t;

    std::vector<DataType> h_b(K * N);
    std::vector<DataType> h_debug(256 * 8, -1);

    // Initialize B[k,n] = k + n*100 (unique for each position)
    auto counter = 0;
    for(index_t k = 0; k < K; ++k)
    {
        for(index_t n = 0; n < N; ++n)
        {
            h_b[k * ldb + n] = static_cast<DataType>(counter++);
        }
    }

    DeviceMem d_b(K * N * sizeof(DataType));
    DeviceMem d_debug(256 * 8 * sizeof(DataType));

    d_b.ToDevice(h_b.data(), K * N * sizeof(DataType));
    d_debug.ToDevice(h_debug.data(), 256 * 8 * sizeof(DataType));

    stream_config stream;
    launch_kernel(stream,
                  make_kernel<256>(TestBDistributionYRepetitionKernel<DataType>{},
                                   dim3(1),
                                   dim3(256),
                                   0,
                                   static_cast<const DataType*>(d_b.GetDeviceBuffer()),
                                   static_cast<DataType*>(d_debug.GetDeviceBuffer()),
                                   ldb));

    hip_check_error(hipDeviceSynchronize());

    d_debug.FromDevice(h_debug.data(), 256 * 8 * sizeof(DataType));

    std::cout << "\n✓ Test completed - check GPU output above\n";

    return 0;
}
