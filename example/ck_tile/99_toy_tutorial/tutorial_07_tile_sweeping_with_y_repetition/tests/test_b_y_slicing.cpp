// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Test B matrix Y-slicing with get_y_sliced_thread_data
 *
 * This test verifies that Y-dimension slicing works correctly by:
 * 1. Loading the full block tile (16×64)
 * 2. Using get_y_sliced_thread_data to extract individual iteration tiles
 * 3. Printing what each warp gets for each iteration
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct TestBYSlicingKernel
{
    static constexpr index_t kBlockSize   = 256;
    static constexpr index_t MWarp        = 2;
    static constexpr index_t NWarp        = 2;
    static constexpr index_t NIterPerWarp = 2;
    static constexpr index_t KIterPerWarp = 1;
    static constexpr index_t kK           = 16; // Fixed to match distribution coverage
    static constexpr index_t kN           = 64; // 2 warps × 2 iters × 16

    CK_TILE_DEVICE void operator()(const DataType* b, index_t ldb) const
    {
        if(get_block_id() != 0)
            return;

        const index_t tid     = threadIdx.x;
        const index_t warp_id = tid / 64;
        const index_t lane_id = tid % 64;

        // Create tensor view for B (row-major K×N)
        const auto b_tensor = make_naive_tensor_view<address_space_enum::global>(
            b, make_tuple(kK, kN), make_tuple(ldb, 1), number<4>{}, number<1>{});

        // B warp-level distribution
        constexpr auto b_warp_dstr_encode =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<4, 4>, sequence<16>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<0, 0>>,
                                       sequence<1>,
                                       sequence<1>>{};

        // B block-level outer distribution with Y-repetition
        constexpr auto b_block_outer_dstr_encode =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<KIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
                                       tuple<sequence<2, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encode, b_warp_dstr_encode);

        constexpr auto b_block_distribution = make_static_tile_distribution(b_block_dstr_encode);

        // Get Y-dimension information
        using BWarpDstr = decltype(make_static_tile_distribution(b_warp_dstr_encode));
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};

        // Create window and load full block tile
        auto b_window = make_tile_window(
            b_tensor, make_tuple(number<kK>{}, number<kN>{}), {0, 0}, b_block_distribution);

        const auto b_block_tile = load_tile(b_window);

        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== B Y-Slicing Test ===\n");
            printf("Block tile: %d×%d (K×N)\n", kK, kN);
            printf("NIterPerWarp=%d, KIterPerWarp=%d\n", NIterPerWarp, KIterPerWarp);
            printf("Input: B[k,n] = k*1000 + n (unique values)\n\n");
        }

        __syncthreads();

        // Test Y-slicing for each warp and iteration
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                // Extract warp tensor for this iteration
                auto b_warp_tensor = make_static_distributed_tensor<DataType>(
                    make_static_tile_distribution(b_warp_dstr_encode));

                b_warp_tensor.get_thread_buffer() = b_block_tile.get_y_sliced_thread_data(
                    merge_sequences(sequence<kIter, nIter>{}, b_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                const auto& warp_buffer = b_warp_tensor.get_thread_buffer();

                // Print from each warp sequentially
                for(int w = 0; w < 4; ++w)
                {
                    __syncthreads();
                    if(warp_id == w && lane_id == 0)
                    {
                        printf("Warp %d (M-warp %d, N-warp %d), NIter=%d, KIter=%d:\n",
                               w,
                               w / NWarp,
                               w % NWarp,
                               static_cast<int>(nIter),
                               static_cast<int>(kIter));
                        printf("  Buffer size: %d\n", static_cast<int>(warp_buffer.size()));
                        printf("  Values: ");
                        for(int i = 0; i < warp_buffer.size() && i < 16; ++i)
                        {
                            printf("%.0f ", static_cast<float>(warp_buffer[i]));
                        }
                        if(warp_buffer.size() > 16)
                            printf("...");
                        printf("\n");
                    }
                }
            });
        });

        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== Expected Pattern ===\n");
            printf("Each warp should get 4 elements per iteration (16 K × 16 N / 64 threads)\n");
            printf("Warp 0, NIter=0: Should have values from B[0:16, 0:16]\n");
            printf("Warp 0, NIter=1: Should have values from B[0:16, 16:32]\n");
            printf("Warp 1, NIter=0: Should have values from B[0:16, 32:48]\n");
            printf("Warp 1, NIter=1: Should have values from B[0:16, 48:64]\n");
            printf("Warps 2&3 should REPLICATE warps 0&1 (MWarp replication)\n");
        }
    }
};

int main()
{
    std::cout << "\n==================================================\n";
    std::cout << "Test B Y-Slicing with get_y_sliced_thread_data\n";
    std::cout << "==================================================\n\n";

    constexpr index_t K   = 16; // Match distribution coverage
    constexpr index_t N   = 64;
    constexpr index_t ldb = N;

    using DataType = half_t;

    std::vector<DataType> h_b(K * N);

    // Initialize B[k,n] = k*1000 + n (easy to identify position)
    auto counter = 0;
    for(index_t k = 0; k < K; ++k)
    {
        for(index_t n = 0; n < N; ++n)
        {
            h_b[k * ldb + n] = static_cast<DataType>(counter++);
        }
    }

    std::cout << "Matrix B (K×N = " << K << "×" << N << "):\n";
    std::cout << "Sample values:\n";
    std::cout << "  B[0,0] = " << static_cast<float>(h_b[0]) << "\n";
    std::cout << "  B[0,16] = " << static_cast<float>(h_b[16]) << "\n";
    std::cout << "  B[0,32] = " << static_cast<float>(h_b[32]) << "\n";
    std::cout << "  B[0,48] = " << static_cast<float>(h_b[48]) << "\n";
    std::cout << "  B[15,0] = " << static_cast<float>(h_b[15 * ldb]) << "\n\n";

    DeviceMem d_b(K * N * sizeof(DataType));
    d_b.ToDevice(h_b.data(), K * N * sizeof(DataType));

    stream_config stream;
    launch_kernel(stream,
                  make_kernel<256>(TestBYSlicingKernel<DataType>{},
                                   dim3(1),
                                   dim3(256),
                                   0,
                                   static_cast<const DataType*>(d_b.GetDeviceBuffer()),
                                   ldb));

    hip_check_error(hipDeviceSynchronize());

    std::cout << "\n✓ Test completed - check GPU output above\n";
    std::cout << "\nIf Y-slicing works correctly, you should see:\n";
    std::cout << "- Each warp gets different N-column ranges for different iterations\n";
    std::cout << "- Warps 0&2 should have identical values (MWarp replication)\n";
    std::cout << "- Warps 1&3 should have identical values (MWarp replication)\n";

    return 0;
}
