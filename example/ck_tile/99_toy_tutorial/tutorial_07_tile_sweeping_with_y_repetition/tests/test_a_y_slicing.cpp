// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Test A matrix Y-slicing with get_y_sliced_thread_data
 *
 * This test verifies that Y-dimension slicing works correctly for A by:
 * 1. Loading the full block tile (64×16)
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
struct TestAYSlicingKernel
{
    static constexpr index_t kBlockSize   = 256;
    static constexpr index_t MWarp        = 2;
    static constexpr index_t NWarp        = 2;
    static constexpr index_t MIterPerWarp = 2;
    static constexpr index_t KIterPerWarp = 1;
    static constexpr index_t kM           = 64; // 2 warps × 2 iters × 16
    static constexpr index_t kK           = 16; // Fixed to match distribution coverage

    CK_TILE_DEVICE void operator()(const DataType* a, index_t lda) const
    {
        if(get_block_id() != 0)
            return;

        const index_t tid     = threadIdx.x;
        const index_t warp_id = tid / 64;
        const index_t lane_id = tid % 64;

        // Create tensor view for A (column-major M×K)
        const auto a_tensor = make_naive_tensor_view<address_space_enum::global>(
            a, make_tuple(kM, kK), make_tuple(1, lda), number<1>{}, number<1>{});

        // A warp-level distribution
        constexpr auto a_warp_dstr_encode =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<16>, sequence<4, 4>>,
                                       tuple<sequence<2, 1>>,
                                       tuple<sequence<0, 0>>,
                                       sequence<2>,
                                       sequence<1>>{};

        // A block-level outer distribution with Y-repetition
        constexpr auto a_block_outer_dstr_encode =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encode, a_warp_dstr_encode);

        constexpr auto a_block_distribution = make_static_tile_distribution(a_block_dstr_encode);

        // Get Y-dimension information
        using AWarpDstr = decltype(make_static_tile_distribution(a_warp_dstr_encode));
        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};

        // Create window and load full block tile
        auto a_window = make_tile_window(
            a_tensor, make_tuple(number<kM>{}, number<kK>{}), {0, 0}, a_block_distribution);

        const auto a_block_tile = load_tile(a_window);

        __syncthreads();

        if(tid == 0)
        {
            printf("\n=== A Y-Slicing Test ===\n");
            printf("Block tile: %d×%d (M×K)\n", kM, kK);
            printf("MIterPerWarp=%d, KIterPerWarp=%d\n", MIterPerWarp, KIterPerWarp);
            printf("Input: A[m,k] = m*1000 + k (unique values)\n\n");
        }

        __syncthreads();

        // Test Y-slicing for each warp and iteration
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // Extract warp tensor for this iteration
                auto a_warp_tensor = make_static_distributed_tensor<DataType>(
                    make_static_tile_distribution(a_warp_dstr_encode));

                // CORRECTED: kIter first, then mIter (matching the Ys_to_Hs order)
                a_warp_tensor.get_thread_buffer() = a_block_tile.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                const auto& warp_buffer = a_warp_tensor.get_thread_buffer();

                // Print from each warp sequentially
                for(int w = 0; w < 4; ++w)
                {
                    __syncthreads();
                    if(warp_id == w && lane_id == 0)
                    {
                        printf("Warp %d (M-warp %d, N-warp %d), MIter=%d, KIter=%d:\n",
                               w,
                               w / NWarp,
                               w % NWarp,
                               static_cast<int>(mIter),
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
            printf("Each warp should get 4 elements per iteration (16 M × 16 K / 64 threads)\n");
            printf("Warp 0, MIter=0: Should have values from A[0:16, 0:16]\n");
            printf("Warp 0, MIter=1: Should have values from A[16:32, 0:16]\n");
            printf("Warp 2, MIter=0: Should have values from A[32:48, 0:16]\n");
            printf("Warp 2, MIter=1: Should have values from A[48:64, 0:16]\n");
            printf("Warps 0&1 should REPLICATE (NWarp replication)\n");
            printf("Warps 2&3 should REPLICATE (NWarp replication)\n");
        }
    }
};

int main()
{
    std::cout << "\n==================================================\n";
    std::cout << "Test A Y-Slicing with get_y_sliced_thread_data\n";
    std::cout << "==================================================\n\n";

    constexpr index_t M   = 64; // 2 warps × 2 iters × 16
    constexpr index_t K   = 16; // Match distribution coverage
    constexpr index_t lda = M;

    using DataType = half_t;

    std::vector<DataType> h_a(M * K);

    // Initialize A[m,k] = m*1000 + k (easy to identify position)
    auto counter = 0;
    for(index_t m = 0; m < M; ++m)
    {
        for(index_t k = 0; k < K; ++k)
        {
            h_a[m + k * lda] = static_cast<DataType>(counter++);
        }
    }

    std::cout << "Matrix A (M×K = " << M << "×" << K << "):\n";
    std::cout << "Sample values:\n";
    std::cout << "  A[0,0] = " << static_cast<float>(h_a[0]) << "\n";
    std::cout << "  A[16,0] = " << static_cast<float>(h_a[16]) << "\n";
    std::cout << "  A[32,0] = " << static_cast<float>(h_a[32]) << "\n";
    std::cout << "  A[48,0] = " << static_cast<float>(h_a[48]) << "\n";
    std::cout << "  A[0,15] = " << static_cast<float>(h_a[15 * lda]) << "\n\n";

    DeviceMem d_a(M * K * sizeof(DataType));
    d_a.ToDevice(h_a.data(), M * K * sizeof(DataType));

    stream_config stream;
    launch_kernel(stream,
                  make_kernel<256>(TestAYSlicingKernel<DataType>{},
                                   dim3(1),
                                   dim3(256),
                                   0,
                                   static_cast<const DataType*>(d_a.GetDeviceBuffer()),
                                   lda));

    hip_check_error(hipDeviceSynchronize());

    std::cout << "\n✓ Test completed - check GPU output above\n";
    std::cout << "\nIf Y-slicing works correctly, you should see:\n";
    std::cout << "- Each warp gets different M-row ranges for different iterations\n";
    std::cout << "- Warps 0&1 should have identical values (NWarp replication)\n";
    std::cout << "- Warps 2&3 should have identical values (NWarp replication)\n";

    return 0;
}
