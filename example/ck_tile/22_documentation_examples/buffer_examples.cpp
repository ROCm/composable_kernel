// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include <cstring>
#include <iostream>
#include <vector>

namespace ck_tile {

struct TileDistributionExample
{
    CK_TILE_DEVICE void operator()(float* global_data,
                                   ck_tile::index_t global_shape_0,
                                   ck_tile::index_t global_shape_1) const
    {
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n=== Tile Distribution Example (Device Kernel) ===\n");
        }
        block_sync_lds();

        // Create a tile distribution encoding
        // This defines how a tensor is distributed across threads
        auto encoding = tile_distribution_encoding<
            sequence<>,           // rs_lengths=[] - No replication dimensions
            tuple<sequence<2, 2>, // hs_lengthss=[[2, 2], [2, 2]] - Hierarchical lengths for each X
                                  // dimension
                  sequence<2, 2>>,
            tuple<sequence<1>, sequence<2>>, // ps_to_rhss_major=[[1], [2]] - P to RH major mappings
            tuple<sequence<0>, sequence<0>>, // ps_to_rhss_minor=[[0], [0]] - P to RH minor mappings
            sequence<1, 2>,                  // ys_to_rhs_major=[1, 2] - Y to RH major mappings
            sequence<1, 1>>{};               // ys_to_rhs_minor=[1, 1] - Y to RH minor mappings

        // Create the tile distribution from the encoding
        auto distribution = make_static_tile_distribution(encoding);

        // Calculate sizes from the distribution encoding
        // x0_size = np.prod(distribution.encoding.hs_lengthss[0])
        constexpr auto hs_lengths_0 = encoding.hs_lengthss_[number<0>{}]; // sequence<2, 2>
        constexpr auto hs_lengths_1 = encoding.hs_lengthss_[number<1>{}]; // sequence<2, 2>

        constexpr index_t x0_size = reduce_on_sequence(hs_lengths_0, multiplies{}, number<1>{});
        constexpr index_t x1_size = reduce_on_sequence(hs_lengths_1, multiplies{}, number<1>{});

        // Print distribution info (only from thread 0)
        if(threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("\n- Tile distribution created:\n");
            printf("  X dimensions: %d\n", distribution.get_num_of_dimension_x());
            printf("  Y dimensions: %d\n", distribution.get_num_of_dimension_y());
            printf("  P dimensions: %d\n", distribution.get_num_of_dimension_p());
            printf("  X lengths: [%d, %d]\n", x0_size, x1_size);
        }
        block_sync_lds();

        // Create packed tensor view (contiguous row-major) using helper
        auto global_view = make_naive_tensor_view_packed<address_space_enum::global>(
            global_data, make_tuple(global_shape_0, global_shape_1));

        // Window configuration
        auto window_lengths = make_tuple(x0_size, x1_size);

        // Get current thread's warp and thread indices
        index_t warp_id   = threadIdx.x / get_warp_size();
        index_t thread_id = threadIdx.x % get_warp_size();

        // Window origin - small offset from origin
        auto window_origin = make_tuple(1, 3); // Small offset from origin

        // Create tile window
        auto tile_window = make_tile_window(global_view,
                                            window_lengths,
                                            {1, 3}, // Window origin as initializer list
                                            distribution);

        // Load distributed tensor
        auto distributed_tensor = tile_window.load();

        // Collect values by sweeping through the distributed tensor
        constexpr index_t max_elements = x0_size * x1_size;
        float collected_values[max_elements];
        index_t value_count = 0;

        // Sweep through the distributed tensor and collect values using sweep_tile API
        sweep_tile(distributed_tensor, [&](auto idx) {
            if(value_count < max_elements)
            {
                collected_values[value_count] = distributed_tensor(idx);
                value_count++;
            }
        });

        // Serialize printing in a fixed order for selected threads only.
        static constexpr int print_thread_ids[] = {0, 1, 64, 65};
        for(int sel : print_thread_ids)
        {
            block_sync_lds();
            if(static_cast<int>(threadIdx.x) == sel)
            {
                printf("Partition index: (warp=%d, thread=%d)\n",
                       static_cast<int>(warp_id),
                       static_cast<int>(thread_id));
                printf("Collected values: ");
                for(index_t i = 0; i < value_count; i++)
                {
                    printf("%.0f", collected_values[i]);
                    if(i < value_count - 1)
                        printf(", ");
                }
                printf("\n\n");
            }
            block_sync_lds();
        }
    }
};
} // namespace ck_tile

int main()
{
    // Host-side allocation & initialization of pattern data
    // Reproduce the compile-time sizes used in the kernel: hs_lengths = [2,2] => x sizes=4; global
    // = 4+5 = 9
    constexpr ck_tile::index_t global_shape_0 = 9;                               // x0_size(4) + 5
    constexpr ck_tile::index_t global_shape_1 = 9;                               // x1_size(4) + 5
    constexpr ck_tile::index_t total_elems    = global_shape_0 * global_shape_1; // 81

    std::vector<float> h_global_data(total_elems);
    for(ck_tile::index_t i = 0; i < global_shape_0; ++i)
    {
        for(ck_tile::index_t j = 0; j < global_shape_1; ++j)
        {
            h_global_data[i * global_shape_1 + j] = static_cast<float>(i * 100 + j);
        }
    }

    ck_tile::DeviceMem d_global_data(sizeof(float) * total_elems);
    d_global_data.ToDevice(h_global_data.data());

    std::cout << "\nGlobal data (host print, to be used by device) shape=("
              << static_cast<int>(global_shape_0) << "," << static_cast<int>(global_shape_1)
              << ")\n\n";
    for(ck_tile::index_t i = 0; i < global_shape_0; ++i)
    {
        for(ck_tile::index_t j = 0; j < global_shape_1; ++j)
        {
            std::cout << h_global_data[i * global_shape_1 + j];
            if(j + 1 < global_shape_1)
                std::cout << "\t";
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    constexpr ck_tile::index_t kBlockSize  = 128;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    constexpr ck_tile::index_t kGridSize   = 1;

    using Kernel   = ck_tile::TileDistributionExample;
    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<float*>(d_global_data.GetDeviceBuffer()),
                                       global_shape_0,
                                       global_shape_1));

    std::cout << "Kernel execution completed. Average time: " << ave_time << " ms" << std::endl;

    return 0;
}
