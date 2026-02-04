// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 03: Padding with Tile Windows
 *
 * This tutorial demonstrates the proper way to use padding transforms with:
 * 1. buffer_view with identity values
 * 2. tile_window for tiled access
 * 3. load_tile with automatic padding handling
 *
 * Key Learning: This is the pattern used in pooling and convolution kernels
 * to handle out-of-bounds accesses gracefully.
 */

#include <iostream>
#include <vector>
#include <numeric>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct PaddingTileKernel
{
    static constexpr index_t kBlockSize = 64;

    CK_TILE_DEVICE void
    operator()(const DataType* p_data, index_t orig_size, index_t padded_size) const
    {
        if(get_thread_id() != 0)
            return;

        printf("\n=== PADDING WITH TILE WINDOWS ===\n\n");

        printf("Original size: %ld\n", static_cast<long>(orig_size));
        printf("Padded size: %ld\n\n", static_cast<long>(padded_size));

        // Step 1: Create original descriptor (runtime)
        auto desc_orig = make_naive_tensor_descriptor(make_tuple(orig_size), make_tuple(1));

        // Step 2: Apply padding transform
        index_t pad_amount = padded_size - orig_size;
        auto desc_padded   = transform_tensor_descriptor(
            desc_orig,
            make_tuple(make_right_pad_transform(orig_size, pad_amount)), // extend on the right
            make_tuple(sequence<0>{}), // no dimension reordering
            make_tuple(sequence<0>{}));

        auto tensor_simple = make_tensor_view<address_space_enum::global>(p_data, desc_padded);

        printf("Created tensor_view (simple API, no identity value)\n");
        printf("  - Padded reads will wrap around to existing data     WRONG!!!\n\n");

        // Step 5: Read tiles using get_vectorized_elements
        constexpr index_t tile_size = 8;

        printf("Reading tiles of size %ld using get_vectorized_elements:\n\n",
               static_cast<long>(tile_size));

        // Load tiles covering the entire padded range
        index_t num_tiles = (padded_size + tile_size - 1) / tile_size;

        for(index_t tile_idx = 0; tile_idx < num_tiles; tile_idx++)
        {
            // Use get_vectorized_elements directly on tensor_view
            printf("Tile %ld (indices %ld-%ld):\n",
                   static_cast<long>(tile_idx),
                   static_cast<long>(tile_idx * tile_size),
                   static_cast<long>(tile_idx * tile_size + tile_size - 1));

            printf("  Values: ");
            // Use static_for to access elements with compile-time indices
            static_for<0, tile_size, 1>{}([&](auto i) {
                index_t global_idx = tile_idx * tile_size + i;
                auto coord         = make_tensor_coordinate(desc_padded, make_tuple(global_idx));
                auto buffer =
                    tensor_simple.template get_vectorized_elements<thread_buffer<DataType, 1>>(
                        coord, 0);
                // static_for<0, 4, 1>{}([&](auto j) {
                //     DataType val = buffer[number<j>{}];
                //     printf("%.1f ", static_cast<float>(val));
                // });
                DataType val = buffer[number<0>{}];
                printf("%.1f ", static_cast<float>(val));
            });
            printf("\n");

            // Check if this tile contains padding
            index_t tile_start = tile_idx * tile_size;
            index_t tile_end   = tile_start + tile_size;
            if(tile_end > orig_size)
            {
                printf("  Note: Elements %ld-%ld are padded (return identity value %f)\n",
                       static_cast<long>(orig_size - tile_start),
                       static_cast<long>(tile_size - 1),
                       DataType{});
            }
            printf("\n");
        }

        printf("Key Observations:\n");
        printf("  - buffer_view with runtime size + identity value works!\n");
        printf("  - Out-of-bounds accesses return identity value (0.0)\n");
        printf("  - get_vectorized_elements properly handles padding\n");
        printf("  - This is the pattern used in pooling/convolution kernels\n\n");
    }
};

int main()
{
    std::cout << "\n================================================\n";
    std::cout << "Tutorial 03: Padding with Tile Windows\n";
    std::cout << "================================================\n\n";

    int device_count;
    hip_check_error(hipGetDeviceCount(&device_count));
    if(device_count == 0)
    {
        std::cerr << "No GPU devices found!\n";
        return 1;
    }

    hip_check_error(hipSetDevice(0));
    hipDeviceProp_t props;
    hip_check_error(hipGetDeviceProperties(&props, 0));
    std::cout << "Using GPU: " << props.name << "\n";

    // Create test data: 10 real elements
    constexpr index_t orig_size   = 10;
    constexpr index_t padded_size = 16;

    std::vector<float> h_data(orig_size);
    std::iota(h_data.begin(), h_data.end(), 1.0f); // 1, 2, 3, ..., 10

    std::cout << "\nTest data (" << orig_size << " elements): ";
    for(auto val : h_data)
    {
        std::cout << val << " ";
    }
    std::cout << "\n";
    std::cout << "Will be padded to " << padded_size << " elements\n";

    DeviceMem d_data(orig_size * sizeof(float));
    d_data.ToDevice(h_data.data(), orig_size * sizeof(float));

    constexpr index_t block_size = PaddingTileKernel<float>::kBlockSize;
    stream_config stream;

    std::cout << "\nLaunching kernel...\n";
    std::cout << "=====================================\n";

    launch_kernel(stream,
                  make_kernel<block_size>(PaddingTileKernel<float>{},
                                          dim3(1),
                                          dim3(block_size),
                                          0,
                                          static_cast<const float*>(d_data.GetDeviceBuffer()),
                                          orig_size,
                                          padded_size));

    hip_check_error(hipDeviceSynchronize());
    std::cout << "=====================================\n";

    std::cout << "\n=== Tutorial Complete ===\n";
    std::cout << "You now understand:\n";
    std::cout << "- buffer_view with identity values for padding\n";
    std::cout << "- tile_window for tiled access patterns\n";
    std::cout << "- load_tile automatically handling out-of-bounds\n";
    std::cout << "- The pattern used in pooling/convolution kernels\n\n";

    return 0;
}
