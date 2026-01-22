// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 04: Tensor Descriptor vs Tensor Adaptor
 *
 * This tutorial demonstrates the key differences between tensor_adaptor and tensor_descriptor:
 * 1. tensor_adaptor - Pure coordinate transformation logic
 * 2. tensor_descriptor - Complete tensor specification with memory info
 * 3. Coordinate operations - Creating and moving coordinates efficiently
 * 4. Practical examples showing when to use each
 *
 * Key Learning: Understanding the relationship and use cases for adaptors vs descriptors
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct DescriptorVsAdaptorKernel
{
    static constexpr index_t kBlockSize = 64;

    // Part 1: Tensor Adaptor - Pure Transformation Logic
    CK_TILE_DEVICE static void demonstrate_tensor_adaptor()
    {
        printf("PART 1: tensor_adaptor - Pure Coordinate Transformation\n");
        printf("========================================================\n\n");

        printf("Purpose: Define HOW to transform coordinates without memory information.\n\n");

        // Example 1.1: Simple tiling transformation
        printf("Example 1.1: Matrix Tiling [M, K] -> [M0, M1, K]\n");
        printf("------------------------------------------------\n");
        {
            constexpr index_t M  = 128;
            constexpr index_t K  = 64;
            constexpr index_t M0 = 4;
            constexpr index_t M1 = 32;

            printf("Input: [M=%ld, K=%ld]\n", static_cast<long>(M), static_cast<long>(K));
            printf("Output: [M0=%ld, M1=%ld, K=%ld]\n",
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K));

            // Create adaptor - only transformation logic
            auto adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
                           make_pass_through_transform(number<K>{})),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));

            printf("\nAdaptor properties:\n");
            printf("  - Stores: Transformation logic only\n");
            printf("  - Does NOT store: Memory size, vectorization info\n");
            printf("  - Can do: Map coordinates [M0, M1, K] -> [M, K]\n");
            printf("  - Cannot do: Calculate memory offsets\n\n");

            // Test coordinate mapping
            auto top_idx    = make_tuple(2, 16, 32);
            auto bottom_idx = adaptor.calculate_bottom_index(top_idx);

            printf("Coordinate mapping test:\n");
            printf("  Input:  [M0=%ld, M1=%ld, K=%ld]\n",
                   static_cast<long>(top_idx.template get<0>()),
                   static_cast<long>(top_idx.template get<1>()),
                   static_cast<long>(top_idx.template get<2>()));
            printf("  Output: [M=%ld, K=%ld]\n",
                   static_cast<long>(bottom_idx[number<0>{}]),
                   static_cast<long>(bottom_idx[number<1>{}]));
            printf("  Calculation: M = M0*M1 + M1 = 2*32 + 16 = %ld\n",
                   static_cast<long>(bottom_idx[number<0>{}]));
        }

        printf("\n");

        // Example 1.2: Reusable transformation pattern
        printf("Example 1.2: Reusable Transformation Pattern\n");
        printf("---------------------------------------------\n");
        {
            printf("Adaptors are reusable - same transformation for different sizes!\n\n");

            // Define a generic 2D tiling pattern
            auto create_tiling_adaptor = [](auto M0, auto M1, auto N0, auto N1) {
                return make_single_stage_tensor_adaptor(
                    make_tuple(make_unmerge_transform(make_tuple(M0, M1)),
                               make_unmerge_transform(make_tuple(N0, N1))),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0, 1>{}, sequence<2, 3>{}));
            };

            // Use same pattern for different matrix sizes
            [[maybe_unused]] auto adaptor_64x64 =
                create_tiling_adaptor(number<4>{}, number<16>{}, number<4>{}, number<16>{});

            [[maybe_unused]] auto adaptor_128x128 =
                create_tiling_adaptor(number<8>{}, number<16>{}, number<8>{}, number<16>{});

            printf("Created two adaptors with same pattern:\n");
            printf("  - 64x64 matrix:   [64, 64] -> [4, 16, 4, 16]\n");
            printf("  - 128x128 matrix: [128, 128] -> [8, 16, 8, 16]\n");
            printf("\nBoth use identical transformation logic!\n");
        }

        printf("\n\n");
    }

    // Part 2: Tensor Descriptor - Complete Specification
    CK_TILE_DEVICE static void demonstrate_tensor_descriptor()
    {
        printf("PART 2: tensor_descriptor - Complete Tensor Specification\n");
        printf("==========================================================\n\n");

        printf("Purpose: Complete tensor with transformation + memory + vectorization info.\n\n");

        // Example 2.1: Creating a descriptor
        printf("Example 2.1: Creating a Descriptor\n");
        printf("-----------------------------------\n");
        {
            constexpr index_t M = 128;
            constexpr index_t K = 64;

            // Create descriptor - includes memory information
            auto desc = make_naive_tensor_descriptor_packed(make_tuple(number<M>{}, number<K>{}));

            printf("Created descriptor for [M=%ld, K=%ld] matrix\n",
                   static_cast<long>(M),
                   static_cast<long>(K));

            auto space_size = desc.get_element_space_size();
            printf("\nDescriptor properties:\n");
            printf("  - Stores: Transformation logic + memory info\n");
            printf("  - element_space_size: %ld elements\n", static_cast<long>(space_size));
            printf("  - Can calculate: Actual memory offsets\n");
            printf("  - Includes: Vectorization guarantees\n\n");

            // Calculate memory offset
            auto offset1 = desc.calculate_offset(make_tuple(10, 20));
            auto offset2 = desc.calculate_offset(make_tuple(0, 0));
            auto offset3 = desc.calculate_offset(make_tuple(M - 1, K - 1));

            printf("Memory offset calculations:\n");
            printf("  [10, 20] -> offset %ld (10*64 + 20)\n", static_cast<long>(offset1));
            printf("  [0, 0]   -> offset %ld (first element)\n", static_cast<long>(offset2));
            printf("  [%ld, %ld] -> offset %ld (last element)\n",
                   static_cast<long>(M - 1),
                   static_cast<long>(K - 1),
                   static_cast<long>(offset3));
        }

        printf("\n");

        // Example 2.2: Transforming a Descriptor
        printf("Example 2.2: Transforming a Descriptor\n");
        printf("---------------------------------------\n");
        {
            constexpr index_t M  = 256;
            constexpr index_t K  = 128;
            constexpr index_t M0 = 4;
            constexpr index_t M1 = 64;

            printf("Step 1: Create initial descriptor\n");
            auto desc_initial =
                make_naive_tensor_descriptor_packed(make_tuple(number<M>{}, number<K>{}));
            printf("  Initial: [M=%ld, K=%ld]\n", static_cast<long>(M), static_cast<long>(K));
            printf("  Memory size: %ld elements\n\n",
                   static_cast<long>(desc_initial.get_element_space_size()));

            printf("Step 2: Transform to add tiling\n");
            auto desc_tiled = transform_tensor_descriptor(
                desc_initial,
                make_tuple(make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
                           make_pass_through_transform(number<K>{})),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));

            printf("  Transformed: [M, K] -> [M0=%ld, M1=%ld, K=%ld]\n",
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K));
            printf("  Memory size preserved: %ld elements\n\n",
                   static_cast<long>(desc_tiled.get_element_space_size()));

            printf("Now we can calculate actual memory offsets!\n");
            auto offset = desc_tiled.calculate_offset(make_tuple(2, 16, 32));
            printf("  [M0=2, M1=16, K=32] -> offset %ld\n", static_cast<long>(offset));
        }

        printf("\n\n");
    }

    // Part 3: Coordinate Operations
    CK_TILE_DEVICE static void demonstrate_coordinate_operations()
    {
        printf("PART 3: Coordinate Operations - Creating and Moving\n");
        printf("====================================================\n\n");

        printf("Purpose: Efficiently track and update positions in tensor space.\n\n");

        // Example 3.1: Creating coordinates
        printf("Example 3.1: Creating Coordinates\n");
        printf("----------------------------------\n");
        {
            constexpr index_t M = 64;
            constexpr index_t K = 32;

            auto desc = make_naive_tensor_descriptor_packed(make_tuple(number<M>{}, number<K>{}));

            printf("Descriptor: [M=%ld, K=%ld]\n\n", static_cast<long>(M), static_cast<long>(K));

            // Create coordinate at position [10, 20]
            auto coord = make_tensor_coordinate(desc, make_tuple(10, 20));

            printf("Created coordinate at [10, 20]:\n");
            printf("  - Top index (user view): [%ld, %ld]\n",
                   static_cast<long>(coord.get_index()[number<0>{}]),
                   static_cast<long>(coord.get_index()[number<1>{}]));
            printf("  - Memory offset: %ld\n", static_cast<long>(coord.get_offset()));
            printf("  - Calculation: 10*32 + 20 = %ld\n", static_cast<long>(coord.get_offset()));
        }

        printf("\n");

        // Example 3.2: Moving coordinates (efficient iteration)
        printf("Example 3.2: Moving Coordinates - Efficient Iteration\n");
        printf("------------------------------------------------------\n");
        {
            constexpr index_t M = 64;
            constexpr index_t K = 32;

            auto desc = make_naive_tensor_descriptor_packed(make_tuple(number<M>{}, number<K>{}));

            printf("Scenario: Iterate through a row of tiles\n");
            printf("Descriptor: [M=%ld, K=%ld]\n\n", static_cast<long>(M), static_cast<long>(K));

            // Start at [0, 0]
            auto coord = make_tensor_coordinate(desc, make_tuple(0, 0));
            printf("Initial position [0, 0]:\n");
            printf("  Offset: %ld\n\n", static_cast<long>(coord.get_offset()));

            // Move to [0, 8] - move by 8 in K dimension
            printf("Move by [0, 8] (8 columns to the right):\n");
            move_tensor_coordinate(desc, coord, make_tuple(0, 8));
            printf("  New position: [%ld, %ld]\n",
                   static_cast<long>(coord.get_index()[number<0>{}]),
                   static_cast<long>(coord.get_index()[number<1>{}]));
            printf("  New offset: %ld\n", static_cast<long>(coord.get_offset()));
            printf("  (Much faster than creating new coordinate!)\n\n");

            // Move to [1, 8] - move by 1 in M dimension
            printf("Move by [1, 0] (1 row down):\n");
            move_tensor_coordinate(desc, coord, make_tuple(1, 0));
            printf("  New position: [%ld, %ld]\n",
                   static_cast<long>(coord.get_index()[number<0>{}]),
                   static_cast<long>(coord.get_index()[number<1>{}]));
            printf("  New offset: %ld\n", static_cast<long>(coord.get_offset()));
            printf("  Calculation: 1*32 + 8 = %ld\n\n", static_cast<long>(coord.get_offset()));

            printf("Why use move_tensor_coordinate?\n");
            printf("  ✓ Incremental update - only recalculates what changed\n");
            printf("  ✓ Skips unnecessary transformations (optimization)\n");
            printf("  ✓ Essential for tile window iteration patterns\n");
            printf("  ✓ Much faster than creating new coordinates\n");
        }

        printf("\n");

        // Example 3.3: Moving with complex transformations
        printf("Example 3.3: Moving Coordinates with Transformations\n");
        printf("----------------------------------------------------\n");
        {
            constexpr index_t M  = 128;
            constexpr index_t K  = 64;
            constexpr index_t M0 = 4;
            constexpr index_t M1 = 32;

            // Create tiled descriptor
            auto desc = make_naive_tensor_descriptor_packed(make_tuple(number<M>{}, number<K>{}));

            auto desc_tiled = transform_tensor_descriptor(
                desc,
                make_tuple(make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
                           make_pass_through_transform(number<K>{})),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));

            printf("Tiled descriptor: [M, K] -> [M0=%ld, M1=%ld, K=%ld]\n\n",
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K));

            // Create coordinate
            auto coord = make_tensor_coordinate(desc_tiled, make_tuple(1, 8, 16));
            printf("Initial: [M0=1, M1=8, K=16]\n");
            printf("  Offset: %ld\n\n", static_cast<long>(coord.get_offset()));

            // Move to next tile in M1 dimension
            move_tensor_coordinate(desc_tiled, coord, make_tuple(0, 4, 0));
            printf("After move [0, 4, 0]:\n");
            printf("  Position: [M0=%ld, M1=%ld, K=%ld]\n",
                   static_cast<long>(coord.get_index()[number<0>{}]),
                   static_cast<long>(coord.get_index()[number<1>{}]),
                   static_cast<long>(coord.get_index()[number<2>{}]));
            printf("  Offset: %ld\n", static_cast<long>(coord.get_offset()));
            printf("\nThe move operation efficiently propagates through transformations!\n");
        }

        printf("\n\n");
    }

    // Part 4: When to Use Which
    CK_TILE_DEVICE static void demonstrate_use_cases()
    {
        printf("PART 4: When to Use Adaptor vs Descriptor\n");
        printf("==========================================\n\n");

        printf("Use tensor_adaptor when:\n");
        printf("  ✓ Designing reusable transformation patterns\n");
        printf("  ✓ Building intermediate transformation stages\n");
        printf("  ✓ Composing transformations with chain_tensor_adaptors()\n");
        printf("  ✓ Memory size is not relevant\n");
        printf("  ✓ Only need coordinate mapping logic\n\n");

        printf("Use tensor_descriptor when:\n");
        printf("  ✓ Working with actual tensors that need memory\n");
        printf("  ✓ Calculating actual memory offsets\n");
        printf("  ✓ Need to know total memory footprint\n");
        printf("  ✓ Require vectorization guarantees\n");
        printf("  ✓ Creating tensor_view for data access\n");
        printf("  ✓ Working with physical memory buffers\n\n");

        printf("Relationship:\n");
        printf("  tensor_descriptor IS-A tensor_adaptor (inheritance)\n");
        printf("  descriptor = adaptor + memory info + vectorization info\n\n");

        printf("Think of it as:\n");
        printf("  adaptor    = \"The recipe\" (how to transform)\n");
        printf("  descriptor = \"Recipe + ingredients + kitchen\" (complete spec)\n\n");
    }

    CK_TILE_DEVICE void operator()() const
    {
        if(get_thread_id() != 0)
            return;

        printf("\n=== TENSOR DESCRIPTOR VS TENSOR ADAPTOR ===\n\n");

        demonstrate_tensor_adaptor();
        demonstrate_tensor_descriptor();
        demonstrate_coordinate_operations();
        demonstrate_use_cases();

        printf("=== KEY TAKEAWAYS ===\n\n");
        printf("1. tensor_adaptor:\n");
        printf("   - Pure coordinate transformation logic\n");
        printf("   - Lightweight, reusable patterns\n");
        printf("   - No memory or vectorization info\n\n");

        printf("2. tensor_descriptor:\n");
        printf("   - Complete tensor specification\n");
        printf("   - Inherits from tensor_adaptor\n");
        printf("   - Adds memory size and vectorization guarantees\n");
        printf("   - Can calculate actual memory offsets\n\n");

        printf("3. Coordinate operations:\n");
        printf("   - make_tensor_coordinate() creates position tracker\n");
        printf("   - move_tensor_coordinate() efficiently updates position\n");
        printf("   - Essential for tile window iteration\n\n");

        printf("4. Use the right tool:\n");
        printf("   - Adaptor for transformation patterns\n");
        printf("   - Descriptor for actual tensors with memory\n\n");
    }
};

int main()
{
    std::cout << "\n================================================\n";
    std::cout << "Tutorial 04: Tensor Descriptor vs Tensor Adaptor\n";
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

    constexpr index_t block_size = DescriptorVsAdaptorKernel<float>::kBlockSize;
    stream_config stream;

    std::cout << "\nLaunching kernel...\n";
    std::cout << "=====================================\n";

    launch_kernel(
        stream,
        make_kernel<block_size>(DescriptorVsAdaptorKernel<float>{}, dim3(1), dim3(block_size), 0));

    hip_check_error(hipDeviceSynchronize());
    std::cout << "=====================================\n";

    std::cout << "\n=== Tutorial Complete ===\n";
    std::cout << "You now understand:\n";
    std::cout << "- The difference between tensor_adaptor and tensor_descriptor\n";
    std::cout << "- When to use each abstraction\n";
    std::cout << "- How to create and move coordinates efficiently\n";
    std::cout << "- The inheritance relationship between them\n";
    std::cout << "- Practical examples of both in action\n\n";

    std::cout << "See DESCRIPTOR_VS_ADAPTOR.md for detailed documentation.\n\n";

    return 0;
}
