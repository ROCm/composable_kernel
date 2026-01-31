// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Tutorial 02: Tensor Adaptors - Advanced Layout Transformations
 *
 * This tutorial teaches the three core tensor adaptor methods:
 * 1. make_single_stage_tensor_adaptor - Create a single-stage transformation
 * 2. transform_tensor_adaptor - Add new transformations to existing adaptor
 * 3. chain_tensor_adaptors - Chain two adaptors together
 *
 * Key Learning: Tensor adaptors enable zero-copy view transformations for
 * complex memory layouts used in high-performance GPU kernels.
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template <typename DataType>
struct TensorAdaptorsKernel
{
    static constexpr index_t kBlockSize = 64;

    // Part 1: make_single_stage_tensor_adaptor examples
    CK_TILE_DEVICE static void demonstrate_single_stage()
    {
        printf("PART 1: <demonstrate_single_stage> make_single_stage_tensor_adaptor\n");
        printf("=========================================\n\n");

        printf(
            "Purpose: Create a tensor adaptor with transformations applied in a single stage.\n");
        printf("This is the foundation for building complex layout transformations.\n\n");

        // Example 1.1: Simple dimension split (Unmerge)
        printf("Example 1.1: Split M dimension for tiling\n");
        printf("------------------------------------------\n");
        {
            constexpr index_t M  = 128;
            constexpr index_t K  = 64;
            constexpr index_t M0 = 4;
            constexpr index_t M1 = M / M0; // M1 = 32

            printf("Input layout: [M=%ld, K=%ld]\n", static_cast<long>(M), static_cast<long>(K));
            printf("Goal: Split M into [M0=%ld, M1=%ld] for tiling\n",
                   static_cast<long>(M0),
                   static_cast<long>(M1));
            /*
            for(index_t m = 0; m < M; m++)
            {
                index_t m0 = m / M1;
                index_t m1 = m % M1;
                // printf("  M=%ld -> [M0=%ld, M1=%ld]\n", static_cast<long>(m),
                //     static_cast<long>(m0), static_cast<long>(m1)); for(index_t k = 0; k < K; k++)
                {
                    index_t input_offset  = m * K + k;
                    index_t output_offset = (m0 * M1 + m1) * K + k;
                    // printf("    K=%ld: input_offset=%ld -> output_offset=%ld\n",
                    //                static_cast<long>(k), static_cast<long>(input_offset),
                    //                static_cast<long>(output_offset));
                }
            }
            */
            auto transforms =
                make_tuple(make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
                           make_pass_through_transform(number<K>{}));

            auto lower_dims = make_tuple(sequence<0>{}, sequence<1>{});    // 2D bottom index
            auto upper_dims = make_tuple(sequence<0, 1>{}, sequence<2>{}); // 3D top index

            auto adaptor = make_single_stage_tensor_adaptor(transforms, lower_dims, upper_dims);

            /*
            for(index_t m0 = 0; m0 < M0; m0++)
                for(index_t m1 = 0; m1 < M1; m1++)
                    for(index_t k = 0; k < K; k++)
                    {
                        index_t m            = m0 * M1 + m1;
                        index_t input_offset = m * K + k;
                        auto top_idx         = make_tuple(m0, m1, k);
                        auto bottom_idx      = adaptor.calculate_bottom_index(top_idx); // [m, k]
                    }
            */
            printf("\nAdaptor created:\n");
            printf("  Input: [M, K] = [%ld, %ld]\n", static_cast<long>(M), static_cast<long>(K));
            printf("  Output: [M0, M1, K] = [%ld, %ld, %ld]\n",
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K));

            auto top_idx    = make_tuple(1, 16, 32);
            auto bottom_idx = adaptor.calculate_bottom_index(top_idx); //[1 * M1 + 16, 32]

            printf("\nTest: [M0=1, M1=16, K=32] -> [M=%ld, K=%ld]\n",
                   static_cast<long>(bottom_idx[number<0>{}]),
                   static_cast<long>(bottom_idx[number<1>{}]));
        }

        printf("\n");

        // Example 1.2: Interleaved layout
        printf("Example 1.2: GEMM C Matrix Tiling (Interleaved)\n");
        printf("------------------------------------------------\n");
        {
            constexpr index_t M  = 256;
            constexpr index_t N  = 256;
            constexpr index_t M0 = 4;
            constexpr index_t M1 = M / M0; // M1 = 64
            constexpr index_t N0 = 4;
            constexpr index_t N1 = N / N0; // N1 = 64

            printf("Input: [M=%ld, N=%ld]\n", static_cast<long>(M), static_cast<long>(N));
            printf("Output: [M0=%ld, N0=%ld, M1=%ld, N1=%ld] (interleaved)\n",
                   static_cast<long>(M0),
                   static_cast<long>(N0),
                   static_cast<long>(M1),
                   static_cast<long>(N1));

            for(index_t m = 0; m < M; m++)
            {
                index_t m0 = m / M1;
                index_t m1 = m % M1;
                for(index_t n = 0; n < N; n++)
                {
                    index_t n0 = n / N1;
                    index_t n1 = n % N1;
                    if(m0 == 2 && n0 == 3 && m1 == 16 && n1 == 32)
                    {
                        index_t input_offset = m * N + n;
                        index_t natural_output_offset =
                            m0 * M1 * N0 * N1 + m1 * N0 * N1 + n0 * N1 + n1;
                        index_t transf_output_offset =
                            m0 * M1 * N0 * N1 + n0 * M1 * N1 + m1 * N1 + n1;
                        printf(
                            "\n[m=%ld, n=%ld] -> [m0=%ld, n0=%ld, m1=%ld, n1=%ld] -> [%ld*M1+%ld, "
                            "%ld*N1+%ld] | input_offset=%ld=%ld*N+%ld  -> "
                            "natural_output_offset=%ld=%ld * M1 * N0 * N1 + %ld * N0 * N1 + %ld * "
                            "N1 + %ld  -> transf_output_offset=%ld=%ld * M1 * N0 * N1 + %ld * M1 * "
                            "N1 + %ld * N1 + %ld",
                            static_cast<long>(m),
                            static_cast<long>(n),
                            static_cast<long>(m0),
                            static_cast<long>(n0),
                            static_cast<long>(m1),
                            static_cast<long>(n1),
                            static_cast<long>(m0),
                            static_cast<long>(m1),
                            static_cast<long>(n0),
                            static_cast<long>(n1),
                            static_cast<long>(input_offset),
                            static_cast<long>(m),
                            static_cast<long>(n),
                            static_cast<long>(natural_output_offset),
                            static_cast<long>(m0),
                            static_cast<long>(m1),
                            static_cast<long>(n0),
                            static_cast<long>(n1),
                            static_cast<long>(transf_output_offset),
                            static_cast<long>(m0),
                            static_cast<long>(n0),
                            static_cast<long>(m1),
                            static_cast<long>(n1));
                    }
                }
            }

            auto transforms =
                make_tuple(make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
                           make_unmerge_transform(make_tuple(number<N0>{}, number<N1>{})));

            auto lower_dims = make_tuple(sequence<0>{}, sequence<1>{}); // 2D bottom index
            auto upper_dims = make_tuple(sequence<0, 2>{},              // M splits to dims 0,2
                                         sequence<1, 3>{}               // N splits to dims 1,3
            ); // 4D top index [M0, N0, M1, N1]

            auto adaptor = make_single_stage_tensor_adaptor(transforms, lower_dims, upper_dims);

            /*
            for(index_t m0 = 0; m0 < M0; m0++)
                for(index_t n0 = 0; n0 < N0; n0++)
                    for(index_t m1 = 0; m1 < M1; m1++)
                        for(index_t n1 = 0; n1 < N1; n1++)
                        {
                            index_t m            = m0 * M1 + m1;
                            index_t n            = n0 * N1 + n1;
                            index_t input_offset = m * N + n;
                            // auto top_idx    = make_tuple(m0, n0, m1, n1);
                            // auto bottom_idx = adaptor.calculate_bottom_index(top_idx); // [m, n]
                        }
            */

            auto top_idx    = make_tuple(2, 3, 16, 32);
            auto bottom_idx = adaptor.calculate_bottom_index(top_idx); //[2 * M1 + 16, 3 * N1 + 32]
            // NOTE: Bottom index computation fuses 0 and 2 dimensions for M, and 1 and 3 for N
            printf("\nTest: [M0=2, N0=3, M1=16, N1=32] -> [M=%ld, N=%ld]\n",
                   static_cast<long>(bottom_idx[number<0>{}]),
                   static_cast<long>(bottom_idx[number<1>{}]));
        }

        printf("\n\n");
    }

    // Part 2: transform_tensor_adaptor examples
    CK_TILE_DEVICE static void demonstrate_transform()
    {
        printf("PART 2: transform_tensor_adaptor\n");
        printf("=================================\n\n");

        printf("Purpose: Add new transformations to an existing tensor adaptor.\n\n");

        // Example 2.1: Two-stage transformation
        printf("Example 2.1: Two-Stage Hierarchical Tiling\n");
        printf("-------------------------------------------\n");
        {
            constexpr index_t M  = 256;
            constexpr index_t K  = 128;
            constexpr index_t M0 = 4;
            constexpr index_t M1 = M / M0; // M1 = 64
            constexpr index_t K0 = 4;
            constexpr index_t K1 = K / K0; // K1 = 32

            printf("Stage 1: [M=%ld, K=%ld] -> [M0=%ld, M1=%ld, K=%ld]\n",
                   static_cast<long>(M),
                   static_cast<long>(K),
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K));

            auto stage1_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
                           make_pass_through_transform(number<K>{})),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));

            printf("Stage 2: [M0=%ld, M1=%ld, K=%ld] -> [M0=%ld, M1=%ld, K0=%ld, K1=%ld]\n",
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K),
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K0),
                   static_cast<long>(K1));

            auto final_adaptor = transform_tensor_adaptor(
                stage1_adaptor,
                make_tuple(make_pass_through_transform(number<M0>{}),
                           make_pass_through_transform(number<M1>{}),
                           make_unmerge_transform(make_tuple(number<K0>{}, number<K1>{}))),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}));

            auto top_idx    = make_tuple(2, 32, 3, 16);
            auto bottom_idx = final_adaptor.calculate_bottom_index(top_idx);
            printf("\nTest: [M0=2, M1=32, K0=3, K1=16] -> [M=%ld, K=%ld]\n",
                   static_cast<long>(bottom_idx[number<0>{}]),
                   static_cast<long>(bottom_idx[number<1>{}]));

            auto test_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
                           make_unmerge_transform(make_tuple(number<K0>{}, number<K1>{}))),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}));
            auto test_bottom_idx = test_adaptor.calculate_bottom_index(top_idx);
            printf("\nTest1: [M0=2, M1=32, K0=3, K1=16] -> [M=%ld, K=%ld]\n",
                   static_cast<long>(test_bottom_idx[number<0>{}]),
                   static_cast<long>(test_bottom_idx[number<1>{}]));
            /*
            for(index_t m0 = 0; m0 < M0; m0++)
                for(index_t m1 = 0; m1 < M1; m1++)
                    for(index_t k0 = 0; k0 < K0; k0++)
                        for(index_t k1 = 0; k1 < K1; k1++)
                        {
                            index_t m            = m0 * M1 + m1;
                            index_t k            = k0 * K1 + k1;
                            index_t input_offset = m * K + k;
                            // auto top_idx    = make_tuple(m0, m1, k0, k1);
                            // auto bottom_idx = final_adaptor.calculate_bottom_index(top_idx); //
                            // [m, k]
                        }
            */
        }

        printf("\n\n");
    }

    // Part 3: chain_tensor_adaptors examples
    CK_TILE_DEVICE static void demonstrate_chain()
    {
        printf("PART 3: chain_tensor_adaptors\n");
        printf("==============================\n\n");

        printf("Purpose: Chain two tensor adaptors sequentially.\n\n");

        printf("Example 3.1: Chain Two Adaptors\n");
        printf("--------------------------------\n");
        {
            constexpr index_t M  = 128;
            constexpr index_t K  = 64;
            constexpr index_t M0 = 4;
            constexpr index_t M1 = M / M0; // M1 = 32;
            constexpr index_t K0 = 4;
            constexpr index_t K1 = K / K0; // K1 = 16;

            printf("Adaptor A: [M=%ld, K=%ld] -> [M0=%ld, M1=%ld, K=%ld]\n",
                   static_cast<long>(M),
                   static_cast<long>(K),
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K));

            auto adaptor_a = make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
                           make_pass_through_transform(number<K>{})),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));

            printf("Adaptor B: [M0=%ld, M1=%ld, K=%ld] -> [M0=%ld, M1=%ld, K0=%ld, K1=%ld]\n",
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K),
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K0),
                   static_cast<long>(K1));

            auto adaptor_b = make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(number<M0>{}),
                           make_pass_through_transform(number<M1>{}),
                           make_unmerge_transform(make_tuple(number<K0>{}, number<K1>{}))),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}));

            auto chained = chain_tensor_adaptors(adaptor_a, adaptor_b); // union of both

            printf("\nChained: [M=%ld, K=%ld] -> [M0=%ld, M1=%ld, K0=%ld, K1=%ld]\n",
                   static_cast<long>(M),
                   static_cast<long>(K),
                   static_cast<long>(M0),
                   static_cast<long>(M1),
                   static_cast<long>(K0),
                   static_cast<long>(K1));

            auto top_idx    = make_tuple(2, 16, 3, 8);
            auto bottom_idx = chained.calculate_bottom_index(top_idx);
            printf("Test: [M0=2, M1=16, K0=3, K1=8] -> [M=%ld, K=%ld]\n",
                   static_cast<long>(bottom_idx[number<0>{}]),
                   static_cast<long>(bottom_idx[number<1>{}]));
        }

        printf("\n\n");
    }

    // Part 4: Real-world GEMM example
    CK_TILE_DEVICE static void demonstrate_gemm_tiling()
    {
        printf("PART 4: Real-World GEMM Tiling Example\n");
        printf("=======================================\n\n");

        constexpr index_t M       = 256;
        constexpr index_t N       = 256;
        constexpr index_t MWaves  = 4;
        constexpr index_t NWaves  = 4;
        constexpr index_t MPerXDL = 16;
        constexpr index_t NPerXDL = 16;
        constexpr index_t M0      = M / (MWaves * MPerXDL); // M0 = 4
        constexpr index_t N0      = N / (NWaves * NPerXDL); // N0 = 4

        printf("GEMM C Matrix: [M=%ld, N=%ld]\n", static_cast<long>(M), static_cast<long>(N));
        printf("Tiling: [M0=%ld, N0=%ld, M1=%ld, N1=%ld, M2=%ld, N2=%ld]\n",
               static_cast<long>(M0),
               static_cast<long>(N0),
               static_cast<long>(MWaves),
               static_cast<long>(NWaves),
               static_cast<long>(MPerXDL),
               static_cast<long>(NPerXDL));

        auto adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(
                           make_tuple(number<M0>{}, number<MWaves>{}, number<MPerXDL>{})),
                       make_unmerge_transform(
                           make_tuple(number<N0>{}, number<NWaves>{}, number<NPerXDL>{}))),
            make_tuple(sequence<0>{}, sequence<1>{}),              // [M, N]
            make_tuple(sequence<0, 2, 4>{}, sequence<1, 3, 5>{})); // [M0, N0, M1, N1, M2, N2]

        auto top_idx    = make_tuple(2, 3, 1, 2, 8, 12);
        auto bottom_idx = adaptor.calculate_bottom_index(top_idx);
        printf("\nTest: [M0=2, N0=3, M1=1, N1=2, M2=8, N2=12] -> [M=%ld, N=%ld]\n",
               static_cast<long>(bottom_idx[number<0>{}]),
               static_cast<long>(bottom_idx[number<1>{}]));

        // Check tile distribution
        for(index_t m0 = 0; m0 < M0; m0++)
            for(index_t n0 = 0; n0 < N0; n0++)
                for(index_t m1 = 0; m1 < MWaves; m1++)
                    for(index_t n1 = 0; n1 < NWaves; n1++)
                    {
                        index_t tile_id = (m0 * N0 + n0) * (MWaves * NWaves) + (m1 * NWaves + n1);

                        auto tile_top_idx    = make_tuple(m0, n0, m1, n1, 0, 0);
                        auto tile_bottom_idx = adaptor.calculate_bottom_index(tile_top_idx);

                        using Coord2D = std::pair<index_t, index_t>;

                        Coord2D top_corner = {tile_bottom_idx[number<0>{}],
                                              tile_bottom_idx[number<1>{}]};

                        tile_top_idx    = make_tuple(m0, n0, m1, n1, MPerXDL - 1, NPerXDL - 1);
                        tile_bottom_idx = adaptor.calculate_bottom_index(tile_top_idx);

                        Coord2D bottom_corner = {tile_bottom_idx[number<0>{}],
                                                 tile_bottom_idx[number<1>{}]};

                        printf("Tile ID %2ld: [M0=%ld, N0=%ld, M1=%ld, N1=%ld] = [%ld, %ld] -> "
                               "[%ld, %ld]\n",
                               static_cast<long>(tile_id),
                               static_cast<long>(m0),
                               static_cast<long>(n0),
                               static_cast<long>(m1),
                               static_cast<long>(n1),
                               static_cast<long>(top_corner.first),
                               static_cast<long>(top_corner.second),
                               static_cast<long>(bottom_corner.first),
                               static_cast<long>(bottom_corner.second));

                        if(tile_id == 171)
                        {
                            // print this tile
                            printf("\nTile %2ld:\n", static_cast<long>(tile_id));

                            for(index_t m2 = 0; m2 < MPerXDL; m2++)
                            {
                                for(index_t n2 = 0; n2 < NPerXDL; n2++)
                                {
                                    auto coord_top    = make_tuple(m0, n0, m1, n1, m2, n2);
                                    auto coord_bottom = adaptor.calculate_bottom_index(coord_top);
                                    printf("  [%3ld, %3ld] ",
                                           static_cast<long>(coord_bottom[number<0>{}]),
                                           static_cast<long>(coord_bottom[number<1>{}]));
                                }
                                printf("\n");
                            }
                            printf("\n");
                        }
                    }

        printf("\n\n");
    }

    // Part 5: Padding Transform - Coordinate mapping demonstration
    CK_TILE_DEVICE static void demonstrate_padding_transform(const DataType* p_data)
    {
        printf("PART 5: Padding Transform - Virtual Padding\n");
        printf("============================================\n\n");

        printf("Demonstrating padding transform with coordinate mapping.\n\n");

        // Original size: 10 elements, pad to 16
        constexpr index_t OrigSize  = 10;
        constexpr index_t PadRight  = 6;
        constexpr index_t TotalSize = OrigSize + PadRight;

        printf("Original size: %ld elements\n", static_cast<long>(OrigSize));
        printf("Padding: +%ld elements (right)\n", static_cast<long>(PadRight));
        printf("Total size: %ld elements\n\n", static_cast<long>(TotalSize));

        // Create padded descriptor
        auto desc_padded = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(number<OrigSize>{})),
            make_tuple(make_right_pad_transform(number<OrigSize>{}, number<PadRight>{})),
            make_tuple(sequence<0>{}),
            make_tuple(sequence<0>{}));

        printf("Coordinate mapping and memory reads:\n");
        printf("------------------------------------\n\n");

        printf("Real area (indices 0-9):\n");
        for(index_t i = 0; i < OrigSize; i++)
        {
            auto coord     = make_tensor_coordinate(desc_padded, make_tuple(i));
            index_t offset = coord.get_offset();
            DataType val   = p_data[offset];

            printf("  Index %ld -> offset %ld -> value %.1f (real data)\n",
                   static_cast<long>(i),
                   static_cast<long>(offset),
                   static_cast<float>(val));
        }

        printf("\nPadded area (indices 10-15):\n");
        for(index_t i = OrigSize; i < TotalSize; i++)
        {
            auto coord     = make_tensor_coordinate(desc_padded, make_tuple(i));
            index_t offset = coord.get_offset(); // XXX: offset does not wrap!!!
            DataType val   = p_data[offset];

            printf("  Index %ld -> offset %ld -> value %.1f (wraps around\?\?\?)\n",
                   static_cast<long>(i),
                   static_cast<long>(offset),
                   static_cast<float>(val));
        }

        printf("\nKey Observations:\n");
        printf("  - Real area (0-9): Maps to offsets 0-9, returns actual data\n");
        printf("  - Padded area (10-15): Offsets wrap (modulo), reads same data\n");
        printf("  - Padding is virtual - no extra memory allocated\n");
        printf("  - In production (pooling/conv), buffer_view with identity value returns 0\n");
        printf("  - Common use: Pad irregular sizes to match tile boundaries\n\n");
    }

    // Part 6: Replicate Transform with comprehensive coordinate testing
    CK_TILE_DEVICE static void demonstrate_replicate_transform()
    {
        printf("PART 5: Replicate Transform - Broadcasting Dimensions\n");
        printf("======================================================\n\n");

        printf("Demonstrating replicate transform with complete coordinate mapping.\n\n");

        // Start with flattened 1D tensor
        constexpr index_t Size = 16; // H*W = 2*8

        printf("Step 1: Create initial 1D descriptor [Size=%ld]\n", static_cast<long>(Size));

        auto desc = make_naive_tensor_descriptor_packed(make_tuple(number<Size>{}));

        printf("  Initial: [16] (flattened)\n\n");

        // Stage 1: Replicate + Unmerge
        printf("Step 2: Apply Replicate and Unmerge\n");
        printf("  Transform 0: Replicate (no input) -> [Rep0=8]\n");
        printf("  Transform 1: Unmerge [16] -> [Dim0=8, Dim1=2]\n");

        auto desc_stage1 = transform_tensor_descriptor(
            desc,
            make_tuple(
                make_replicate_transform(make_tuple(number<8>{})),           // Broadcast to 8
                make_unmerge_transform(make_tuple(number<8>{}, number<2>{})) // Split 16 -> [8,2]
                ),
            make_tuple(sequence<>{}, sequence<0>{}), // Replicate has no input, Unmerge uses dim 0
            make_tuple(sequence<0>{}, sequence<1, 2>{}) // Rep0=dim0, Unmerge produces dims 1,2
        );

        printf("\n  After Stage 1: [Rep0=8, Dim0=8, Dim1=2]\n");
        printf("  Total: 3 dimensions\n\n");

        // Stage 2: Merge Rep0 with Dim0
        printf("Step 3: Merge [Rep0, Dim0] -> [Merged=64]\n");

        auto desc_final = transform_tensor_descriptor(
            desc_stage1,
            make_tuple(
                make_merge_transform(make_tuple(number<8>{}, number<8>{})), // Merge Rep0, Dim0
                make_pass_through_transform(number<2>{})                    // Dim1 unchanged
                ),
            make_tuple(sequence<0, 1>{}, sequence<2>{}), // Merge dims 0,1; pass-through dim 2
            make_tuple(sequence<0>{}, sequence<1>{})     // Output: [Merged, Dim1]
        );

        printf("\n  Final: [Merged=64, Dim1=2]\n\n");

        // Comprehensive coordinate testing - ALL coordinates
        printf("COORDINATE MAPPING TEST - ALL %ld coordinates:\n", static_cast<long>(64 * 2));
        printf("=======================================================\n");
        printf("Format: [Merged, Dim1] -> memory_offset\n\n");

        auto lengths_final = desc_final.get_lengths();
        index_t merged_len = lengths_final[number<0>{}];
        index_t dim1_len   = lengths_final[number<1>{}];

        printf("Descriptor: [Merged=%ld, Dim1=%ld] = %ld total coordinates\n",
               static_cast<long>(merged_len),
               static_cast<long>(dim1_len),
               static_cast<long>(merged_len * dim1_len));
        printf("Memory: Only 16 locations (broadcasting effect!)\n\n");

        // Print ALL coordinates to show broadcasting pattern
        index_t count = 0;
        for(index_t merged = 0; merged < merged_len; merged++)
        {
            for(index_t dim1 = 0; dim1 < dim1_len; dim1++)
            {
                auto coord     = make_tensor_coordinate(desc_final, make_tuple(merged, dim1));
                index_t offset = coord.get_offset();
                printf("  [%2ld, %ld] -> offset %2ld",
                       static_cast<long>(merged),
                       static_cast<long>(dim1),
                       static_cast<long>(offset));

                // Add newline every 4 coordinates for readability
                count++;
                if(count % 2 == 0)
                {
                    printf("\n");
                }
                else
                {
                    printf("  |  ");
                }
            }
        }
        if(count % 2 != 0)
            printf("\n");

        printf("\nKey Observations:\n");
        printf("  - Total coordinates: %ld (Merged=%ld × Dim1=%ld)\n",
               static_cast<long>(merged_len * dim1_len),
               static_cast<long>(merged_len),
               static_cast<long>(dim1_len));
        printf("  - Memory locations: 16 (original size)\n");
        printf("  - Broadcasting ratio: %ld:1 (each memory location accessed by %ld coordinates)\n",
               static_cast<long>((merged_len * dim1_len) / 16),
               static_cast<long>((merged_len * dim1_len) / 16));
        printf("  - Replicate dimension creates virtual coordinates without memory cost!\n\n");
    }

    CK_TILE_DEVICE void operator()(const DataType* p_data) const
    {
        if(get_thread_id() != 0)
            return;

        printf("\n=== TENSOR ADAPTORS IN CK_TILE ===\n\n");

        demonstrate_single_stage();
        demonstrate_transform();
        demonstrate_chain();
        demonstrate_gemm_tiling();
        demonstrate_padding_transform(p_data);
        demonstrate_replicate_transform();

        printf("=== KEY TAKEAWAYS ===\n\n");
        printf("1. make_single_stage_tensor_adaptor:\n");
        printf("   - Creates adaptor with transformations in one stage\n");
        printf("   - Foundation for all tensor layout transformations\n\n");

        printf("2. transform_tensor_adaptor:\n");
        printf("   - Adds new transformations to existing adaptor\n");
        printf("   - Enables incremental building of complex layouts\n\n");

        printf("3. chain_tensor_adaptors:\n");
        printf("   - Composes two (or more) adaptors sequentially\n");
        printf("   - Enables modular transformation design\n\n");

        printf("4. Replicate transform:\n");
        printf("   - Broadcasts dimensions (creates from nothing)\n");
        printf("   - Useful for repeating data across tiles\n\n");

        printf("5. All transformations are zero-copy views!\n\n");
    }
};

int main()
{
    std::cout << "\n================================================\n";
    std::cout << "Tutorial 02: Tensor Adaptors\n";
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

    // Allocate data for padding example (16 elements, but only first 10 have real data)
    constexpr index_t data_size = 16;
    std::vector<float> h_data(data_size, 0.0f);           // Initialize all to 0
    std::iota(h_data.begin(), h_data.begin() + 10, 1.0f); // First 10: 1,2,3,...,10

    std::cout << "\nTest data (first 10 real, last 6 padding zeros): ";
    for(size_t i = 0; i < h_data.size(); i++)
    {
        std::cout << h_data[i];
        if(i < h_data.size() - 1)
            std::cout << " ";
    }
    std::cout << "\n";

    DeviceMem d_data(data_size * sizeof(float));
    d_data.ToDevice(h_data.data(), data_size * sizeof(float));

    constexpr index_t block_size = TensorAdaptorsKernel<float>::kBlockSize;
    stream_config stream;

    std::cout << "\nLaunching kernel...\n";
    std::cout << "=====================================\n";

    launch_kernel(stream,
                  make_kernel<block_size>(TensorAdaptorsKernel<float>{},
                                          dim3(1),
                                          dim3(block_size),
                                          0,
                                          static_cast<const float*>(d_data.GetDeviceBuffer())));

    hip_check_error(hipDeviceSynchronize());
    std::cout << "=====================================\n";

    std::cout << "\n=== Tutorial Complete ===\n";
    std::cout << "You now understand:\n";
    std::cout << "- make_single_stage_tensor_adaptor for basic transformations\n";
    std::cout << "- transform_tensor_adaptor for incremental building\n";
    std::cout << "- chain_tensor_adaptors for composing transformations\n";
    std::cout << "- Padding transform with actual get_vectorized_elements reads\n";
    std::cout << "- Replicate transform with broadcasting\n";
    std::cout << "- Real-world GEMM tiling patterns\n\n";

    return 0;
}
