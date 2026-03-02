// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file tile_distribution_encoding_register_mapper.hpp
 * @brief Utility for register / matrix coordinate mapping from TileDistributionEncoding
 * @details Defines TileDistrEncRegMap, which takes a TileDistributionEncoding and provides
 * functions for mapping matrix fragment coordinates to register coordinates (lane, vector item) and
 * vice versa. This is only meant for tile distributions encodings that describe register mappings.
 *
 * A repeat dimension is allowed in which case multiple (lane, vector item) pairs are mapped to the
 * same matrix coordinates. The inverse map takes a "repeat index" to distinguish between them.
 *
 * print() functions are included for printing dimensions and formatted forward and backwards
 * mappings similar to the AMD Matrix Calculator.
 */

#pragma once

#include <stdio.h>
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

namespace ck_tile::core::arch::mma {

// Utility to calculate register mappings from a Tile Distribution Encoding.
template <typename TileDistrEnc>
struct TileDistrEncRegMap
{
    // Make sure this is a proper Tile Distr Encoding for Lane Vector mapping.
    static_assert(TileDistrEnc::NDimR <= 1);
    static_assert(TileDistrEnc::NDimX == 2);
    static_assert(TileDistrEnc::NDimP == 1);

    static constexpr auto ps_ys_to_xs_adaptor =
        make_static_tile_distribution(TileDistrEnc{}).get_ps_ys_to_xs_adaptor();

    static constexpr index_t mat_major_size =
        container_reduce(typename TileDistrEnc::HsLengthss{}[number<0>{}], multiplies<>{}, 1);
    static constexpr index_t mat_minor_size =
        container_reduce(typename TileDistrEnc::HsLengthss{}[number<1>{}], multiplies<>{}, 1);
    static constexpr index_t num_repeat = [] {
        if constexpr(TileDistrEnc::NDimR > 0)
        {
            return typename TileDistrEnc::RsLengths{}[number<0>{}];
        }
        else
        {
            return 1; // Necessary to deal with empty "repeat" sequences.
        }
    }();
    static constexpr index_t num_lanes = ps_ys_to_xs_adaptor.get_top_dimension_length(number<0>{});
    static constexpr index_t num_vector_items =
        container_reduce(TileDistrEnc::detail::ys_lengths_, multiplies<>{}, 1);

    // Check for 0 dims (will break things much earlier but let's have an extra check).
    static_assert(mat_major_size > 0);
    static_assert(mat_minor_size > 0);
    static_assert(num_repeat > 0);
    static_assert(num_lanes > 0);
    static_assert(num_vector_items > 0);

    CK_TILE_HOST_DEVICE static constexpr auto
    calc_matrix_indices_from_lane_vector(index_t lane_inx, index_t vector_inx)
    {
        // For some reason the Y dimension is not treated the same as the P dimension and we need to
        // manually unmerge the Y dimension index into its hidden indices before being able to use
        // it...
        array<index_t, TileDistrEnc::NDimY> y_hidden_inx;
        for(index_t i = TileDistrEnc::NDimY - 1; i >= 0; --i)
        {
            y_hidden_inx[i] = vector_inx % TileDistrEnc::detail::ys_lengths_[i];
            vector_inx /= TileDistrEnc::detail::ys_lengths_[i];
        }

        const auto ps_ys_idx = container_concat(array<index_t, 1>{lane_inx}, y_hidden_inx);
        return ps_ys_to_xs_adaptor.calculate_bottom_index(ps_ys_idx);
    }

    struct LaneVec
    {
        index_t lane = -1; // Sentinel for invalid pairs
        index_t vec  = -1;
    };

    using InverseMap =
        std::array<std::array<std::array<LaneVec, num_repeat>, mat_minor_size>, mat_major_size>;

    // TODO: In theory this could be done with inverted merge unmerge operations.
    CK_TILE_HOST_DEVICE static constexpr InverseMap calc_inverse_map()
    {
        InverseMap im{};
        for(index_t l = 0; l < num_lanes; ++l)
        {
            for(index_t v = 0; v < num_vector_items; ++v)
            {
                auto res = calc_matrix_indices_from_lane_vector(l, v); // Matrix major, minor inx;

                // We assume that repeated matrix elements appear at increasing L and V indices.
                for(index_t r = 0; r < num_repeat; r++)
                {
                    auto& lv = im[res[0]][res[1]][r];
                    if(lv.lane < 0)
                    {
                        lv.lane = l; // TODO: c++20 designated initializers
                        lv.vec  = v;
                    }
                }
            }
        }
        return im;
    }

    CK_TILE_HOST_DEVICE static void print_dims()
    {
        printf("Matrix dims major, minor, repeat = %d %d %d\n",
               mat_major_size,
               mat_minor_size,
               num_repeat);
        printf("Num lanes, vector items = %d %d\n", num_lanes, num_vector_items);
    }

    CK_TILE_HOST_DEVICE static void print_mapping()
    {
        printf("(lane, vector) item to matrix element\n L | ");
        for(index_t v = 0; v < num_vector_items; v++)
        {
            printf("vec%2d | ", v);
        }
        printf("\n");

        for(index_t l = 0; l < num_lanes; l++)
        {
            printf("%2d | ", l);
            for(index_t v = 0; v < num_vector_items; v++)
            {
                auto res = calc_matrix_indices_from_lane_vector(l, v);
                printf("%2d %2d | ", res[0], res[1]);
            }
            printf("\n");
        }
    }

    CK_TILE_HOST_DEVICE static void print_inverse_mapping()
    {
        InverseMap im = calc_inverse_map();
        printf("Matrix element to (lane, vector item). Elements are replicated an additional %d "
               "time(s) in higher lanes. \n",
               num_repeat - 1);
        printf("Mat| ");
        for(index_t k = 0; k < mat_minor_size; k++)
        {
            printf("   %2d | ", k);
        }
        printf("\n");

        for(index_t m = 0; m < mat_major_size; m++)
        {
            printf("%2d | ", m);
            for(index_t k = 0; k < mat_minor_size; k++)
            {
                printf("%2d %2d | ", im[m][k][0].lane, im[m][k][0].vec);
            }
            printf("\n");
        }
    }

    CK_TILE_HOST_DEVICE static void print()
    {
        print_dims();
        print_mapping();
        print_inverse_mapping();
    }
};
} // namespace ck_tile::core::arch::mma
