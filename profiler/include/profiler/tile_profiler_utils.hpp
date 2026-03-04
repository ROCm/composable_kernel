// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"

namespace ck_tile::builder::profiling {

namespace ckt = ck_tile::builder::test;

template <auto SIGNATURE>
auto parse_conv_args(int arg_idx, char* const argv[])
{
    const std::size_t G = static_cast<size_t>(std::stol(argv[arg_idx++]));
    const std::size_t N = static_cast<size_t>(std::stol(argv[arg_idx++]));
    const std::size_t K = static_cast<size_t>(std::stol(argv[arg_idx++]));
    const std::size_t C = static_cast<size_t>(std::stol(argv[arg_idx++]));

    constexpr auto num_dim_spatial = SIGNATURE.spatial_dim;

    std::vector<std::size_t> filter_spatial_lengths(num_dim_spatial);
    std::vector<std::size_t> input_spatial_lengths(num_dim_spatial);
    std::vector<std::size_t> conv_filter_strides(num_dim_spatial);
    std::vector<std::size_t> conv_filter_dilations(num_dim_spatial);
    std::vector<std::size_t> input_left_pads(num_dim_spatial);
    std::vector<std::size_t> input_right_pads(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        filter_spatial_lengths[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_spatial_lengths[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_strides[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_dilations[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_left_pads[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_right_pads[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = N,
                .groups          = G,
                .input_channels  = C,
                .output_channels = K,
                .image  = ckt::filter_extent_from_vector<num_dim_spatial>(input_spatial_lengths),
                .filter = ckt::filter_extent_from_vector<num_dim_spatial>(filter_spatial_lengths),
            },
        .filter_strides   = ckt::filter_extent_from_vector<num_dim_spatial>(conv_filter_strides),
        .filter_dilation  = ckt::filter_extent_from_vector<num_dim_spatial>(conv_filter_dilations),
        .input_left_pad   = ckt::filter_extent_from_vector<num_dim_spatial>(input_left_pads),
        .input_right_pad  = ckt::filter_extent_from_vector<num_dim_spatial>(input_right_pads),
        .a_elementwise_op = {},
        .b_elementwise_op = {},
        .cde_elementwise_op = {},
    };
    return args;
}

} // namespace ck_tile::builder::profiling
