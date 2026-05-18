// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"

namespace ck_tile::builder::profiling {

namespace ckt = ck_tile::builder::test;

inline std::vector<int> get_split_k_values(const std::string& split_k)
{
    std::vector<int> split_k_list = {/*auto deduce value*/ -1, 1, 2, 4, 8, 16, 32, 64, 128};

    if(split_k != "all")
    {
        try
        {
            int split_k_value = std::stoi(split_k);
            split_k_list      = {split_k_value};
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
    return split_k_list;
}

template <auto SIGNATURE>
inline std::tuple<double, double>
get_rtol_atol(const int num_accums, const int k_batch, const float max_accumulated_value)
{
    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;
    using ComputeType = DataType;
    using AccDataType = float;

    // Assign middle value of the range for auto deduce
    const int num_accums_split_k = k_batch > 0 ? k_batch : 64;
    auto rtol = ck_tile::get_relative_threshold<ComputeType, DataType, AccDataType>(
        num_accums / num_accums_split_k);
    auto atol = ck_tile::get_absolute_threshold<ComputeType, DataType, AccDataType>(
        max_accumulated_value / num_accums_split_k, num_accums / num_accums_split_k);
    // Calculate error due to split_k accumulation
    auto rtol_split_k =
        ck_tile::get_relative_threshold<DataType, DataType, DataType>(num_accums_split_k);
    auto atol_split_k = ck_tile::get_absolute_threshold<DataType, DataType, DataType>(
        max_accumulated_value, num_accums_split_k);
    // Use higher threshold
    rtol = std::max(rtol, rtol_split_k);
    atol = std::max(atol, atol_split_k);
    return std::make_tuple(rtol, atol);
}

template <auto SIGNATURE>
inline ckt::Args<SIGNATURE> parse_conv_args(int arg_idx, char* const argv[])
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
