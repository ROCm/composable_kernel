// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"

#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_fp32.inc"
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_bf16.inc"
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_fp16.inc"
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_ndhwgc_fp32.inc"
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_ndhwgc_bf16.inc"
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_ndhwgc_fp16.inc"

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

/// @brief `run_grouped_conv_forward_tile_algs()` run all grouped conv fwd instances.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see run_grouped_conv_forward_tile_algs()
template <auto SIGNATURE>
std::tuple<bool, float, std::string>
run_grouped_conv_forward_tile_algs(const ckt::Args<SIGNATURE>& args,
                                   const ckt::Inputs<SIGNATURE>& inputs,
                                   const ckt::Outputs<SIGNATURE>& outputs,
                                   const ck_tile::stream_config& s_conf)
{
    float best_avg_time = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    bool is_supported;
    float avg_time;
    bool valid = true;

    auto reference = ckt::alloc_outputs(args);
    using ReferenceInstance =
        typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
    auto ref_conv                    = ReferenceInstance{};
    [[maybe_unused]] auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());

    [[maybe_unused]] auto run_alg = [&](auto&& run_alg_func) {
        std::tie(is_supported, avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(is_supported)
        {
            const auto errors = ckt::validate(args, outputs, reference.get()).get_errors();
            for(const auto& error : errors)
            {
                valid = false;
                std::cout << "Number of incorrect values: " << error.wrong_elements
                          << " Is all zero:" << error.is_all_zero() << std::endl;
            }
            best_avg_time = std::min(best_avg_time, avg_time);
            best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms,";
        }
        std::cout << " " << op_name << std::endl;
    };

    if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP16_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_BF16_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP32_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_nhwgc_fp32_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP16_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_ndhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_BF16_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_ndhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP32_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/grouped_convolution_forward_tile_ndhwgc_fp32_calls.inc"
    }
    else
    {
        std::cout << "Signature not supported" << std::endl;
        return std::make_tuple(false, best_avg_time, best_op_name);
    }
    return std::make_tuple(valid, best_avg_time, best_op_name);
}

} // namespace ck_tile::builder::profiling
