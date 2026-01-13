// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

#include "../../experimental/builder/test/utils/ckb_conv_tile_test_configs.hpp"
#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"

#include "ck_tile/builder/testing/conv_fwd_ck_tile.hpp"
#include "ck_tile/builder/testing/conv_fwd_reference.hpp"

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

#include "../../experimental/builder/src/grouped_convolution_forward_tile_nhwgc_fp32.inc"
#include "../../experimental/builder/src/grouped_convolution_forward_tile_nhwgc_bf16.inc"
#include "../../experimental/builder/src/grouped_convolution_forward_tile_nhwgc_fp16.inc"
#include "../../experimental/builder/src/grouped_convolution_forward_tile_ndhwgc_fp32.inc"
#include "../../experimental/builder/src/grouped_convolution_forward_tile_ndhwgc_bf16.inc"
#include "../../experimental/builder/src/grouped_convolution_forward_tile_ndhwgc_fp16.inc"

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
    float avg_time;
    bool valid = true;

    auto reference = ckt::alloc_outputs(args);
    using ReferenceInstance =
        typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
    auto ref_conv = ReferenceInstance{};
    ckt::run(ref_conv, args, inputs, reference.get());

    [[maybe_unused]] auto run_alg = [&](auto&& run_alg_func) {
        std::tie(avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(avg_time > 0.f)
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
#include "../../experimental/builder/src/grouped_convolution_forward_tile_nhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_BF16_FWD)
    {
#include "../../experimental/builder/src/grouped_convolution_forward_tile_nhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP32_FWD)
    {
#include "../../experimental/builder/src/grouped_convolution_forward_tile_nhwgc_fp32_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP16_FWD)
    {
#include "../../experimental/builder/src/grouped_convolution_forward_tile_ndhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_BF16_FWD)
    {
#include "../../experimental/builder/src/grouped_convolution_forward_tile_ndhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP32_FWD)
    {
#include "../../experimental/builder/src/grouped_convolution_forward_tile_ndhwgc_fp32_calls.inc"
    }
    else
    {
        std::cout << "Signature not supported" << std::endl;
        return std::make_tuple(false, best_avg_time, best_op_name);
    }
    return std::make_tuple(valid, best_avg_time, best_op_name);
}

} // namespace ck_tile::builder::profiling
