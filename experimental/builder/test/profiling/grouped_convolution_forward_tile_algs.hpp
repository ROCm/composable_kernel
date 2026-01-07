// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

#include "../utils/ckb_conv_tile_test_configs.hpp"
#include "../utils/ckb_conv_test_utils.hpp"
#include "../utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"

#include "ck_tile/builder/testing/conv_fwd_ck_tile.hpp"

#include "ck_tile/host.hpp"

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

/// @brief `run_grouped_conv_forward_tile_algs()` run all grouped conv fwd instances.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see run_grouped_conv_forward_tile_algs()
template <auto SIGNATURE>
std::tuple<float, std::string>
run_grouped_conv_forward_tile_algs(const ckt::Args<SIGNATURE>& args,
                                   const ckt::Inputs<SIGNATURE>& inputs,
                                   const ckt::Outputs<SIGNATURE>& outputs,
                                   const ck_tile::stream_config& s_conf);

#include "grouped_convolution_forward_tile_nhwgc_fp32.inc"
#include "grouped_convolution_forward_tile_nhwgc_bf16.inc"
#include "grouped_convolution_forward_tile_nhwgc_fp16.inc"
#include "grouped_convolution_forward_tile_ndhwgc_fp32.inc"
#include "grouped_convolution_forward_tile_ndhwgc_bf16.inc"
#include "grouped_convolution_forward_tile_ndhwgc_fp16.inc"

template <>
std::tuple<float, std::string> run_grouped_conv_forward_tile_algs<SIGNATURE_NHWGC_FP32_FWD>(
    const ckt::Args<SIGNATURE_NHWGC_FP32_FWD>& args,
    const ckt::Inputs<SIGNATURE_NHWGC_FP32_FWD>& inputs,
    const ckt::Outputs<SIGNATURE_NHWGC_FP32_FWD>& outputs,
    const ck_tile::stream_config& s_conf)
{
    float best_avg_time = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    float avg_time;

    auto ref_conv =
        ckb::ConvBuilder<SIGNATURE_NHWGC_FP32_FWD, ckt::ConvAlgorithm_Reference{}>::Instance{};
    ckt::run(ref_conv, args, inputs.get(), reference.get());

    auto run_alg = [&](auto&& run_alg_func) {
        std::tie(avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        ck_tile::check_err(outputs.get(), reference.get());

        if(avg_time > 0.f)
        {
            best_avg_time = std::min(best_avg_time, avg_time);
            best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms,";
        }
        std::cout << " " << op_name << std::endl;
    };

#include "grouped_convolution_forward_tile_nhwgc_fp32_calls.inc"

    return std::make_tuple(best_avg_time, best_op_name);
}

template <>
std::tuple<float, std::string> run_grouped_conv_forward_tile_algs<SIGNATURE_NHWGC_BF16_FWD>(
    const ckt::Args<SIGNATURE_NHWGC_BF16_FWD>& args,
    const ckt::Inputs<SIGNATURE_NHWGC_BF16_FWD>& inputs,
    const ckt::Outputs<SIGNATURE_NHWGC_BF16_FWD>& outputs,
    const ck_tile::stream_config& s_conf)
{
    float best_avg_time = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    float avg_time;

    auto ref_conv =
        ckb::ConvBuilder<SIGNATURE_NHWGC_BF16_FWD, ckt::ConvAlgorithm_Reference{}>::Instance{};
    ckt::run(ref_conv, args, inputs.get(), reference.get());

    auto run_alg = [&](auto&& run_alg_func) {
        std::tie(avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(avg_time > 0.f)
        {
            best_avg_time = std::min(best_avg_time, avg_time);
            best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms,";
        }
        std::cout << " " << op_name << std::endl;
    };

#include "grouped_convolution_forward_tile_nhwgc_bf16_calls.inc"

    return std::make_tuple(best_avg_time, best_op_name);
}

template <>
std::tuple<float, std::string> run_grouped_conv_forward_tile_algs<SIGNATURE_NHWGC_FP16_FWD>(
    const ckt::Args<SIGNATURE_NHWGC_FP16_FWD>& args,
    const ckt::Inputs<SIGNATURE_NHWGC_FP16_FWD>& inputs,
    const ckt::Outputs<SIGNATURE_NHWGC_FP16_FWD>& outputs,
    const ck_tile::stream_config& s_conf)
{
    float best_avg_time = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    float avg_time;

    auto ref_conv =
        ckb::ConvBuilder<SIGNATURE_NHWGC_FP16_FWD, ckt::ConvAlgorithm_Reference{}>::Instance{};
    ckt::run(ref_conv, args, inputs.get(), reference.get());

    auto run_alg = [&](auto&& run_alg_func) {
        std::tie(avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(avg_time > 0.f)
        {
            best_avg_time = std::min(best_avg_time, avg_time);
            best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms,";
        }
        std::cout << " " << op_name << std::endl;
    };

#include "grouped_convolution_forward_tile_nhwgc_fp16_calls.inc"

    return std::make_tuple(best_avg_time, best_op_name);
}

template <>
std::tuple<float, std::string> run_grouped_conv_forward_tile_algs<SIGNATURE_NDHWGC_FP32_FWD>(
    const ckt::Args<SIGNATURE_NDHWGC_FP32_FWD>& args,
    const ckt::Inputs<SIGNATURE_NDHWGC_FP32_FWD>& inputs,
    const ckt::Outputs<SIGNATURE_NDHWGC_FP32_FWD>& outputs,
    const ck_tile::stream_config& s_conf)
{
    float best_avg_time = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    float avg_time;

    auto ref_conv =
        ckb::ConvBuilder<SIGNATURE_NDHWGC_FP32_FWD, ckt::ConvAlgorithm_Reference{}>::Instance{};
    ckt::run(ref_conv, args, inputs.get(), reference.get());

    auto run_alg = [&](auto&& run_alg_func) {
        std::tie(avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(avg_time > 0.f)
        {
            best_avg_time = std::min(best_avg_time, avg_time);
            best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms,";
        }
        std::cout << " " << op_name << std::endl;
    };

#include "grouped_convolution_forward_tile_ndhwgc_fp32_calls.inc"

    return std::make_tuple(best_avg_time, best_op_name);
}

template <>
std::tuple<float, std::string> run_grouped_conv_forward_tile_algs<SIGNATURE_NDHWGC_BF16_FWD>(
    const ckt::Args<SIGNATURE_NDHWGC_BF16_FWD>& args,
    const ckt::Inputs<SIGNATURE_NDHWGC_BF16_FWD>& inputs,
    const ckt::Outputs<SIGNATURE_NDHWGC_BF16_FWD>& outputs,
    const ck_tile::stream_config& s_conf)
{
    float best_avg_time = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    float avg_time;

    auto ref_conv =
        ckb::ConvBuilder<SIGNATURE_NDHWGC_BF16_FWD, ckt::ConvAlgorithm_Reference{}>::Instance{};
    ckt::run(ref_conv, args, inputs.get(), reference.get());

    auto run_alg = [&](auto&& run_alg_func) {
        std::tie(avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(avg_time > 0.f)
        {
            best_avg_time = std::min(best_avg_time, avg_time);
            best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms,";
        }
        std::cout << " " << op_name << std::endl;
    };

#include "grouped_convolution_forward_tile_ndhwgc_bf16_calls.inc"

    return std::make_tuple(best_avg_time, best_op_name);
}

template <>
std::tuple<float, std::string> run_grouped_conv_forward_tile_algs<SIGNATURE_NDHWGC_FP16_FWD>(
    const ckt::Args<SIGNATURE_NDHWGC_FP16_FWD>& args,
    const ckt::Inputs<SIGNATURE_NDHWGC_FP16_FWD>& inputs,
    const ckt::Outputs<SIGNATURE_NDHWGC_FP16_FWD>& outputs,
    const ck_tile::stream_config& s_conf)
{
    float best_avg_time = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    float avg_time;

    auto ref_conv =
        ckb::ConvBuilder<SIGNATURE_NDHWGC_FP16_FWD, ckt::ConvAlgorithm_Reference{}>::Instance{};
    ckt::run(ref_conv, args, inputs.get(), reference.get());

    auto run_alg = [&](auto&& run_alg_func) {
        std::tie(avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(avg_time > 0.f)
        {
            best_avg_time = std::min(best_avg_time, avg_time);
            best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms,";
        }
        std::cout << " " << op_name << std::endl;
    };

#include "grouped_convolution_forward_tile_ndhwgc_fp16_calls.inc"

    return std::make_tuple(best_avg_time, best_op_name);
}

} // namespace ck_tile::builder::profiling
