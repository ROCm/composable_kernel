// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <tuple>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"

#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"
#include "tile_profiler_utils.hpp"

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp32.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp32.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_bf16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_bf16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp16.inc"

#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp32_streamk.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp32_streamk.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_bf16_streamk.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp16_streamk.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_bf16_streamk.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp16_streamk.inc"

template <auto SIGNATURE>
void run_cpu_validation(const ckt::Args<SIGNATURE>& args,
                        const ckt::Outputs<SIGNATURE>& outputs,
                        const ckt::Outputs<SIGNATURE>& reference)
{
    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;
    const auto conv_param = args.to_ck_tile_conv_param();

    const std::size_t weight_bytes_num = conv_param.template GetWeightByte<DataType>();
    std::vector<DataType> wei(weight_bytes_num / sizeof(DataType));
    std::vector<DataType> ref(weight_bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(
        hipMemcpy(&ref.data()[0], reference.weight, weight_bytes_num, hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(
        hipMemcpy(&wei.data()[0], outputs.weight, weight_bytes_num, hipMemcpyDeviceToHost));
    ck_tile::check_err(wei, ref, "\tError: Incorrect results!");
}

/// @brief `run_grouped_conv_backward_weight_tile_algs()` run all grouped conv fwd instances.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see run_grouped_conv_backward_weight_tile_algs()
template <auto SIGNATURE>
std::tuple<bool, float, std::string, int>
run_grouped_conv_backward_weight_tile_algs(const ckt::Args<SIGNATURE>& args,
                                           const std::string& split_k,
                                           const ckt::Inputs<SIGNATURE>& inputs,
                                           const ckt::Outputs<SIGNATURE>& outputs,
                                           const ck_tile::stream_config& s_conf)
{
    bool dummy_run_executed = false;
    float best_avg_time     = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    int best_split_k;
    bool is_supported;
    float avg_time;
    bool all_instances_valid = true;

    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;

    auto reference = ckt::alloc_outputs(args);
    using ReferenceInstance =
        typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
    auto ref_conv   = ReferenceInstance{};
    auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());

    const auto conv_param = args.to_ck_tile_conv_param();

    // Get max possible value in the output
    const std::size_t weight_bytes_num = conv_param.template GetWeightByte<DataType>();
    std::vector<DataType> ref(weight_bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(
        hipMemcpy(&ref.data()[0], reference.get().weight, weight_bytes_num, hipMemcpyDeviceToHost));
    const float max_accumulated_value = *std::max_element(ref.begin(), ref.end());
    const index_t num_accums = std::accumulate(std::begin(conv_param.output_spatial_lengths_),
                                               std::end(conv_param.output_spatial_lengths_),
                                               static_cast<std::size_t>(1),
                                               std::multiplies<std::size_t>()) *
                               conv_param.N_;
    const auto split_k_values = get_split_k_values(split_k);

    auto run_alg = [&](auto&& run_alg_func) {
        for(auto& k_batch : split_k_values)
        {
            ckt::Args<SIGNATURE> args_k_batch = args;
            args_k_batch.k_batch              = k_batch;
            std::tie(is_supported, avg_time, op_name) =
                run_alg_func(args_k_batch, inputs, outputs, s_conf);
            if(is_supported)
            {
                if((s_conf.time_kernel_ || s_conf.flush_cache_) && !dummy_run_executed)
                {
                    // Run first instance twice when profiling to stabilize timing
                    std::tie(is_supported, avg_time, op_name) =
                        run_alg_func(args_k_batch, inputs, outputs, s_conf);
                    dummy_run_executed = true;
                }
                ckt::ValidationReport report;
                auto&& [rtol, atol] =
                    get_rtol_atol<SIGNATURE>(num_accums, k_batch, max_accumulated_value);
                ckt::Outputs<SIGNATURE>::reflect(
                    args_k_batch,
                    [&](std::string_view name,
                        const auto& desc,
                        void* ckt::Outputs<SIGNATURE>::*ptr) {
                        report.check(name, desc, outputs.*ptr, reference.get().*ptr, rtol, atol);
                    });

                const bool valid = report.get_errors().empty();
                best_avg_time    = std::min(best_avg_time, avg_time);
                best_op_name     = best_avg_time < avg_time ? best_op_name : op_name;
                best_split_k     = best_avg_time < avg_time ? best_split_k : k_batch;
                if(valid)
                {
                    std::cout << "[Valid] Perf: " << std::setw(10) << avg_time << " ms," << " "
                              << op_name << ", SplitK " << k_batch << std::endl;
                }
                else
                {
                    std::cout << "[Error] " << op_name << ", SplitK " << k_batch << std::endl;
                    for(const auto& error : report.get_errors())
                    {
                        std::cout << "\tNumber of incorrect values: " << error.wrong_elements
                                  << " Is all zero:" << error.is_all_zero()
                                  << " max err: " << error.max_error << std::endl;
                        // Check with cpu verification to get a values
                        run_cpu_validation<SIGNATURE>(args_k_batch, outputs, reference.get());
                    }
                    all_instances_valid = false;
                }
            }
            else
            {
                std::cout << "[Not supported] " << op_name << ", SplitK " << k_batch << std::endl;
            }
        }
    };

    if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP16_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp16_calls.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp16_streamk_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_BF16_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_bf16_calls.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_bf16_streamk_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP32_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp32_calls.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp32_streamk_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP16_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp16_calls.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp16_streamk_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_BF16_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_bf16_calls.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_bf16_streamk_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP32_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp32_calls.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp32_streamk_calls.inc"
    }
    else
    {
        std::cout << "Signature not supported" << std::endl;
        return std::make_tuple(false, best_avg_time, best_op_name, best_split_k);
    }
    return std::make_tuple(all_instances_valid, best_avg_time, best_op_name, best_split_k);
}

} // namespace ck_tile::builder::profiling
