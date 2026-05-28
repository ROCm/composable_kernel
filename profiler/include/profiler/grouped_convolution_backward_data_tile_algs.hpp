// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <tuple>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_data_gpu.hpp"

#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"
#include "tile_profiler_common.hpp"
#include "tile_profiler_utils.hpp"

namespace ck_tile::builder::profiling {

#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_ndhwgc_fp32.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_nhwgc_fp32.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_nhwgc_bf16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_nhwgc_fp16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_ndhwgc_bf16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_ndhwgc_fp16.inc"

/// @brief `run_grouped_conv_backward_data_tile_algs()` run all grouped conv bwd data instances.
///
/// @tparam SIGNATURE Backward data convolution signature.
///
/// @see run_grouped_conv_backward_data_tile_algs()
template <auto SIGNATURE>
std::tuple<bool, float, std::string, int, int>
run_grouped_conv_backward_data_tile_algs(const ckt::Args<SIGNATURE>& args,
                                         const std::string& split_k,
                                         const index_t instance_index,
                                         const ckt::Inputs<SIGNATURE>& inputs,
                                         const ckt::Outputs<SIGNATURE>& outputs,
                                         const ck_tile::stream_config& s_conf,
                                         bool do_verification = true)
{
    using DataType = DeduceDataType<SIGNATURE>;

    // Run first instance as dummy to get proper time from the first instance
    bool dummy_run_executed = false;
    float best_avg_time     = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    int best_split_k                = 0;
    ck::index_t best_instance_index = -1;
    bool is_supported               = false;
    float avg_time;
    bool all_instances_valid = true;

    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;

    const auto conv_param       = args.to_ck_tile_conv_param();
    float max_accumulated_value = 0.f;
    auto reference              = ckt::alloc_outputs(args);
    if(do_verification)
    {
        using ReferenceInstance =
            typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
        auto ref_conv                    = ReferenceInstance{};
        [[maybe_unused]] auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());

        // Get max possible value in the output
        const std::size_t input_bytes_num = conv_param.template GetInputByte<DataType>();
        std::vector<DataType> ref(input_bytes_num / sizeof(DataType));
        HIP_CHECK_ERROR(hipMemcpy(
            &ref.data()[0], reference.get().input, input_bytes_num, hipMemcpyDeviceToHost));
        max_accumulated_value = *std::max_element(ref.begin(), ref.end());
    }

    const index_t num_accums = conv_param.K_;

    // BWD data doesn't support split-K autodeduce value -1
    auto split_k_values = get_split_k_values(split_k);
    split_k_values.erase(std::remove(split_k_values.begin(), split_k_values.end(), -1),
                         split_k_values.end());

    index_t num_kernel = 0;
    auto run_alg       = [&](auto&& run_alg_func) {
        num_kernel++;
        // Skip if a specific instance was requested and this isn't it
        const bool running_specific_instance = (instance_index != -1);
        const bool current_is_target         = (num_kernel - 1 == instance_index);
        if(running_specific_instance && !current_is_target)
        {
            return;
        }

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
                    // Run first instance twice
                    std::tie(is_supported, avg_time, op_name) =
                        run_alg_func(args_k_batch, inputs, outputs, s_conf);
                    dummy_run_executed = true;
                }
                bool valid = true;
                if(do_verification)
                {
                    ckt::ValidationReport report;
                    auto&& [rtol, atol] =
                        get_rtol_atol<SIGNATURE>(num_accums, k_batch, max_accumulated_value);
                    ckt::Outputs<SIGNATURE>::reflect(
                        args_k_batch,
                        [&](std::string_view name,
                            const auto& desc,
                            void* ckt::Outputs<SIGNATURE>::*ptr) {
                            report.check(
                                name, desc, outputs.*ptr, reference.get().*ptr, rtol, atol);
                        });

                    valid = report.get_errors().empty();
                    if(!valid)
                    {
                        std::cout << "[Error] " << op_name << ", SplitK " << k_batch << std::endl;
                        for(const auto& error : report.get_errors())
                        {
                            std::cout << "\tNumber of incorrect values: " << error.wrong_elements
                                      << " Is all zero:" << error.is_all_zero()
                                      << " max err: " << error.max_error << std::endl;
                            run_cpu_validation<SIGNATURE, ConvBuffer::Input>(
                                args_k_batch, outputs, reference.get());
                        }
                        all_instances_valid = false;
                    }
                }
                if(valid)
                {
                    if(avg_time < best_avg_time)
                    {
                        best_instance_index = num_kernel - 1;
                    }
                    best_avg_time = std::min(best_avg_time, avg_time);
                    best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
                    best_split_k  = best_avg_time < avg_time ? best_split_k : k_batch;
                    std::cout << "[Valid] Perf: " << std::setw(10) << avg_time << " ms," << " "
                              << op_name << " (instance " << num_kernel - 1 << "), SplitK "
                              << k_batch << std::endl;
                }
            }
            else
            {
                std::cout << "[Not supported] " << op_name << ", SplitK " << k_batch << std::endl;
            }
        }
    };

    if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP16_BWD_DATA)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_nhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_BF16_BWD_DATA)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_nhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP32_BWD_DATA)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_nhwgc_fp32_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP16_BWD_DATA)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_ndhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_BF16_BWD_DATA)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_ndhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP32_BWD_DATA)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_data/grouped_convolution_backward_data_tile_ndhwgc_fp32_calls.inc"
    }
    else
    {
        std::cout << "Signature not supported" << std::endl;
        return std::make_tuple(
            false, best_avg_time, best_op_name, best_split_k, best_instance_index);
    }
    return std::make_tuple(
        all_instances_valid, best_avg_time, best_op_name, best_split_k, best_instance_index);
}

} // namespace ck_tile::builder::profiling
