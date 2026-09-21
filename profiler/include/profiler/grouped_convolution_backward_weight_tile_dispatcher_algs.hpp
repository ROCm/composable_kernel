// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Dispatcher-based backward weight profiler header.
// Drop-in replacement for grouped_convolution_backward_weight_tile_algs.hpp
// that uses the CK Dispatcher registry instead of CK Builder .inc files.

#pragma once

#include <iostream>
#include <tuple>

#include "grouped_convolution_signatures.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"
#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "tile_profiler_common.hpp"
#include "tile_profiler_dispatcher_common.hpp"
#include "tile_profiler_utils.hpp"

namespace ck_tile::builder::profiling {

/// @brief Dispatcher-based `run_grouped_conv_backward_weight_tile_algs()`.
/// Iterates all registered dispatcher kernels instead of builder-generated .inc files.
template <auto SIGNATURE>
std::tuple<bool, float, std::string, int>
run_grouped_conv_backward_weight_tile_algs(const ckt::Args<SIGNATURE>& args,
                                           const std::string& split_k,
                                           const ckt::Inputs<SIGNATURE>& inputs,
                                           const ckt::Outputs<SIGNATURE>& outputs,
                                           const ck_tile::stream_config& s_conf,
                                           bool do_verification = true)
{
    using DataType = DeduceDataType<SIGNATURE>;

    bool dummy_run_executed = false;
    float best_avg_time     = std::numeric_limits<float>::max();
    std::string best_op_name;
    int best_split_k         = 1;
    bool all_instances_valid = true;

    auto reference = ckt::alloc_outputs(args);
    if(do_verification)
    {
        reference = compute_reference<SIGNATURE>(args, inputs);
    }

    const auto conv_param = args.to_ck_tile_conv_param();

    float max_accumulated_value = 0.f;
    if(do_verification)
    {
        // Get max possible value in the output for tolerance calculation
        const std::size_t weight_bytes_num = conv_param.template GetWeightByte<DataType>();
        std::vector<DataType> ref(weight_bytes_num / sizeof(DataType));
        HIP_CHECK_ERROR(hipMemcpy(
            &ref.data()[0], reference.get().weight, weight_bytes_num, hipMemcpyDeviceToHost));
        max_accumulated_value = *std::max_element(ref.begin(), ref.end());
    }
    const index_t num_accums = std::accumulate(std::begin(conv_param.output_spatial_lengths_),
                                               std::end(conv_param.output_spatial_lengths_),
                                               static_cast<std::size_t>(1),
                                               std::multiplies<std::size_t>()) *
                               conv_param.N_;
    const auto split_k_values = get_split_k_values(split_k);

    // Register all generated backward weight kernels
    static bool kernels_registered = false;
    if(!kernels_registered)
    {
        const auto arch_name = get_runtime_arch_name();
        ck_tile::dispatcher::register_all_grouped_conv_bwd_weight_kernels(arch_name);
        kernels_registered = true;
    }

    // Get backward weight kernels matching data type, spatial dims, and layout
    constexpr const char* dtype_str  = get_dtype_string<SIGNATURE>();
    constexpr const char* layout_str = get_layout_string<SIGNATURE>();
    constexpr int ndim               = SIGNATURE.spatial_dim;
    auto& registry                   = ck_tile::dispatcher::GroupedConvRegistry::instance();
    auto all_kernels = registry.filter([](const ck_tile::dispatcher::GroupedConvKernelInstance& k) {
        return k.key().op == ck_tile::dispatcher::GroupedConvOp::BackwardWeight &&
               k.key().dtype_in == dtype_str && k.key().ndim_spatial == ndim &&
               k.key().layout == layout_str;
    });

    // Set up thread-local buffer context
    setup_dispatch_context(inputs.input, inputs.output, outputs.weight, s_conf);

    constexpr bool use_instance_string = true;

    // Iterate all kernels x split-K values
    for(const auto* kernel : all_kernels)
    {
        for(auto& k_batch : split_k_values)
        {
            auto problem = args_to_problem<SIGNATURE>(
                args, ck_tile::dispatcher::GroupedConvOp::BackwardWeight, k_batch);
            ck_tile::dispatcher::g_conv_dispatch_buffers.split_k = k_batch;

            std::string op_name = kernel->name(use_instance_string);

            auto [is_supported, avg_time] =
                run_kernel_with_warmup(kernel, problem, op_name, s_conf, dummy_run_executed);

            if(!is_supported)
            {
                std::cout << "[Not supported] " << op_name << ", SplitK " << k_batch << std::endl;
                continue;
            }

            bool valid = true;
            if(do_verification)
            {
                auto&& [rtol, atol] =
                    get_rtol_atol<SIGNATURE>(num_accums, k_batch, max_accumulated_value);
                valid = validate_and_report<SIGNATURE, ConvBuffer::Weight>(
                    args, outputs, reference.get(), rtol, atol);
                if(!valid)
                {
                    std::cout << "[Error] " << op_name << ", SplitK " << k_batch << std::endl;
                    all_instances_valid = false;
                }
            }
            if(avg_time < best_avg_time)
            {
                best_avg_time = avg_time;
                best_op_name  = op_name;
                best_split_k  = k_batch;
            }
            if(valid)
            {
                const char* prefix = do_verification ? "[Valid]" : "[Not Validated]";
                std::cout << prefix << " Perf: " << std::setw(10) << avg_time << " ms," << " "
                          << op_name << ", SplitK " << k_batch << std::endl;
            }
        }
    }

    return std::make_tuple(all_instances_valid, best_avg_time, best_op_name, best_split_k);
}

} // namespace ck_tile::builder::profiling
