// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Dispatcher-based forward profiler header.
// Drop-in replacement for grouped_convolution_forward_tile_algs.hpp
// that uses the CK Dispatcher registry instead of CK Builder .inc files.

#pragma once

#include <iostream>
#include <tuple>

#include "grouped_convolution_signatures.hpp"
#include "common.hpp"
#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"
#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "tile_profiler_common.hpp"
#include "tile_profiler_dispatcher_common.hpp"
#include "tile_profiler_utils.hpp"

namespace ck_tile::builder::profiling {

/// @brief Dispatcher-based `run_grouped_conv_forward_tile_algs()`.
/// Iterates all registered dispatcher kernels instead of builder-generated .inc files.
template <auto SIGNATURE>
std::tuple<bool, float, std::string>
run_grouped_conv_forward_tile_algs(const ckt::Args<SIGNATURE>& args,
                                   const ckt::Inputs<SIGNATURE>& inputs,
                                   const ckt::Outputs<SIGNATURE>& outputs,
                                   const ck_tile::stream_config& s_conf,
                                   bool do_verification = true)
{
    using DataType = DeduceDataType<SIGNATURE>;

    bool dummy_run_executed = false;
    float best_avg_time     = std::numeric_limits<float>::max();
    std::string best_op_name;
    bool valid = true;

    auto reference = ckt::alloc_outputs(args);
    if(do_verification)
    {
        reference = compute_reference<SIGNATURE>(args, inputs);
    }

    // Register all generated forward kernels
    static bool kernels_registered = false;
    if(!kernels_registered)
    {
        const auto arch_name = get_runtime_arch_name();
        ck_tile::dispatcher::register_all_grouped_conv_fwd_kernels(arch_name);
        kernels_registered = true;
    }

    // Get forward kernels matching data type, spatial dims, and layout
    constexpr const char* dtype_str  = get_dtype_string<SIGNATURE>();
    constexpr const char* layout_str = get_layout_string<SIGNATURE>();
    constexpr int ndim               = SIGNATURE.spatial_dim;
    auto& registry                   = ck_tile::dispatcher::GroupedConvRegistry::instance();
    auto all_kernels = registry.filter([](const ck_tile::dispatcher::GroupedConvKernelInstance& k) {
        return k.key().op == ck_tile::dispatcher::GroupedConvOp::Forward &&
               k.key().dtype_in == dtype_str && k.key().ndim_spatial == ndim &&
               k.key().layout == layout_str;
    });

    // Set up thread-local buffer context
    setup_dispatch_context(inputs.input, inputs.weight, outputs.output, s_conf);

    auto problem = args_to_problem<SIGNATURE>(args, ck_tile::dispatcher::GroupedConvOp::Forward);

    constexpr bool use_instance_string = true;

    for(const auto* kernel : all_kernels)
    {
        std::string op_name = kernel->name(use_instance_string);

        auto [is_supported, avg_time] =
            run_kernel_with_warmup(kernel, problem, op_name, s_conf, dummy_run_executed);

        if(!is_supported)
        {
            std::cout << "[Not supported] " << op_name << std::endl;
            continue;
        }

        if(avg_time < best_avg_time)
        {
            best_avg_time = avg_time;
            best_op_name  = op_name;
        }
        std::cout << "Perf: " << std::setw(10) << avg_time << " ms," << " " << op_name << std::endl;

        if(do_verification &&
           !validate_and_report<SIGNATURE, ConvBuffer::Output>(args,
                                                               outputs,
                                                               reference.get(),
                                                               ck::profiler::get_rtol<DataType>(),
                                                               ck::profiler::get_atol<DataType>()))
        {
            valid = false;
        }
    }

    return std::make_tuple(valid, best_avg_time, best_op_name);
}

} // namespace ck_tile::builder::profiling
