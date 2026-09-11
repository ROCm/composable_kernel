// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Common utilities shared by the 3 dispatcher-based profiler headers.
// Separated from tile_profiler_common.hpp because it depends on
// ck_tile/dispatcher/* headers that aren't available in builder-only builds.

#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/register_all_grouped_conv_kernels.hpp"

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

/// Get dispatcher dtype string from SIGNATURE.
template <auto SIGNATURE>
constexpr const char* get_dtype_string()
{
    if constexpr(SIGNATURE.data_type == ckb::DataType::FP32)
        return "fp32";
    else if constexpr(SIGNATURE.data_type == ckb::DataType::FP16)
        return "fp16";
    else
        return "bf16";
}

/// Get dispatcher layout string from SIGNATURE.
template <auto SIGNATURE>
constexpr const char* get_layout_string()
{
    constexpr auto layout = SIGNATURE.input.config.layout;
    if constexpr(layout == ckb::TensorLayout::NHWGC)
        return "nhwgc";
    else if constexpr(layout == ckb::TensorLayout::NDHWGC)
        return "ndhwgc";
    else if constexpr(layout == ckb::TensorLayout::NGCHW)
        return "ngchw";
    else
        return "unknown";
}

/// Query the runtime GPU architecture name (e.g. "gfx950").
inline std::string get_runtime_arch_name()
{
    hipDeviceProp_t props{};
    int device = 0;
    ck_tile::hip_check_error(hipGetDevice(&device));
    ck_tile::hip_check_error(hipGetDeviceProperties(&props, device));
    std::string name(props.gcnArchName);
    auto pos = name.find(':');
    if(pos != std::string::npos)
        name = name.substr(0, pos);
    return name;
}

/// Convert builder Args to dispatcher GroupedConvProblem.
/// The `op` parameter selects the convolution direction.
template <auto SIGNATURE>
inline ck_tile::dispatcher::GroupedConvProblem args_to_problem(
    const ckt::Args<SIGNATURE>& args, ck_tile::dispatcher::GroupedConvOp op, int split_k = 1)
{
    const auto conv_param = args.to_ck_tile_conv_param();
    ck_tile::dispatcher::GroupedConvProblem problem;

    problem.N       = conv_param.N_;
    problem.C       = conv_param.C_;
    problem.K       = conv_param.K_;
    problem.G       = conv_param.G_;
    problem.op      = op;
    problem.split_k = split_k;

    constexpr int ndim = SIGNATURE.spatial_dim;

    if constexpr(ndim == 2)
    {
        problem.input_spatial = {
            1, conv_param.input_spatial_lengths_[0], conv_param.input_spatial_lengths_[1]};
        problem.filter_spatial = {
            1, conv_param.filter_spatial_lengths_[0], conv_param.filter_spatial_lengths_[1]};
        problem.output_spatial = {
            1, conv_param.output_spatial_lengths_[0], conv_param.output_spatial_lengths_[1]};
        problem.stride = {
            1, conv_param.conv_filter_strides_[0], conv_param.conv_filter_strides_[1]};
        problem.padding  = {0, conv_param.input_left_pads_[0], conv_param.input_left_pads_[1]};
        problem.dilation = {
            1, conv_param.conv_filter_dilations_[0], conv_param.conv_filter_dilations_[1]};
    }
    else if constexpr(ndim == 3)
    {
        problem.input_spatial  = {conv_param.input_spatial_lengths_[0],
                                  conv_param.input_spatial_lengths_[1],
                                  conv_param.input_spatial_lengths_[2]};
        problem.filter_spatial = {conv_param.filter_spatial_lengths_[0],
                                  conv_param.filter_spatial_lengths_[1],
                                  conv_param.filter_spatial_lengths_[2]};
        problem.output_spatial = {conv_param.output_spatial_lengths_[0],
                                  conv_param.output_spatial_lengths_[1],
                                  conv_param.output_spatial_lengths_[2]};
        problem.stride         = {conv_param.conv_filter_strides_[0],
                                  conv_param.conv_filter_strides_[1],
                                  conv_param.conv_filter_strides_[2]};
        problem.padding        = {conv_param.input_left_pads_[0],
                                  conv_param.input_left_pads_[1],
                                  conv_param.input_left_pads_[2]};
        problem.dilation       = {conv_param.conv_filter_dilations_[0],
                                  conv_param.conv_filter_dilations_[1],
                                  conv_param.conv_filter_dilations_[2]};
    }

    return problem;
}

/// Set up the thread-local dispatch buffer context for kernel execution.
inline void setup_dispatch_context(const void* input_ptr,
                                   const void* weight_ptr,
                                   void* output_ptr,
                                   const ck_tile::stream_config& s_conf,
                                   int split_k = 1)
{
    auto& ctx        = ck_tile::dispatcher::g_conv_dispatch_buffers;
    ctx.input_ptr    = input_ptr;
    ctx.weight_ptr   = weight_ptr;
    ctx.output_ptr   = output_ptr;
    ctx.warmup       = s_conf.cold_niters_;
    ctx.repeat       = s_conf.nrepeat_;
    ctx.benchmarking = s_conf.time_kernel_;
    ctx.split_k      = split_k;
}

/// Run a single dispatcher kernel with warmup/dummy-run handling.
/// Returns {is_supported, avg_time}. Updates dummy_run_executed flag.
inline std::tuple<bool, float>
run_kernel_with_warmup(const ck_tile::dispatcher::GroupedConvKernelInstance* kernel,
                       const ck_tile::dispatcher::GroupedConvProblem& problem,
                       const std::string& op_name,
                       const ck_tile::stream_config& s_conf,
                       bool& dummy_run_executed)
{
    bool is_supported = kernel->is_supported(problem);
    if(!is_supported)
        return {false, 0.0f};

    float avg_time{0};

    try
    {
        avg_time     = kernel->run(problem, nullptr);
        is_supported = true;
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "[Exception] " << op_name << " : " << e.what() << std::endl;
        ck_tile::hip_check_error(hipDeviceSynchronize());
        ck_tile::hip_check_error(hipGetLastError());
        return {false, 0.0f};
    }

    if((s_conf.time_kernel_ || s_conf.flush_cache_) && !dummy_run_executed)
    {
        try
        {
            avg_time = kernel->run(problem, nullptr);
        }
        catch(const std::runtime_error& e)
        {
            std::cerr << "[Exception] " << op_name << " : " << e.what() << std::endl;
            ck_tile::hip_check_error(hipDeviceSynchronize());
            ck_tile::hip_check_error(hipGetLastError());
            return {false, 0.0f};
        }
        dummy_run_executed = true;
    }

    return {is_supported, avg_time};
}

} // namespace ck_tile::builder::profiling
