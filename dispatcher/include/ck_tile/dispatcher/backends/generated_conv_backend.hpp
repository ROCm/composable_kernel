// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Generated Convolution Kernel Backend
//
// Wraps CK Tile grouped convolution launchers for use through the
// GroupedConvDispatcher.  Each generated kernel launcher is wrapped in
// a ConvKernelRunFn that builds the correct host-args type (forward,
// bwd-data, or bwd-weight) and calls Launcher::launch().

#pragma once

#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include <hip/hip_runtime.h>
#include <functional>

namespace ck_tile {
namespace dispatcher {
namespace backends {

// Buffer context is defined in grouped_conv_registry.hpp (g_conv_dispatch_buffers)
// so there's no circular dependency.

// Helper: build ck_tile::conv::ConvParam from GroupedConvProblem
inline ck_tile::conv::ConvParam make_conv_param_2d(const GroupedConvProblem& p)
{
    return ck_tile::conv::ConvParam{
        2,
        static_cast<ck_tile::index_t>(p.G),
        static_cast<ck_tile::index_t>(p.N),
        static_cast<ck_tile::index_t>(p.K),
        static_cast<ck_tile::index_t>(p.C),
        {static_cast<ck_tile::index_t>(p.filter_spatial[1]),
         static_cast<ck_tile::index_t>(p.filter_spatial[2])},
        {static_cast<ck_tile::index_t>(p.input_spatial[1]),
         static_cast<ck_tile::index_t>(p.input_spatial[2])},
        {static_cast<ck_tile::index_t>(p.stride[1]), static_cast<ck_tile::index_t>(p.stride[2])},
        {static_cast<ck_tile::index_t>(p.dilation[1]),
         static_cast<ck_tile::index_t>(p.dilation[2])},
        {static_cast<ck_tile::index_t>(p.padding[1]), static_cast<ck_tile::index_t>(p.padding[2])},
        {static_cast<ck_tile::index_t>(p.padding[1]), static_cast<ck_tile::index_t>(p.padding[2])}};
}

inline ck_tile::conv::ConvParam make_conv_param_3d(const GroupedConvProblem& p)
{
    return ck_tile::conv::ConvParam{3,
                                    static_cast<ck_tile::index_t>(p.G),
                                    static_cast<ck_tile::index_t>(p.N),
                                    static_cast<ck_tile::index_t>(p.K),
                                    static_cast<ck_tile::index_t>(p.C),
                                    {static_cast<ck_tile::index_t>(p.filter_spatial[0]),
                                     static_cast<ck_tile::index_t>(p.filter_spatial[1]),
                                     static_cast<ck_tile::index_t>(p.filter_spatial[2])},
                                    {static_cast<ck_tile::index_t>(p.input_spatial[0]),
                                     static_cast<ck_tile::index_t>(p.input_spatial[1]),
                                     static_cast<ck_tile::index_t>(p.input_spatial[2])},
                                    {static_cast<ck_tile::index_t>(p.stride[0]),
                                     static_cast<ck_tile::index_t>(p.stride[1]),
                                     static_cast<ck_tile::index_t>(p.stride[2])},
                                    {static_cast<ck_tile::index_t>(p.dilation[0]),
                                     static_cast<ck_tile::index_t>(p.dilation[1]),
                                     static_cast<ck_tile::index_t>(p.dilation[2])},
                                    {static_cast<ck_tile::index_t>(p.padding[0]),
                                     static_cast<ck_tile::index_t>(p.padding[1]),
                                     static_cast<ck_tile::index_t>(p.padding[2])},
                                    {static_cast<ck_tile::index_t>(p.padding[0]),
                                     static_cast<ck_tile::index_t>(p.padding[1]),
                                     static_cast<ck_tile::index_t>(p.padding[2])}};
}

// Create a RunFn for a forward convolution launcher (2D or 3D)
template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::RunFn make_conv_fwd_run_fn()
{
    return [](const GroupedConvProblem& problem, void* stream) -> float {
        auto& ctx  = g_conv_dispatch_buffers;
        auto param = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        ck_tile::GroupedConvFwdHostArgs<> args(
            param, ctx.input_ptr, ctx.weight_ptr, {}, ctx.output_ptr, 1);
        ck_tile::stream_config sc;
        sc.stream_id_    = reinterpret_cast<hipStream_t>(stream);
        sc.time_kernel_  = ctx.benchmarking;
        sc.log_level_    = 0;
        sc.cold_niters_  = ctx.benchmarking ? ctx.warmup : 0;
        sc.nrepeat_      = ctx.benchmarking ? ctx.repeat : 1;
        sc.is_gpu_timer_ = ctx.benchmarking;
        return LauncherType::launch(args, sc);
    };
}

// Create a RunFn for a backward-data convolution launcher.
// Dispatcher convention: run(dY, W, dX, problem) where dX is computed.
// BwdDataHostArgs(param, in_ptr=dX, wei_ptr=W, {}, out_ptr=dY, k_batch)
template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::RunFn make_conv_bwd_data_run_fn()
{
    return [](const GroupedConvProblem& problem, void* stream) -> float {
        auto& ctx  = g_conv_dispatch_buffers;
        auto param = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        ck_tile::GroupedConvBwdDataHostArgs args(
            param,
            ctx.output_ptr, // in_ptr = dX (being computed)
            ctx.weight_ptr, // wei_ptr = W
            {},
            ctx.input_ptr, // out_ptr = dY (gradient from next layer)
            1);
        ck_tile::stream_config sc;
        sc.stream_id_    = reinterpret_cast<hipStream_t>(stream);
        sc.time_kernel_  = ctx.benchmarking;
        sc.log_level_    = 0;
        sc.cold_niters_  = ctx.benchmarking ? ctx.warmup : 0;
        sc.nrepeat_      = ctx.benchmarking ? ctx.repeat : 1;
        sc.is_gpu_timer_ = ctx.benchmarking;
        return LauncherType::launch(args, sc);
    };
}

// Create a RunFn for a backward-weight convolution launcher.
// Dispatcher convention: run(X, dY, dW, problem) where dW is computed.
// BwdWeightHostArgs(param, in_ptr=X, wei_ptr=dW, {}, out_ptr=dY, k_batch)
template <typename LauncherType, int NDim>
inline GroupedConvKernelInstance::RunFn make_conv_bwd_weight_run_fn()
{
    return [](const GroupedConvProblem& problem, void* stream) -> float {
        auto& ctx         = g_conv_dispatch_buffers;
        auto param        = (NDim == 2) ? make_conv_param_2d(problem) : make_conv_param_3d(problem);
        const int k_batch = (ctx.split_k > 1) ? ctx.split_k : 1;
        ck_tile::GroupedConvBwdWeightHostArgs args(param,
                                                   ctx.input_ptr,  // in_ptr = X
                                                   ctx.output_ptr, // wei_ptr = dW (being computed)
                                                   {},
                                                   ctx.weight_ptr, // out_ptr = dY
                                                   k_batch);
        ck_tile::stream_config sc;
        sc.stream_id_    = reinterpret_cast<hipStream_t>(stream);
        sc.time_kernel_  = ctx.benchmarking;
        sc.log_level_    = 0;
        sc.cold_niters_  = ctx.benchmarking ? ctx.warmup : 0;
        sc.nrepeat_      = ctx.benchmarking ? ctx.repeat : 1;
        sc.is_gpu_timer_ = ctx.benchmarking;
        return LauncherType::launch(args, sc);
    };
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
