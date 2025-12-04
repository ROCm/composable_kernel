// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Convolution Dispatcher ctypes Library
 *
 * Provides C API for Python ctypes integration.
 * Supports forward convolution. Backward operations require additional headers.
 *
 * REQUIRED: Forward kernel header must be force-included via -include flag.
 * OPTIONAL: Backward kernels can be added with CONV_BWD_DATA_AVAILABLE/CONV_BWD_WEIGHT_AVAILABLE
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_conv.so")
 *   lib.conv_dispatcher_init()
 *   lib.conv_dispatcher_run(...)
 */

#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>

#include "ck_tile/dispatcher/conv_utils.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile::dispatcher;

// Global state
static ConvRegistry* g_registry     = nullptr;
static ConvDispatcher* g_dispatcher = nullptr;
static std::vector<const ConvKernelInstance*> g_kernels;

extern "C" {

// =============================================================================
// Initialization
// =============================================================================

int conv_dispatcher_init()
{
    if(g_registry)
        return 0; // Already initialized

    g_registry   = new ConvRegistry();
    g_dispatcher = new ConvDispatcher(g_registry);

    // Register kernel configurations
    using namespace ck_tile::dispatcher::conv_decl;

    // Forward kernels (required - must be force-included)
    ConvKernelSet fwd_set;
    fwd_set.add(ConvSignature().dtype("fp16").layout("nhwgc").conv_type("forward").dims(2),
                ConvAlgorithm()
                    .tile(1, 128, 128)
                    .wave(2, 2, 1)
                    .warp(32, 32, 16)
                    .pipeline("compv3")
                    .scheduler("intrawave"),
                "gfx942");
    g_registry->register_set(fwd_set, ConvRegistry::Priority::High);

#ifdef CONV_BWD_DATA_AVAILABLE
    // Backward data kernels
    ConvKernelSet bwd_data_set;
    bwd_data_set.add(ConvSignature().dtype("fp16").layout("nhwgc").conv_type("bwd_data").dims(2),
                     ConvAlgorithm()
                         .tile(1, 128, 128)
                         .wave(2, 2, 1)
                         .warp(32, 32, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave"),
                     "gfx942");
    g_registry->register_set(bwd_data_set, ConvRegistry::Priority::High);
#endif

#ifdef CONV_BWD_WEIGHT_AVAILABLE
    // Backward weight kernels
    ConvKernelSet bwd_weight_set;
    bwd_weight_set.add(
        ConvSignature().dtype("fp16").layout("nhwgc").conv_type("bwd_weight").dims(2),
        ConvAlgorithm()
            .tile(1, 64, 64)
            .wave(2, 2, 1)
            .warp(16, 16, 32)
            .pipeline("compv3")
            .scheduler("intrawave"),
        "gfx942");
    g_registry->register_set(bwd_weight_set, ConvRegistry::Priority::High);
#endif

    return 0;
}

int conv_dispatcher_cleanup()
{
    delete g_dispatcher;
    delete g_registry;
    g_dispatcher = nullptr;
    g_registry   = nullptr;
    g_kernels.clear();
    return 0;
}

// =============================================================================
// Registry Management
// =============================================================================

int conv_dispatcher_get_kernel_count()
{
    if(!g_registry)
        return 0;
    return static_cast<int>(g_registry->size());
}

int conv_dispatcher_get_kernel_name(int index, char* buffer, int buffer_size)
{
    if(!g_registry || index < 0)
        return -1;

    const auto& kernels = g_registry->all_kernels();
    if(static_cast<size_t>(index) >= kernels.size())
        return -1;

    const auto* kernel = kernels[index];
    std::strncpy(buffer, kernel->name().c_str(), buffer_size - 1);
    buffer[buffer_size - 1] = '\0';

    return 0;
}

// =============================================================================
// Problem Definition
// =============================================================================

struct ConvProblemC
{
    int N, G, C, K;
    int input_d, input_h, input_w;
    int filter_z, filter_y, filter_x;
    int stride_d, stride_h, stride_w;
    int pad_d, pad_h, pad_w;
    int dilation_d, dilation_h, dilation_w;
    int direction; // 0=forward, 1=bwd_data, 2=bwd_weight
};

// =============================================================================
// Kernel Selection
// =============================================================================

int conv_dispatcher_is_supported(const ConvProblemC* prob)
{
    if(!g_registry || !prob)
        return 0;

    ConvProblem problem;
    problem.N              = prob->N;
    problem.G              = prob->G;
    problem.C              = prob->C;
    problem.K              = prob->K;
    problem.input_spatial  = {prob->input_d, prob->input_h, prob->input_w};
    problem.filter_spatial = {prob->filter_z, prob->filter_y, prob->filter_x};
    problem.stride         = {prob->stride_d, prob->stride_h, prob->stride_w};
    problem.padding        = {prob->pad_d, prob->pad_h, prob->pad_w};
    problem.dilation       = {prob->dilation_d, prob->dilation_h, prob->dilation_w};
    problem.op             = static_cast<ConvOp>(prob->direction);
    problem.compute_output_size();

    const auto* kernel = g_dispatcher->select(problem);
    return kernel ? 1 : 0;
}

int conv_dispatcher_select_kernel(const ConvProblemC* prob, char* kernel_name, int buffer_size)
{
    if(!g_registry || !prob)
        return -1;

    ConvProblem problem;
    problem.N              = prob->N;
    problem.G              = prob->G;
    problem.C              = prob->C;
    problem.K              = prob->K;
    problem.input_spatial  = {prob->input_d, prob->input_h, prob->input_w};
    problem.filter_spatial = {prob->filter_z, prob->filter_y, prob->filter_x};
    problem.stride         = {prob->stride_d, prob->stride_h, prob->stride_w};
    problem.padding        = {prob->pad_d, prob->pad_h, prob->pad_w};
    problem.dilation       = {prob->dilation_d, prob->dilation_h, prob->dilation_w};
    problem.op             = static_cast<ConvOp>(prob->direction);
    problem.compute_output_size();

    const auto* kernel = g_dispatcher->select(problem);
    if(!kernel)
        return -1;

    std::strncpy(kernel_name, kernel->name().c_str(), buffer_size - 1);
    kernel_name[buffer_size - 1] = '\0';

    return 0;
}

// =============================================================================
// Convolution Execution
// =============================================================================

// Helper to build ConvParam
static ck_tile::conv::ConvParam build_conv_param(const ConvProblemC* prob)
{
    // Determine if this is 2D or 3D convolution
    const bool is_3d = (prob->input_d > 1 || prob->filter_z > 1);

    if(is_3d)
    {
        // 3D convolution: use all spatial dimensions
        return ck_tile::conv::ConvParam{3,
                                        prob->G,
                                        prob->N,
                                        prob->K,
                                        prob->C,
                                        {prob->filter_z, prob->filter_y, prob->filter_x},
                                        {prob->input_d, prob->input_h, prob->input_w},
                                        {prob->stride_d, prob->stride_h, prob->stride_w},
                                        {prob->dilation_d, prob->dilation_h, prob->dilation_w},
                                        {prob->pad_d, prob->pad_h, prob->pad_w},
                                        {prob->pad_d, prob->pad_h, prob->pad_w}};
    }
    else
    {
        // 2D convolution: only use H, W dimensions
        return ck_tile::conv::ConvParam{2,
                                        prob->G,
                                        prob->N,
                                        prob->K,
                                        prob->C,
                                        {prob->filter_y, prob->filter_x},
                                        {prob->input_h, prob->input_w},
                                        {prob->stride_h, prob->stride_w},
                                        {prob->dilation_h, prob->dilation_w},
                                        {prob->pad_h, prob->pad_w},
                                        {prob->pad_h, prob->pad_w}};
    }
}

// Forward convolution (required - kernel header must be force-included)
static float run_forward(const void* input_ptr,
                         const void* weight_ptr,
                         void* output_ptr,
                         const ConvProblemC* prob,
                         void* stream)
{
    auto conv_param = build_conv_param(prob);

    ck_tile::GroupedConvFwdHostArgs<> args(conv_param, input_ptr, weight_ptr, {}, output_ptr, 1);

    ck_tile::stream_config stream_cfg{static_cast<hipStream_t>(stream), true, 1, 3, 10};

    // SelectedConvKernelLauncher is defined in the force-included forward kernel header
    return SelectedConvKernelLauncher::launch(args, stream_cfg);
}

#ifdef CONV_BWD_DATA_AVAILABLE
// Backward data convolution (optional)
static float run_bwd_data(const void* grad_output_ptr,
                          const void* weight_ptr,
                          void* grad_input_ptr,
                          const ConvProblemC* prob,
                          void* stream)
{
    auto conv_param = build_conv_param(prob);

    ck_tile::GroupedConvBwdDataHostArgs args(
        conv_param, grad_input_ptr, weight_ptr, {}, grad_output_ptr, 1);

    ck_tile::stream_config stream_cfg{static_cast<hipStream_t>(stream), true, 1, 3, 10};

    return SelectedConvBwdDataLauncher::launch(args, stream_cfg);
}
#endif

#ifdef CONV_BWD_WEIGHT_AVAILABLE
// Backward weight convolution (optional)
// Parameters:
//   input_ptr:       original forward input X (const, read-only)
//   grad_output_ptr: gradient from next layer dY (const, read-only)
//   grad_weight_ptr: gradient of weights dW (writable, OUTPUT)
static float run_bwd_weight(const void* input_ptr,
                            const void* grad_output_ptr,
                            void* grad_weight_ptr,
                            const ConvProblemC* prob,
                            void* stream)
{
    auto conv_param = build_conv_param(prob);

    // GroupedConvBwdWeightHostArgs constructor order:
    //   (param, in=X, wei=dW (output), ds, out=dY (input), k_batch)
    // Note: wei_ptr is the OUTPUT (grad_weight), out_ptr is the INPUT (grad_output)
    ck_tile::GroupedConvBwdWeightHostArgs args(
        conv_param, input_ptr, grad_weight_ptr, {}, grad_output_ptr, 1);

    ck_tile::stream_config stream_cfg{static_cast<hipStream_t>(stream), true, 1, 3, 10};

    return SelectedConvBwdWeightLauncher::launch(args, stream_cfg);
}
#endif

float conv_dispatcher_run(const void* input_ptr,
                          const void* weight_ptr,
                          void* output_ptr,
                          const ConvProblemC* prob,
                          void* stream)
{
    if(!g_dispatcher || !prob)
        return -1.0f;

    // Build problem for kernel selection
    ConvProblem problem;
    problem.N              = prob->N;
    problem.G              = prob->G;
    problem.C              = prob->C;
    problem.K              = prob->K;
    problem.input_spatial  = {prob->input_d, prob->input_h, prob->input_w};
    problem.filter_spatial = {prob->filter_z, prob->filter_y, prob->filter_x};
    problem.stride         = {prob->stride_d, prob->stride_h, prob->stride_w};
    problem.padding        = {prob->pad_d, prob->pad_h, prob->pad_w};
    problem.dilation       = {prob->dilation_d, prob->dilation_h, prob->dilation_w};
    problem.op             = static_cast<ConvOp>(prob->direction);
    problem.compute_output_size();

    // Select kernel
    const auto* kernel = g_dispatcher->select(problem);
    if(!kernel)
        return -1.0f;

    // Dispatch based on direction
    switch(prob->direction)
    {
    case 0: // Forward (always available)
        return run_forward(input_ptr, weight_ptr, output_ptr, prob, stream);

#ifdef CONV_BWD_DATA_AVAILABLE
    case 1: // Backward data
        return run_bwd_data(input_ptr, weight_ptr, output_ptr, prob, stream);
#endif

#ifdef CONV_BWD_WEIGHT_AVAILABLE
    case 2: // Backward weight
        // Convention: caller passes (grad_output, input, grad_weight_buffer)
        // in the (input_ptr, weight_ptr, output_ptr) slots respectively.
        // This is consistent with bwd_data where grad_output goes in input_ptr slot.
        // run_bwd_weight expects: (input, grad_output, grad_weight)
        return run_bwd_weight(weight_ptr, input_ptr, output_ptr, prob, stream);
#endif

    default: return -1.0f;
    }
}

// =============================================================================
// Info
// =============================================================================

const char* conv_dispatcher_version() { return "1.0.0"; }

int conv_dispatcher_has_kernels()
{
    return 1; // Forward kernel is required
}

int conv_dispatcher_has_bwd_data()
{
#ifdef CONV_BWD_DATA_AVAILABLE
    return 1;
#else
    return 0;
#endif
}

int conv_dispatcher_has_bwd_weight()
{
#ifdef CONV_BWD_WEIGHT_AVAILABLE
    return 1;
#else
    return 0;
#endif
}

} // extern "C"
