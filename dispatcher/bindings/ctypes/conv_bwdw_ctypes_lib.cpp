// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Convolution Backward Weight Dispatcher ctypes Library
 *
 * SEPARATE library for backward weight to avoid template conflicts with
 * forward/backward_data kernels in the main conv_ctypes_lib.
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libdispatcher_conv_bwdw_lib.so")
 *   lib.conv_bwdw_init()
 *   lib.conv_bwdw_run(...)
 */

#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>

// Minimal includes - matching the C++ example
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/gemm.hpp" // Must be before grouped_convolution for TileGemmTraits
#include "ck_tile/ops/grouped_convolution.hpp"

// Global state - minimal, no registry needed for direct launch
static bool g_bwdw_initialized = false;

extern "C" {

// =============================================================================
// Initialization (minimal - just sets flag)
// =============================================================================

int conv_bwdw_init()
{
    g_bwdw_initialized = true;
    return 0; // Return 0 on success (consistent with other init functions)
}

void conv_bwdw_cleanup() { g_bwdw_initialized = false; }

// =============================================================================
// Problem Structure (same as main library)
// =============================================================================

struct ConvBwdwProblemC
{
    int N, G, C, K;
    int input_d, input_h, input_w;
    int filter_z, filter_y, filter_x;
    int stride_d, stride_h, stride_w;
    int pad_d, pad_h, pad_w;
    int dilation_d, dilation_h, dilation_w;
    int split_k;
};

// =============================================================================
// Backward Weight Execution
// =============================================================================

#ifdef CONV_BWD_WEIGHT_AVAILABLE
static ck_tile::conv::ConvParam build_conv_param(const ConvBwdwProblemC* prob)
{
    const bool is_3d = (prob->input_d > 1 || prob->filter_z > 1);

    if(is_3d)
    {
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

static float run_bwd_weight_impl(const void* input_ptr,
                                 const void* grad_output_ptr,
                                 void* grad_weight_ptr,
                                 const ConvBwdwProblemC* prob,
                                 void* stream)
{
    auto conv_param = build_conv_param(prob);

    // Backward weight: A=input, B=grad_output, C=grad_weight
    ck_tile::GroupedConvBwdWeightHostArgs args(conv_param,
                                               input_ptr,       // in_ptr = input
                                               grad_weight_ptr, // wei_ptr = grad_weight (output)
                                               {},              // ds_ptr
                                               grad_output_ptr, // out_ptr = grad_output
                                               (prob->split_k > 1) ? prob->split_k : 1);

    ck_tile::stream_config stream_cfg{static_cast<hipStream_t>(stream), true, 1, 3, 10};

    return SelectedConvBwdWeightLauncher::launch(args, stream_cfg);
}
#endif

float conv_bwdw_run(const void* input_ptr,
                    const void* grad_output_ptr,
                    void* grad_weight_ptr,
                    const ConvBwdwProblemC* prob,
                    void* stream)
{
#ifdef CONV_BWD_WEIGHT_AVAILABLE
    // Validate all required pointers before kernel launch
    if(!g_bwdw_initialized || !prob)
        return -1.0f;
    if(!input_ptr || !grad_output_ptr || !grad_weight_ptr)
        return -1.0f; // Null data pointer would cause kernel crash
    return run_bwd_weight_impl(input_ptr, grad_output_ptr, grad_weight_ptr, prob, stream);
#else
    return -1.0f;
#endif
}

// =============================================================================
// Info
// =============================================================================

const char* conv_bwdw_version() { return "1.0.0"; }

int conv_bwdw_has_kernels()
{
#ifdef CONV_BWD_WEIGHT_AVAILABLE
    return 1;
#else
    return 0;
#endif
}

int conv_bwdw_get_kernel_count()
{
#ifdef CONV_BWD_WEIGHT_AVAILABLE
    return 1;
#else
    return 0;
#endif
}

int conv_bwdw_get_kernel_name(int index, char* buffer, int buffer_size)
{
#ifdef CONV_BWD_WEIGHT_AVAILABLE
    if(index != 0 || !buffer || buffer_size <= 0)
        return -1;
    std::strncpy(buffer, CONV_BWD_WEIGHT_KERNEL_NAME, buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
#else
    return -1;
#endif
}

} // extern "C"
