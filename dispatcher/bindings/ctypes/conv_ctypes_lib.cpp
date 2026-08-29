// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Multi-kernel grouped convolution dispatcher for Python ctypes.
//
// Supports: forward / backward-data / backward-weight x 2D / 3D
//
// The dispatch header (conv_python_dispatch.hpp) is force-included via
// -include and brings in ALL compiled kernels with these aliases:
//
//   2D launchers (from include_all headers):
//     SelectedConvKernelLauncher      (forward 2D)
//     SelectedConvBwdDataLauncher     (backward-data 2D)
//     SelectedConvBwdWeightLauncher   (backward-weight 2D)
//
//   3D launchers (from dispatch header):
//     ConvFwd3dLauncher              (forward 3D)
//     ConvBwdData3dLauncher          (backward-data 3D)
//     ConvBwdWeight3dLauncher        (backward-weight 3D)
//
// Usage from Python:
//   lib = ctypes.CDLL("libdispatcher_conv_lib.so")
//   lib.conv_dispatcher_init()
//   lib.conv_dispatcher_run(input, weight, output, &problem, stream)

#include <cstring>
#include <stdexcept>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

extern "C" {

// =========================================================================
// Problem definition (matches Python ctypes struct exactly)
// =========================================================================
enum ConvDirection
{
    CONV_FORWARD    = 0,
    CONV_BWD_DATA   = 1,
    CONV_BWD_WEIGHT = 2
};

struct ConvProblemC
{
    int N, G, C, K;
    int input_d, input_h, input_w;
    int filter_z, filter_y, filter_x;
    int stride_d, stride_h, stride_w;
    int pad_d, pad_h, pad_w;
    int dilation_d, dilation_h, dilation_w;
    int direction;
    int split_k;
};

// =========================================================================
// Initialization / lifecycle
// =========================================================================
int conv_dispatcher_init() { return 0; }
int conv_dispatcher_cleanup() { return 0; }

// =========================================================================
// Library info
// =========================================================================
const char* conv_dispatcher_version() { return "2.0.0"; }

int conv_dispatcher_has_kernels()
{
#if defined(CONV_FWD_2D_AVAILABLE) || defined(CONV_FWD_3D_AVAILABLE)
    return 1;
#else
    return 0;
#endif
}

int conv_dispatcher_has_bwd_data()
{
#if defined(CONV_BWD_DATA_2D_AVAILABLE) || defined(CONV_BWD_DATA_3D_AVAILABLE)
    return 1;
#else
    return 0;
#endif
}

int conv_dispatcher_has_bwd_weight()
{
#if defined(CONV_BWD_WEIGHT_2D_AVAILABLE) || defined(CONV_BWD_WEIGHT_3D_AVAILABLE)
    return 1;
#else
    return 0;
#endif
}

int conv_dispatcher_get_kernel_count()
{
    return CONV_KERNEL_COUNT; // defined in conv_python_dispatch.hpp
}

int conv_dispatcher_get_kernel_name(int index, char* buffer, int buffer_size)
{
    if(!buffer || buffer_size <= 0 || index < 0 || index >= CONV_KERNEL_COUNT)
        return -1;
    std::strncpy(buffer, CONV_KERNEL_NAMES[index], buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
    return 0;
}

// =========================================================================
// Support query
// =========================================================================
bool conv_dispatcher_is_supported(const ConvProblemC* prob)
{
    if(!prob)
        return false;
    const bool is_3d = (prob->input_d > 1 || prob->filter_z > 1);
    switch(prob->direction)
    {
    case CONV_FORWARD:
#if defined(CONV_FWD_3D_AVAILABLE)
        if(is_3d)
            return true;
#endif
#if defined(CONV_FWD_2D_AVAILABLE)
        if(!is_3d)
            return true;
#endif
        return false;
    case CONV_BWD_DATA:
#if defined(CONV_BWD_DATA_3D_AVAILABLE)
        if(is_3d)
            return true;
#endif
#if defined(CONV_BWD_DATA_2D_AVAILABLE)
        if(!is_3d)
            return true;
#endif
        return false;
    case CONV_BWD_WEIGHT:
#if defined(CONV_BWD_WEIGHT_3D_AVAILABLE)
        if(is_3d)
            return true;
#endif
#if defined(CONV_BWD_WEIGHT_2D_AVAILABLE)
        if(!is_3d)
            return true;
#endif
        return false;
    default: return false;
    }
}

// =========================================================================
// ConvParam builders
// =========================================================================
static ck_tile::conv::ConvParam make_param_2d(const ConvProblemC* p)
{
    return ck_tile::conv::ConvParam{2,
                                    p->G,
                                    p->N,
                                    p->K,
                                    p->C,
                                    {p->filter_y, p->filter_x},
                                    {p->input_h, p->input_w},
                                    {p->stride_h, p->stride_w},
                                    {p->dilation_h, p->dilation_w},
                                    {p->pad_h, p->pad_w},
                                    {p->pad_h, p->pad_w}};
}

static ck_tile::conv::ConvParam make_param_3d(const ConvProblemC* p)
{
    return ck_tile::conv::ConvParam{3,
                                    p->G,
                                    p->N,
                                    p->K,
                                    p->C,
                                    {p->filter_z, p->filter_y, p->filter_x},
                                    {p->input_d, p->input_h, p->input_w},
                                    {p->stride_d, p->stride_h, p->stride_w},
                                    {p->dilation_d, p->dilation_h, p->dilation_w},
                                    {p->pad_d, p->pad_h, p->pad_w},
                                    {p->pad_d, p->pad_h, p->pad_w}};
}

// =========================================================================
// Kernel launch helpers
// =========================================================================

#ifdef CONV_FWD_2D_AVAILABLE
static float
launch_fwd_2d(const void* in, const void* wei, void* out, const ConvProblemC* p, hipStream_t stream)
{
    auto param = make_param_2d(p);
    ck_tile::GroupedConvFwdHostArgs<> args(param, in, wei, {}, out, 1);
    ck_tile::stream_config sc{stream, true, 1, 3, 10};
    return SelectedConvKernelLauncher::launch(args, sc);
}
#endif

#ifdef CONV_FWD_3D_AVAILABLE
static float
launch_fwd_3d(const void* in, const void* wei, void* out, const ConvProblemC* p, hipStream_t stream)
{
    auto param = make_param_3d(p);
    ck_tile::GroupedConvFwdHostArgs<> args(param, in, wei, {}, out, 1);
    ck_tile::stream_config sc{stream, true, 1, 3, 10};
    return ConvFwd3dLauncher::launch(args, sc);
}
#endif

#ifdef CONV_BWD_DATA_2D_AVAILABLE
static float launch_bwd_data_2d(
    const void* dy, const void* wei, void* dx, const ConvProblemC* p, hipStream_t stream)
{
    auto param = make_param_2d(p);
    ck_tile::GroupedConvBwdDataHostArgs args(param, dx, wei, {}, dy, 1);
    ck_tile::stream_config sc{stream, true, 1, 3, 10};
    return SelectedConvBwdDataLauncher::launch(args, sc);
}
#endif

#ifdef CONV_BWD_DATA_3D_AVAILABLE
static float launch_bwd_data_3d(
    const void* dy, const void* wei, void* dx, const ConvProblemC* p, hipStream_t stream)
{
    auto param = make_param_3d(p);
    ck_tile::GroupedConvBwdDataHostArgs args(param, dx, wei, {}, dy, 1);
    ck_tile::stream_config sc{stream, true, 1, 3, 10};
    return ConvBwdData3dLauncher::launch(args, sc);
}
#endif

#ifdef CONV_BWD_WEIGHT_2D_AVAILABLE
static float launch_bwd_weight_2d(
    const void* x, const void* dy, void* dw, const ConvProblemC* p, hipStream_t stream)
{
    auto param        = make_param_2d(p);
    const int k_batch = (p->split_k > 1) ? p->split_k : 1;
    ck_tile::GroupedConvBwdWeightHostArgs args(param, x, dw, {}, dy, k_batch);
    ck_tile::stream_config sc{stream, true, 1, 3, 10};
    return SelectedConvBwdWeightLauncher::launch(args, sc);
}
#endif

#ifdef CONV_BWD_WEIGHT_3D_AVAILABLE
static float launch_bwd_weight_3d(
    const void* x, const void* dy, void* dw, const ConvProblemC* p, hipStream_t stream)
{
    auto param        = make_param_3d(p);
    const int k_batch = (p->split_k > 1) ? p->split_k : 1;
    ck_tile::GroupedConvBwdWeightHostArgs args(param, x, dw, {}, dy, k_batch);
    ck_tile::stream_config sc{stream, true, 1, 3, 10};
    return ConvBwdWeight3dLauncher::launch(args, sc);
}
#endif

// =========================================================================
// Main dispatch
//
//  direction=0 (forward):     a=X(input),      b=W(weight),      c=Y(output)
//  direction=1 (bwd_data):    a=dY(grad_out),  b=W(weight),      c=dX(grad_in)
//  direction=2 (bwd_weight):  a=X(input),      b=dY(grad_out),   c=dW(grad_wei)
// =========================================================================
float conv_dispatcher_run(
    const void* a_ptr, const void* b_ptr, void* c_ptr, const ConvProblemC* prob, void* stream)
{
    if(!prob || !a_ptr || !b_ptr || !c_ptr)
        return -1.0f;

    const bool is_3d       = (prob->input_d > 1 || prob->filter_z > 1);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    try
    {
        switch(prob->direction)
        {
        case CONV_FORWARD:
#ifdef CONV_FWD_3D_AVAILABLE
            if(is_3d)
                return launch_fwd_3d(a_ptr, b_ptr, c_ptr, prob, hip_stream);
#endif
#ifdef CONV_FWD_2D_AVAILABLE
            if(!is_3d)
                return launch_fwd_2d(a_ptr, b_ptr, c_ptr, prob, hip_stream);
#endif
            return -2.0f;

        case CONV_BWD_DATA:
#ifdef CONV_BWD_DATA_3D_AVAILABLE
            if(is_3d)
                return launch_bwd_data_3d(a_ptr, b_ptr, c_ptr, prob, hip_stream);
#endif
#ifdef CONV_BWD_DATA_2D_AVAILABLE
            if(!is_3d)
                return launch_bwd_data_2d(a_ptr, b_ptr, c_ptr, prob, hip_stream);
#endif
            return -2.0f;

        case CONV_BWD_WEIGHT:
#ifdef CONV_BWD_WEIGHT_3D_AVAILABLE
            if(is_3d)
                return launch_bwd_weight_3d(a_ptr, b_ptr, c_ptr, prob, hip_stream);
#endif
#ifdef CONV_BWD_WEIGHT_2D_AVAILABLE
            if(!is_3d)
                return launch_bwd_weight_2d(a_ptr, b_ptr, c_ptr, prob, hip_stream);
#endif
            return -2.0f;

        default: return -1.0f;
        }
    }
    catch(const std::exception&)
    {
        return -3.0f; // Kernel rejected args (e.g. unsupported tile/channel combo)
    }
    catch(...)
    {
        return -3.0f;
    }
}

} // extern "C"
