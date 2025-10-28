// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile::builder {

enum class DataType
{
    FP32,
    FP16,
    BF16,
    FP8,
    I8,
    U8
};

// Memory layouts for 1D convolution tensors.
// G: Group, N: Batch, K: Output Channel, C: Input Channel, W: Width
// Enum defines Input, Weight, and Output tensor layouts respectively.
enum class GroupConvLayout1D
{
    GNWC_GKXC_GNWK,
    NWGC_GKXC_NWGK,
    NGCW_GKXC_NGKW,
    NGCW_GKCX_NGKW
};

// Memory layouts for 2D convolution tensors.
// G: Group, N: Batch, K: Output Channel, C: Input Channel, Y: Height, X: Width, H: Height
// Enum defines Input, Weight, and Output tensor layouts respectively.
enum class GroupConvLayout2D
{
    GNHWC_GKYXC_GNHWK,
    NHWGC_GKYXC_NHWGK,
    NGCHW_GKYXC_NGKHW,
    NGCHW_GKCYX_NGKHW
};

// Memory layouts for 3D convolution tensors.
// G: Group, N: Batch, K: Output Channel, C: Input Channel, Z: Depth, Y: Height, X: Width, D: Depth,
// H: Height Enum defines Input, Weight, and Output tensor layouts respectively.
enum class GroupConvLayout3D
{
    GNDHWC_GKZYXC_GNDHWK,
    NDHWGC_GKZYXC_NDHWGK,
    NGCDHW_GKZYXC_NGKDHW,
    NGCDHW_GKCZYX_NGKDHW,
};

// Direction of the convolution operation.
enum class ConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHT
};

// Fused element-wise operations.
enum class ElementwiseOperation
{
    BIAS,
    BIAS_CLAMP,
    BIAS_BNORM_CLAMP,
    BILINEAR,
    CLAMP,
    SCALE,
    PASS_THROUGH
};

// Enums for the current block GEMM pipeline versions.
enum class BlockGemmPipelineVersion
{
    V1,
    V2,
    V3,
    V4,
    V5
};

// Enums for the forward convolution specialization.
enum class ConvFwdSpecialization
{
    DEFAULT,
    FILTER_1X1_PAD0,
    FILTER_1X1_STRIDE1_PAD0,
    FILTER_3x3
};

} // namespace ck_tile::builder
