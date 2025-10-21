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
    I8
};

// Memory layouts for convolution tensors, following PyTorch conventions.
enum class GroupConvLayout
{
    CHANNELS_LAST, // e.g., NHWGC
    CHANNELS_FIRST // e.g., NGCHW
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
    BILINEAR,
    CLAMP,
    SCALE,
    PASS_THROUGH
};

// Enums for the current block GEMM pipeline versions.
enum class BlockGemmPipelineVersion
{
    V1,
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
    ODD_C,
    FILTER_3x3
};

} // namespace ck_tile::builder
