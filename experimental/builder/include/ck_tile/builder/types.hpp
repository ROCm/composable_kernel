#pragma once

namespace ck_tile::builder {

enum class DataType
{
    FP64,
    FP32,
    FP16,
    BF16,
    S16,
    S8,
    S4,
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

} // namespace ck_tile::builder
