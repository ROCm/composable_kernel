// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

struct GroupConvLayout
{
    union
    {
        GroupConvLayout1D _1d;
        GroupConvLayout2D _2d;
        GroupConvLayout3D _3d;
    };

    constexpr GroupConvLayout(GroupConvLayout1D layout) : _1d(layout) {}
    constexpr GroupConvLayout(GroupConvLayout2D layout) : _2d(layout) {}
    constexpr GroupConvLayout(GroupConvLayout3D layout) : _3d(layout) {}
};

// Direction of the convolution operation.
enum class ConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHT
};

// Forward convolution device operations.
enum class FwdGroupConvDeviceOperation
{
    DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK,
    DeviceGroupedConvFwdMultipleD_Wmma_CShuffle,
    DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle,
    DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3,
    DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor
};

// Backward data convolution device operations.
enum class BwdDataGroupConvDeviceOperation
{
    DeviceGroupedConvBwdDataMultipleD,
    DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle,
    DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
};

// Backward weight convolution device operations.
enum class BwdWeightGroupConvDeviceOperation
{
    DeviceGroupedConvBwdWeight,
    DeviceGroupedConvBwdWeight_Dl,
    DeviceGroupedConvBwdWeight_Xdl_CShuffle,
    DeviceGroupedConvBwdWeight_Xdl_CShuffleV3,
    DeviceGroupedConvBwdWeight_Wmma_CShuffle,
    DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle,
    DeviceGroupedConvBwdWeightMultipleD,
    DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle,
};

// Structural type for device operation
struct GroupConvDeviceOp
{
    union
    {
        FwdGroupConvDeviceOperation _fwd;
        BwdDataGroupConvDeviceOperation _bwd_data;
        BwdWeightGroupConvDeviceOperation _bwd_weight;
    };

    constexpr GroupConvDeviceOp(FwdGroupConvDeviceOperation op) : _fwd(op) {}
    constexpr GroupConvDeviceOp(BwdDataGroupConvDeviceOperation op) : _bwd_data(op) {}
    constexpr GroupConvDeviceOp(BwdWeightGroupConvDeviceOperation op) : _bwd_weight(op) {}
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

// Enums for pipeline versions & schedulers
enum class PipelineVersion
{
    V1,
    V2,
    V3,
    V4,
    V5,
    WEIGHT_ONLY
};

// Enums for the GEMM specialization.
enum struct GemmSpecialization
{
    // Gemm
    Default,
    MPadding,
    NPadding,
    KPadding,
    MNPadding,
    MKPadding,
    NKPadding,
    MNKPadding,
    // Gemm + Gemm
    OPadding,
    MOPadding,
    NOPadding,
    KOPadding,
    MNOPadding,
    MKOPadding,
    NKOPadding,
    MNKOPadding
};

// Enums for the forward convolution specialization.
enum class ConvFwdSpecialization
{
    DEFAULT,
    FILTER_1X1_PAD0,
    FILTER_1X1_STRIDE1_PAD0,
    FILTER_3x3
};

// Enums for the backward data convolution specialization.
enum class ConvBwdDataSpecialization
{
    DEFAULT,
    FILTER_1X1_STRIDE1_PAD0,
};

// Enums for the backward weight convolution specialization.
enum class ConvBwdWeightSpecialization
{
    DEFAULT,
    FILTER_1X1_STRIDE1_PAD0,
    FILTER_1X1_PAD0,
    ODD_C,
};

// Enums for the Gemm padding.
enum class GemmPadding
{
    DEFAULT,
    M_PADDING,
    N_PADDING,
    K_PADDING,
    MN_PADDING,
    MK_PADDING,
    NK_PADDING,
    MNK_PADDING,
    O_PADDING,
    MO_PADDING,
    NO_PADDING,
    KO_PADDING,
    MNO_PADDING,
    MKO_PADDING,
    NKO_PADDING,
    MNKO_PADDING,
};

enum class PipelineScheduler
{
    DEFAULT,
    INTRAWAVE,
    INTERWAVE
};

} // namespace ck_tile::builder
