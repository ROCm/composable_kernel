// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string_view>
#include <variant>

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

// ostream operator overloads for enum classes
inline std::ostream& operator<<(std::ostream& os, DataType dt)
{
    using enum DataType;
    switch(dt)
    {
    case FP16: return os << "FP16";
    case FP32: return os << "FP32";
    case BF16: return os << "BF16";
    case FP8: return os << "FP8";
    case I8: return os << "I8";
    case U8: return os << "U8";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, ConvDirection dir)
{
    using enum ConvDirection;
    switch(dir)
    {
    case FORWARD: return os << "Forward";
    case BACKWARD_DATA: return os << "Backward Data";
    case BACKWARD_WEIGHT: return os << "Backward Weight";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, GroupConvLayout1D layout)
{
    using enum GroupConvLayout1D;
    switch(layout)
    {
    case GNWC_GKXC_GNWK: return os << "GNWC_GKXC_GNWK";
    case NWGC_GKXC_NWGK: return os << "NWGC_GKXC_NWGK";
    case NGCW_GKXC_NGKW: return os << "NGCW_GKXC_NGKW";
    case NGCW_GKCX_NGKW: return os << "NGCW_GKCX_NGKW";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, GroupConvLayout2D layout)
{
    using enum GroupConvLayout2D;
    switch(layout)
    {
    case GNHWC_GKYXC_GNHWK: return os << "GNHWC_GKYXC_GNHWK";
    case NHWGC_GKYXC_NHWGK: return os << "NHWGC_GKYXC_NHWGK";
    case NGCHW_GKYXC_NGKHW: return os << "NGCHW_GKYXC_NGKHW";
    case NGCHW_GKCYX_NGKHW: return os << "NGCHW_GKCYX_NGKHW";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, GroupConvLayout3D layout)
{
    using enum GroupConvLayout3D;
    switch(layout)
    {
    case GNDHWC_GKZYXC_GNDHWK: return os << "GNDHWC_GKZYXC_GNDHWK";
    case NDHWGC_GKZYXC_NDHWGK: return os << "NDHWGC_GKZYXC_NDHWGK";
    case NGCDHW_GKZYXC_NGKDHW: return os << "NGCDHW_GKZYXC_NGKDHW";
    case NGCDHW_GKCZYX_NGKDHW: return os << "NGCDHW_GKCZYX_NGKDHW";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, FwdGroupConvDeviceOperation op)
{
    using enum FwdGroupConvDeviceOperation;
    switch(op)
    {
    case DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK:
        return os << "DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK";
    case DeviceGroupedConvFwdMultipleD_Wmma_CShuffle:
        return os << "DeviceGroupedConvFwdMultipleD_Wmma_CShuffle";
    case DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle:
        return os << "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle";
    case DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3:
        return os << "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3";
    case DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor:
        return os << "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, BwdDataGroupConvDeviceOperation op)
{
    using enum BwdDataGroupConvDeviceOperation;
    switch(op)
    {
    case DeviceGroupedConvBwdDataMultipleD: return os << "DeviceGroupedConvBwdDataMultipleD";
    case DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle:
        return os << "DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle";
    case DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1:
        return os << "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, BwdWeightGroupConvDeviceOperation op)
{
    using enum BwdWeightGroupConvDeviceOperation;
    switch(op)
    {
    case DeviceGroupedConvBwdWeight: return os << "DeviceGroupedConvBwdWeight";
    case DeviceGroupedConvBwdWeight_Dl: return os << "DeviceGroupedConvBwdWeight_Dl";
    case DeviceGroupedConvBwdWeight_Xdl_CShuffle:
        return os << "DeviceGroupedConvBwdWeight_Xdl_CShuffle";
    case DeviceGroupedConvBwdWeight_Xdl_CShuffleV3:
        return os << "DeviceGroupedConvBwdWeight_Xdl_CShuffleV3";
    case DeviceGroupedConvBwdWeight_Wmma_CShuffle:
        return os << "DeviceGroupedConvBwdWeight_Wmma_CShuffle";
    case DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle:
        return os << "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle";
    case DeviceGroupedConvBwdWeightMultipleD: return os << "DeviceGroupedConvBwdWeightMultipleD";
    case DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle:
        return os << "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, ElementwiseOperation op)
{
    using enum ElementwiseOperation;
    switch(op)
    {
    case BIAS: return os << "BIAS";
    case BIAS_CLAMP: return os << "BIAS_CLAMP";
    case BIAS_BNORM_CLAMP: return os << "BIAS_BNORM_CLAMP";
    case BILINEAR: return os << "BILINEAR";
    case CLAMP: return os << "CLAMP";
    case SCALE: return os << "SCALE";
    case PASS_THROUGH: return os << "PASS_THROUGH";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, PipelineVersion ver)
{
    using enum PipelineVersion;
    switch(ver)
    {
    case V1: return os << "V1";
    case V2: return os << "V2";
    case V3: return os << "V3";
    case V4: return os << "V4";
    case V5: return os << "V5";
    case WEIGHT_ONLY: return os << "WEIGHT_ONLY";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, GemmSpecialization spec)
{
    using enum GemmSpecialization;
    switch(spec)
    {
    case Default: return os << "Default";
    case MPadding: return os << "MPadding";
    case NPadding: return os << "NPadding";
    case KPadding: return os << "KPadding";
    case MNPadding: return os << "MNPadding";
    case MKPadding: return os << "MKPadding";
    case NKPadding: return os << "NKPadding";
    case MNKPadding: return os << "MNKPadding";
    case OPadding: return os << "OPadding";
    case MOPadding: return os << "MOPadding";
    case NOPadding: return os << "NOPadding";
    case KOPadding: return os << "KOPadding";
    case MNOPadding: return os << "MNOPadding";
    case MKOPadding: return os << "MKOPadding";
    case NKOPadding: return os << "NKOPadding";
    case MNKOPadding: return os << "MNKOPadding";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, ConvFwdSpecialization spec)
{
    using enum ConvFwdSpecialization;
    switch(spec)
    {
    case DEFAULT: return os << "DEFAULT";
    case FILTER_1X1_PAD0: return os << "FILTER_1X1_PAD0";
    case FILTER_1X1_STRIDE1_PAD0: return os << "FILTER_1X1_STRIDE1_PAD0";
    case FILTER_3x3: return os << "FILTER_3x3";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, ConvBwdDataSpecialization spec)
{
    using enum ConvBwdDataSpecialization;
    switch(spec)
    {
    case DEFAULT: return os << "DEFAULT";
    case FILTER_1X1_STRIDE1_PAD0: return os << "FILTER_1X1_STRIDE1_PAD0";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, ConvBwdWeightSpecialization spec)
{
    using enum ConvBwdWeightSpecialization;
    switch(spec)
    {
    case DEFAULT: return os << "DEFAULT";
    case FILTER_1X1_STRIDE1_PAD0: return os << "FILTER_1X1_STRIDE1_PAD0";
    case FILTER_1X1_PAD0: return os << "FILTER_1X1_PAD0";
    case ODD_C: return os << "ODD_C";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, GemmPadding padding)
{
    using enum GemmPadding;
    switch(padding)
    {
    case DEFAULT: return os << "DEFAULT";
    case M_PADDING: return os << "M_PADDING";
    case N_PADDING: return os << "N_PADDING";
    case K_PADDING: return os << "K_PADDING";
    case MN_PADDING: return os << "MN_PADDING";
    case MK_PADDING: return os << "MK_PADDING";
    case NK_PADDING: return os << "NK_PADDING";
    case MNK_PADDING: return os << "MNK_PADDING";
    case O_PADDING: return os << "O_PADDING";
    case MO_PADDING: return os << "MO_PADDING";
    case NO_PADDING: return os << "NO_PADDING";
    case KO_PADDING: return os << "KO_PADDING";
    case MNO_PADDING: return os << "MNO_PADDING";
    case MKO_PADDING: return os << "MKO_PADDING";
    case NKO_PADDING: return os << "NKO_PADDING";
    case MNKO_PADDING: return os << "MNKO_PADDING";
    default: return os << "Unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, PipelineScheduler sched)
{
    using enum PipelineScheduler;
    switch(sched)
    {
    case DEFAULT: return os << "DEFAULT";
    case INTRAWAVE: return os << "INTRAWAVE";
    case INTERWAVE: return os << "INTERWAVE";
    default: return os << "Unknown";
    }
}

// ostream operator overload for std::variant of layout types
inline std::ostream&
operator<<(std::ostream& os,
           const std::variant<GroupConvLayout1D, GroupConvLayout2D, GroupConvLayout3D>& layout)
{
    std::visit([&os](const auto& l) { os << l; }, layout);
    return os;
}

// ostream operator overload for std::variant of convolution specializations
inline std::ostream& operator<<(std::ostream& os,
                                const std::variant<ConvFwdSpecialization,
                                                   ConvBwdDataSpecialization,
                                                   ConvBwdWeightSpecialization>& spec)
{
    std::visit([&os](const auto& s) { os << s; }, spec);
    return os;
}

} // namespace ck_tile::builder
