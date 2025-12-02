// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string_view>
#include <variant>
#include <bit>
#include <array>

namespace ck_tile::builder {

enum class DataType
{
    UNDEFINDED = 0,
    FP32,
    FP16,
    BF16,
    FP8,
    INT32,
    I8,
    U8
};

enum class TensorLayout
{
    UNDEFINED,

    // Bias tensors
    GC,
    G_C_strided,
    G_K_strided,

    // 1D conv input tensor
    GNCW,
    GNWC,
    NWGC,
    NGCW,
    G_NW_C_strided,

    // 2D conv input tensor
    GNCHW,
    GNHWC,
    NHWGC,
    NGCHW,
    G_NHW_C_strided,

    // 3D conv input tensor
    GNCDHW,
    GNDHWC,
    NDHWGC,
    NGCDHW,
    G_NDHW_C_strided,

    // 1D conv weight tensor
    GKXC,
    GKCX,
    KXGC,
    G_K_X_C_strided,

    // 2D conv weight tensor
    GKYXC,
    GKCYX,
    KYXGC,
    G_K_YX_C_strided,

    // 3D conv weight tensor
    GKZYXC,
    GKCZYX,
    KZYXGC,
    G_K_ZYX_C_strided,

    // 1D conv output tensor
    GNKW,
    GNWK,
    NWGK,
    NGKW,
    G_NW_K_strided,

    // 2D conv output tensor
    GNKHW,
    GNHWK,
    NHWGK,
    NGKHW,
    G_NHW_K_strided,

    // 3D conv output tensor
    GNKDHW,
    GNDHWK,
    NDHWGK,
    NGKDHW,
    G_NDHW_K_strided
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
    BIAS_BNORM_CLAMP,
    SCALE,
    CLAMP,
    PASS_THROUGH,
    SCALEADD_SCALEADD_RELU
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

enum class ConvAlgorithmSpecialization
{
    LARGE_TENSOR
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
    case INT32: return os << "INT32";
    case I8: return os << "I8";
    case U8: return os << "U8";
    case UNDEFINDED: return os << "UNDEFINDED";
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

inline std::ostream& operator<<(std::ostream& os, ElementwiseOperation op)
{
    using enum ElementwiseOperation;
    switch(op)
    {
    case CLAMP: return os << "CLAMP";
    case SCALE: return os << "SCALE";
    case PASS_THROUGH: return os << "PASS_THROUGH";
    case BIAS_BNORM_CLAMP: return os << "BIAS_BNORM_CLAMP";
    case SCALEADD_SCALEADD_RELU: return os << "SCALEADD_SCALEADD_RELU";
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

inline std::ostream& operator<<(std::ostream& os, TensorLayout layout)
{
    using enum TensorLayout;
    switch(layout)
    {
    case GNCW: return os << "GNCW";
    case GNWC: return os << "GNWC";
    case NWGC: return os << "NWGC";
    case NGCW: return os << "NGCW";
    case G_NW_C_strided: return os << "G_NW_C_strided";
    case GNCHW: return os << "GNCHW";
    case GNHWC: return os << "GNHWC";
    case NHWGC: return os << "NHWGC";
    case NGCHW: return os << "NGCHW";
    case G_NHW_C_strided: return os << "G_NHW_C_strided";
    case GNCDHW: return os << "GNCDHW";
    case GNDHWC: return os << "GNDHWC";
    case NDHWGC: return os << "NDHWGC";
    case NGCDHW: return os << "NGCDHW";
    case G_NDHW_C_strided: return os << "G_NDHW_C_strided";
    case GKXC: return os << "GKXC";
    case GKCX: return os << "GKCX";
    case KXGC: return os << "KXGC";
    case G_K_X_C_strided: return os << "G_K_X_C_strided";
    case GKYXC: return os << "GKYXC";
    case GKCYX: return os << "GKCYX";
    case KYXGC: return os << "KYXGC";
    case G_K_YX_C_strided: return os << "G_K_YX_C_strided";
    case GKZYXC: return os << "GKZYXC";
    case GKCZYX: return os << "GKCZYX";
    case KZYXGC: return os << "KZYXGC";
    case G_K_ZYX_C_strided: return os << "G_K_ZYX_C_strided";
    case GNKW: return os << "GNKW";
    case GNWK: return os << "GNWK";
    case NWGK: return os << "NWGK";
    case NGKW: return os << "NGKW";
    case G_NW_K_strided: return os << "G_NW_K_strided";
    case GNKHW: return os << "GNKHW";
    case GNHWK: return os << "GNHWK";
    case NHWGK: return os << "NHWGK";
    case NGKHW: return os << "NGKHW";
    case G_NHW_K_strided: return os << "G_NHW_K_strided";
    case GNKDHW: return os << "GNKDHW";
    case GNDHWK: return os << "GNDHWK";
    case NDHWGK: return os << "NDHWGK";
    case NGKDHW: return os << "NGKDHW";
    case G_NDHW_K_strided: return os << "G_NDHW_K_strided";
    case GC: return os << "GC";
    case G_C_strided: return os << "G_C_strided";
    case G_K_strided: return os << "G_K_strided";
    case UNDEFINED: return os << "UNDEFINED";
    default: return os << "Unknown";
    }
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
