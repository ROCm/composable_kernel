#pragma once

#include <type_traits>

namespace ck_tile::builder {

enum class GemmImplementationType
{
    XDL,
    WMMA,
    DL
};

enum class ConvolutionDirection
{
    Forward,
    BackwardData,
    BackwardWeight
};

enum class UniversalGemmSupport
{
    Supported,
    NotSupported
};

enum class SplitKSupport
{
    Supported,
    SupportedTwoStage,
    NotSupported
};

enum class DepthwiseOptimization
{
    X16,
    X8,
    X4,
    X2,
    NotSupported
};

enum class LargeTensorSupport
{
    Supported,
    SplitBatch,
    NotSupported
};

enum class ImplementationType
{
    ExplicitDefault,
    ExplicitMPadding,
    ExplicitNPadding,
    ExplicitKPadding,
    ExplicitMNPadding,
    ExplicitMKPadding,
    ExplicitNKPadding,
    ExplicitMNKPadding,
    Implicit
};

enum class GemmPipelineVersion
{
    Naive,
    ComputeFriendly,
    MemFriendly,
    ComputeFriendlyDoubleLDS,
    ComputeFriendlyDoubleGlobalPrefetch
};

enum class GemmPipelineScheduler
{
    Intrawave,
    Interwave
};

enum class ConvolutionSpecialization
{
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    Filter3x3
};

enum class MFMAInstructionSize
{
    M16N16,
    M32N32
};

template <typename T>
concept ConvAlgorithm = std::is_class_v<T>;

} // namespace ck_tile::builder
