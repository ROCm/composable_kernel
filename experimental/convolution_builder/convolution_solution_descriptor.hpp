#pragma once
#include <iostream>
#include <concepts>

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


enum class GemmPipelineVersion
{
    V1,
    V2,
    V3,
    V4,
    V5
};

enum class GemmPipelineScheduler
{
    Intrawave,
    Interwave
};

enum class SplitKSupport
{
    Supported,
    SupportedTwoStage,
    NotSupported
};

enum class MergedGroups
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

enum class ElementwiseOperation {
    Bias,
    BiasClamp,
    Bilinear,
    Clamp,
    Scale,
    PassThrough
};


template <typename T>
concept SolutionDescriptorV1 = requires {
    {T::GemmImplementationType_} -> std::convertible_to<GemmImplementationType>;
    {T::ConvolutionDirection_} -> std::convertible_to<ConvolutionDirection>;
    {T::GemmPipelineVersion_} -> std::convertible_to<const GemmPipelineVersion>;
    {T::GemmPipelineScheduler_} -> std::convertible_to<const GemmPipelineScheduler>;
    {T::SplitKSupport_} -> std::convertible_to<const SplitKSupport>;
    {T::MergedGroups_} -> std::convertible_to<const MergedGroups>;
    {T::LargeTensorSupport_} -> std::convertible_to<const LargeTensorSupport>;
    {T::ImplementationType_} -> std::convertible_to<const ImplementationType>;
    {T::ElementwiseOperation_} -> std::convertible_to<const ElementwiseOperation>;
};

struct GroupedConvBase {
    static constexpr GemmPipelineVersion GemmPipelineVersion_ = GemmPipelineVersion::V1;
    static constexpr GemmPipelineScheduler GemmPipelineScheduler_ = GemmPipelineScheduler::Intrawave;
    static constexpr SplitKSupport SplitKSupport_ = SplitKSupport::NotSupported;
    static constexpr MergedGroups MergedGroups_ = MergedGroups::NotSupported;
    static constexpr LargeTensorSupport LargeTensorSupport_ = LargeTensorSupport::NotSupported;
    static constexpr ImplementationType ImplementationType_ = ImplementationType::Implicit;
    static constexpr ElementwiseOperation ElementwiseOperation_ = ElementwiseOperation::PassThrough;
};

struct GroupedConvBaseXdl : public GroupedConvBase {
    static constexpr GemmImplementationType GemmImplementationType_ = GemmImplementationType::XDL;
};
