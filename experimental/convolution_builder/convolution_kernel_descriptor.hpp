#pragma once
#include <concepts>

enum class KernelDescriptorVersion
{
    V1
};

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

enum class UniversalGemmSupport {
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

enum class ElementwiseOperation {
    Bias,
    BiasClamp,
    Bilinear,
    Clamp,
    Scale,
    PassThrough
};


template <typename T>
concept KernelDescriptorV1 = requires {
    {T::KernelDescriptorVersion_} -> std::convertible_to<KernelDescriptorVersion>;
    {T::GemmImplementationType_} -> std::convertible_to<GemmImplementationType>;
    {T::ConvolutionDirection_} -> std::convertible_to<ConvolutionDirection>;
    {T::UniversalGemmSupport_} -> std::convertible_to<const UniversalGemmSupport>;
    {T::SplitKSupport_} -> std::convertible_to<const SplitKSupport>;
    {T::DepthwiseOptimization_} -> std::convertible_to<const DepthwiseOptimization>;
    {T::LargeTensorSupport_} -> std::convertible_to<const LargeTensorSupport>;
    {T::ImplementationType_} -> std::convertible_to<const ImplementationType>;
    {T::ElementwiseOperation_} -> std::convertible_to<const ElementwiseOperation>;
} && (T::KernelDescriptorVersion_ == KernelDescriptorVersion::V1);

struct GroupedConvBase {
    static constexpr UniversalGemmSupport UniversalGemmSupport_ = UniversalGemmSupport::NotSupported;
    static constexpr SplitKSupport SplitKSupport_ = SplitKSupport::NotSupported;
    static constexpr DepthwiseOptimization DepthwiseOptimization_ = DepthwiseOptimization::NotSupported;
    static constexpr LargeTensorSupport LargeTensorSupport_ = LargeTensorSupport::NotSupported;
    static constexpr ImplementationType ImplementationType_ = ImplementationType::Implicit;
    static constexpr ElementwiseOperation ElementwiseOperation_ = ElementwiseOperation::PassThrough;
};

struct GroupedConvBaseXdl : public GroupedConvBase {
    static constexpr GemmImplementationType GemmImplementationType_ = GemmImplementationType::XDL;
};

struct GroupedConvBaseXdlV1 : public GroupedConvBaseXdl {
    static constexpr KernelDescriptorVersion KernelDescriptorVersion_ = KernelDescriptorVersion::V1;
};

