#pragma once
#include <concepts>

#include "ck/utility/data_type.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/ck.hpp"

enum class ImplementationDescriptorVersion
{
    V1
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

enum class ConvolutionSpecialization {
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    Filter3x3
};

enum class MFMAInstructionSize {
    M16N16,
    M32N32
};


template <typename T>
concept ImplementationDescriptorV1 = requires {
    {T::ImplementationDescriptorVersion_} -> std::convertible_to<ImplementationDescriptorVersion>;
    {T::ConvolutionSpecialization_} -> std::convertible_to<ConvolutionSpecialization>;
    {T::GemmPipelineVersion_} -> std::convertible_to<GemmPipelineVersion>;
    {T::GemmPipelineScheduler_} -> std::convertible_to<GemmPipelineScheduler>;
    {T::BlockSize_} -> std::convertible_to<int>;
    {T::TileSizes_} -> std::convertible_to<std::tuple<int, int, int>>;
    {T::K1_} -> std::convertible_to<int>;
    {T::MFMAInstructionSize_} -> std::convertible_to<MFMAInstructionSize>;
    {T::XdlPerWave_} -> std::convertible_to<std::tuple<int, int>>;
    {T::GlobalTransferVectorSize_} -> std::convertible_to<std::tuple<int, int, int>>;
    {T::LDSStoreVectorSize_} -> std::convertible_to<std::tuple<int, int>>;
} && (T::ImplementationDescriptorVersion_ == ImplementationDescriptorVersion::V1);

struct ImplementationDefaultV1 {
    static constexpr ImplementationDescriptorVersion ImplementationDescriptorVersion_ = ImplementationDescriptorVersion::V1;
    static constexpr ConvolutionSpecialization ConvolutionSpecialization_ = ConvolutionSpecialization::Default;
    static constexpr GemmPipelineVersion GemmPipelineVersion_ = GemmPipelineVersion::Naive;
    static constexpr GemmPipelineScheduler GemmPipelineScheduler_ = GemmPipelineScheduler::Intrawave;
};
