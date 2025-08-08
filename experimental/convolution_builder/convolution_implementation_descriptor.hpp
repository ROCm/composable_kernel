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


enum class ConvolutionSpecialization {
    Default,
    Filter1x1Pad0,
    Filter1x1Stride1Pad0,
    Filter3x3
};

enum class ConvolutionLayout {
    NHWGC_GKYXC_NHWGK,
    NGCHW_GKCYX_NGKHW
};

enum class MFMAInstructionSize {
    M16N16,
    M32N32
};


template <typename T>
concept ImplementationDescriptorV1 = requires {
    {T::ImplementationDescriptorVersion_} -> std::convertible_to<ImplementationDescriptorVersion>;
    {T::NDimSpatial_} -> std::convertible_to<int>;
    typename T::DataType;
    typename T::ElementwiseOpDataTypes;
    {T::ConvolutionSpecialization_} -> std::convertible_to<ConvolutionSpecialization>;
    {T::ConvolutionLayout_} -> std::convertible_to<ConvolutionLayout>;
    {T::BlockSize_} -> std::convertible_to<int>;
    {T::TileSizes_} -> std::convertible_to<std::tuple<int, int, int>>;
    {T::K1_} -> std::convertible_to<int>;
    {T::MFMAInstructionSize_} -> std::convertible_to<MFMAInstructionSize>;
    {T::XdlPerWave_} -> std::convertible_to<std::tuple<int, int>>;
    {T::GlobalTransferVectorSize_} -> std::convertible_to<std::tuple<int, int, int>>;
    {T::LDSStoreVectorSize_} -> std::convertible_to<std::tuple<int, int>>;
} && (T::ImplementationDescriptorVersion_ == ImplementationDescriptorVersion::V1);

struct ImplementationBaseV1 {
    static constexpr ImplementationDescriptorVersion ImplementationDescriptorVersion_ = ImplementationDescriptorVersion::V1;
    using DataType = ck::bhalf_t;
    using ElementwiseOpDataTypes = ck::Tuple<>;
    static constexpr ConvolutionSpecialization ConvolutionSpecialization_ = ConvolutionSpecialization::Default;
};

struct BF16ImplementationBaseV1 : public ImplementationBaseV1 {
    using DataType = ck::bhalf_t;
};

struct F32ImplementationBaseV1 : public ImplementationBaseV1  {
    using DataType = float;
};

struct F16ImplementationBaseV1 : public ImplementationBaseV1  {
    using DataType = ck::half_t;
};

struct NWCImplementationBaseV1 : public ImplementationBaseV1  {
    static constexpr int NDimSpatial_ = 1;
    static constexpr ConvolutionLayout ConvolutionLayout_ = ConvolutionLayout::NHWGC_GKYXC_NHWGK;
};

struct NHWCImplementationBaseV1 : public ImplementationBaseV1  {
    static constexpr int NDimSpatial_ = 2;
    static constexpr ConvolutionLayout ConvolutionLayout_ = ConvolutionLayout::NHWGC_GKYXC_NHWGK;
};

struct NDHWCImplementationBaseV1 : public ImplementationBaseV1  {
    static constexpr int NDimSpatial_ = 3;
    static constexpr ConvolutionLayout ConvolutionLayout_ = ConvolutionLayout::NHWGC_GKYXC_NHWGK;
};
