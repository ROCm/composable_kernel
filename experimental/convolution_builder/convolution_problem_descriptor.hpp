#pragma once
#include <concepts>

#include "ck/utility/data_type.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/ck.hpp"

enum class ProblemDescriptorVersion
{
    V1
};

enum class ConvolutionLayout {
    NHWGC_GKYXC_NHWGK,
    NGCHW_GKCYX_NGKHW
};

template <typename T>
concept ProblemDescriptorV1 = requires {
    {T::ProblemDescriptorVersion_} -> std::convertible_to<ProblemDescriptorVersion>;
    {T::NDimSpatial_} -> std::convertible_to<int>;
    typename T::DataType;
    typename T::ElementwiseOpDataTypes;
    {T::ConvolutionLayout_} -> std::convertible_to<ConvolutionLayout>;
} && (T::ProblemDescriptorVersion_ == ProblemDescriptorVersion::V1);

struct ProblemBaseV1 {
    static constexpr ProblemDescriptorVersion ProblemDescriptorVersion_ = ProblemDescriptorVersion::V1;
    using ElementwiseOpDataTypes = ck::Tuple<>;
};

struct BF16ProblemBaseV1 : public ProblemBaseV1 {
    using DataType = ck::bhalf_t;
};

struct F32ProblemBaseV1 : public ProblemBaseV1  {
    using DataType = float;
};

struct F16ProblemBaseV1 : public ProblemBaseV1  {
    using DataType = ck::half_t;
};

struct NWGCProblemBaseV1 : public ProblemBaseV1  {
    static constexpr int NDimSpatial_ = 1;
    static constexpr ConvolutionLayout ConvolutionLayout_ = ConvolutionLayout::NHWGC_GKYXC_NHWGK;
};

struct NHWGCProblemBaseV1 : public ProblemBaseV1  {
    static constexpr int NDimSpatial_ = 2;
    static constexpr ConvolutionLayout ConvolutionLayout_ = ConvolutionLayout::NHWGC_GKYXC_NHWGK;
};

struct NDHWGCProblemBaseV1 : public ProblemBaseV1  {
    static constexpr int NDimSpatial_ = 3;
    static constexpr ConvolutionLayout ConvolutionLayout_ = ConvolutionLayout::NHWGC_GKYXC_NHWGK;
};
