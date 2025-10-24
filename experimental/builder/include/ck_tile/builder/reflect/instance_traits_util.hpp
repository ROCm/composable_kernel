// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// Utility functions and helpers for instance_traits.hpp
// Contains helper functions to convert types, enums, and sequences to string representations.
// The helper function are consteval so that unknown cases cause compile-time errors.

#pragma once

#include <array>
#include <string>
#include <string_view>
#include <sstream>
#include <type_traits>
#include <ck/utility/data_type.hpp>
#include <ck/utility/sequence.hpp>
#include <ck/utility/blkgemmpipe_scheduler.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>

namespace ck_tile::reflect::detail {

// Metaprogramming helper to convert ck::Sequence to constexpr std::array
template <typename Seq>
struct SequenceToArray;

template <ck::index_t... Is>
struct SequenceToArray<ck::Sequence<Is...>>
{
    static constexpr std::array<int, sizeof...(Is)> value = {static_cast<int>(Is)...};
};

// Convert data types to string names
template <typename T>
consteval std::string_view type_name()
{
    if constexpr(std::is_same_v<T, ck::half_t>)
        return "fp16";
    else if constexpr(std::is_same_v<T, float>)
        return "fp32";
    else if constexpr(std::is_same_v<T, double>)
        return "fp64";
    else if constexpr(std::is_same_v<T, int8_t>)
        return "s8";
    else if constexpr(std::is_same_v<T, int32_t>)
        return "s32";
    else if constexpr(std::is_same_v<T, ck::bhalf_t>)
        return "bf16";
    else if constexpr(std::is_same_v<T, ck::f8_t>)
        return "fp8";
    else if constexpr(std::is_same_v<T, ck::bf8_t>)
        return "bf8";
    else
        static_assert(false, "unknown_type");
}

// Convert layout types to string names
template <typename T>
constexpr std::string_view layout_name()
{
    if constexpr(requires {
                     { T::name } -> std::convertible_to<std::string_view>;
                 })
        return T::name;
    else
        static_assert(false, "layout type is missing name attribute");
}

// Convert element-wise operation types to string names
template <typename T>
constexpr std::string_view elementwise_op_name()
{
    namespace element_wise = ck::tensor_operation::element_wise;

    if constexpr(std::is_same_v<T, element_wise::PassThrough>)
        return "PassThrough";
    else if constexpr(std::is_same_v<T, element_wise::Scale>)
        return "Scale";
    else if constexpr(std::is_same_v<T, element_wise::Bilinear>)
        return "Bilinear";
    else if constexpr(std::is_same_v<T, element_wise::Add>)
        return "Add";
    else if constexpr(std::is_same_v<T, element_wise::AddRelu>)
        return "AddRelu";
    else if constexpr(std::is_same_v<T, element_wise::Relu>)
        return "Relu";
    else if constexpr(std::is_same_v<T, element_wise::BiasNormalizeInInferClamp>)
        return "BiasNormalizeInInferClamp";
    else if constexpr(std::is_same_v<T, element_wise::Clamp>)
        return "Clamp";
    else if constexpr(std::is_same_v<T, element_wise::AddClamp>)
        return "AddClamp";
    else
        static_assert(false, "unknown_op");
}

// Convert ConvolutionForwardSpecialization enum to string
constexpr std::string_view
conv_fwd_spec_name(ck::tensor_operation::device::ConvolutionForwardSpecialization spec)
{
    using ck::tensor_operation::device::ConvolutionForwardSpecialization;
    switch(spec)
    {
    case ConvolutionForwardSpecialization::Default: return "Default";
    case ConvolutionForwardSpecialization::Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
    case ConvolutionForwardSpecialization::Filter1x1Pad0: return "Filter1x1Pad0";
    case ConvolutionForwardSpecialization::Filter3x3: return "Filter3x3";
    case ConvolutionForwardSpecialization::OddC: return "OddC";
    }
}

// Convert GemmSpecialization enum to string
constexpr std::string_view gemm_spec_name(ck::tensor_operation::device::GemmSpecialization spec)
{
    using ck::tensor_operation::device::GemmSpecialization;
    switch(spec)
    {
    case GemmSpecialization::Default: return "Default";
    case GemmSpecialization::MPadding: return "MPadding";
    case GemmSpecialization::NPadding: return "NPadding";
    case GemmSpecialization::KPadding: return "KPadding";
    case GemmSpecialization::MNPadding: return "MNPadding";
    case GemmSpecialization::MKPadding: return "MKPadding";
    case GemmSpecialization::NKPadding: return "NKPadding";
    case GemmSpecialization::MNKPadding: return "MNKPadding";
    case GemmSpecialization::OPadding: return "OPadding";
    case GemmSpecialization::MOPadding: return "MOPadding";
    case GemmSpecialization::NOPadding: return "NOPadding";
    case GemmSpecialization::KOPadding: return "KOPadding";
    case GemmSpecialization::MNOPadding: return "MNOPadding";
    case GemmSpecialization::MKOPadding: return "MKOPadding";
    case GemmSpecialization::NKOPadding: return "NKOPadding";
    case GemmSpecialization::MNKOPadding: return "MNKOPadding";
    }
}

// Convert BlockGemmPipelineScheduler enum to string
constexpr std::string_view pipeline_scheduler_name(ck::BlockGemmPipelineScheduler sched)
{
    using ck::BlockGemmPipelineScheduler;
    switch(sched)
    {
    case BlockGemmPipelineScheduler::Intrawave: return "Intrawave";
    case BlockGemmPipelineScheduler::Interwave: return "Interwave";
    }
}

// Convert BlockGemmPipelineVersion enum to string
constexpr std::string_view pipeline_version_name(ck::BlockGemmPipelineVersion ver)
{
    using ck::BlockGemmPipelineVersion;
    switch(ver)
    {
    case BlockGemmPipelineVersion::v1: return "v1";
    case BlockGemmPipelineVersion::v2: return "v2";
    case BlockGemmPipelineVersion::v3: return "v3";
    case BlockGemmPipelineVersion::v4: return "v4";
    case BlockGemmPipelineVersion::v5: return "v5";
    }
}

// Convert std::array to string
template <typename T, std::size_t N>
inline std::string array_to_string(const std::array<T, N>& arr)
{
    std::ostringstream oss;
    oss << "Seq(";
    for(std::size_t i = 0; i < arr.size(); ++i)
    {
        if(i > 0)
            oss << ",";
        oss << arr[i];
    }
    oss << ")";
    return oss.str();
}

// Handle ck::Tuple (empty tuple for DsLayout/DsDataType)
template <typename T>
constexpr std::string_view tuple_name()
{
    // For now, just check if it's an empty tuple
    return "EmptyTuple";
}

} // namespace ck_tile::reflect::detail
