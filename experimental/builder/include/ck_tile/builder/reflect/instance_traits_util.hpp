// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// Utility functions and helpers for instance_traits.hpp
// Contains helper functions to convert types, enums, and sequences to string representations

#pragma once

#include <array>
#include <string>
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
inline std::string type_name()
{
    if constexpr(std::is_same_v<T, ck::half_t>)
        return "half";
    else if constexpr(std::is_same_v<T, float>)
        return "float";
    else if constexpr(std::is_same_v<T, double>)
        return "double";
    else if constexpr(std::is_same_v<T, int8_t>)
        return "int8";
    else if constexpr(std::is_same_v<T, int32_t>)
        return "int32";
    else if constexpr(std::is_same_v<T, ck::bhalf_t>)
        return "bfloat16";
    else if constexpr(std::is_same_v<T, ck::f8_t>)
        return "fp8";
    else if constexpr(std::is_same_v<T, ck::bf8_t>)
        return "bf8";
    else
        return "unknown_type";
}

// Convert layout types to string names
template <typename T>
inline std::string layout_name()
{
    // Convolution layouts
    if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::GNHWC>)
        return "GNHWC";
    else if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::GKYXC>)
        return "GKYXC";
    else if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::GNHWK>)
        return "GNHWK";
    else if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::GKZYXC>)
        return "GKZYXC";
    else if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::GNDHWC>)
        return "GNDHWC";
    else if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::GNDHWK>)
        return "GNDHWK";
    else if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::NHWGC>)
        return "NHWGC";
    else if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::KYXGC>)
        return "KYXGC";
    else if constexpr(std::is_same_v<T, ck::tensor_layout::convolution::NHWGK>)
        return "NHWGK";
    else
        return "unknown_layout";
}

// Convert element-wise operation types to string names
template <typename T>
inline std::string elementwise_op_name()
{
    if constexpr(std::is_same_v<T, ck::tensor_operation::element_wise::PassThrough>)
        return "PassThrough";
    else if constexpr(std::is_same_v<T, ck::tensor_operation::element_wise::Scale>)
        return "Scale";
    else if constexpr(std::is_same_v<T, ck::tensor_operation::element_wise::Bilinear>)
        return "Bilinear";
    else if constexpr(std::is_same_v<T, ck::tensor_operation::element_wise::Add>)
        return "Add";
    else if constexpr(std::is_same_v<T, ck::tensor_operation::element_wise::AddRelu>)
        return "AddRelu";
    else if constexpr(std::is_same_v<T, ck::tensor_operation::element_wise::Relu>)
        return "Relu";
    else
        return "unknown_op";
}

// Convert ConvolutionForwardSpecialization enum to string
inline std::string
conv_fwd_spec_name(ck::tensor_operation::device::ConvolutionForwardSpecialization spec)
{
    switch(spec)
    {
    case ck::tensor_operation::device::ConvolutionForwardSpecialization::Default: return "Default";
    case ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0:
        return "Filter1x1Stride1Pad0";
    case ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0:
        return "Filter1x1Pad0";
    case ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3:
        return "Filter3x3";
    case ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC: return "OddC";
    }
    return "unknown_conv_spec";
}

// Convert GemmSpecialization enum to string
inline std::string gemm_spec_name(ck::tensor_operation::device::GemmSpecialization spec)
{
    switch(spec)
    {
    case ck::tensor_operation::device::GemmSpecialization::Default: return "Default";
    case ck::tensor_operation::device::GemmSpecialization::MPadding: return "MPadding";
    case ck::tensor_operation::device::GemmSpecialization::NPadding: return "NPadding";
    case ck::tensor_operation::device::GemmSpecialization::KPadding: return "KPadding";
    case ck::tensor_operation::device::GemmSpecialization::MNPadding: return "MNPadding";
    case ck::tensor_operation::device::GemmSpecialization::MKPadding: return "MKPadding";
    case ck::tensor_operation::device::GemmSpecialization::NKPadding: return "NKPadding";
    case ck::tensor_operation::device::GemmSpecialization::MNKPadding: return "MNKPadding";
    case ck::tensor_operation::device::GemmSpecialization::OPadding: return "OPadding";
    case ck::tensor_operation::device::GemmSpecialization::MOPadding: return "MOPadding";
    case ck::tensor_operation::device::GemmSpecialization::NOPadding: return "NOPadding";
    case ck::tensor_operation::device::GemmSpecialization::KOPadding: return "KOPadding";
    case ck::tensor_operation::device::GemmSpecialization::MNOPadding: return "MNOPadding";
    case ck::tensor_operation::device::GemmSpecialization::MKOPadding: return "MKOPadding";
    case ck::tensor_operation::device::GemmSpecialization::NKOPadding: return "NKOPadding";
    case ck::tensor_operation::device::GemmSpecialization::MNKOPadding: return "MNKOPadding";
    }
    return "unknown_gemm_spec";
}

// Convert BlockGemmPipelineScheduler enum to string
inline std::string pipeline_scheduler_name(ck::BlockGemmPipelineScheduler sched)
{
    switch(sched)
    {
    case ck::BlockGemmPipelineScheduler::Intrawave: return "Intrawave";
    case ck::BlockGemmPipelineScheduler::Interwave: return "Interwave";
    default: return "unknown_scheduler";
    }
}

// Convert BlockGemmPipelineVersion enum to string
inline std::string pipeline_version_name(ck::BlockGemmPipelineVersion ver)
{
    switch(ver)
    {
    case ck::BlockGemmPipelineVersion::v1: return "v1";
    case ck::BlockGemmPipelineVersion::v2: return "v2";
    case ck::BlockGemmPipelineVersion::v3: return "v3";
    case ck::BlockGemmPipelineVersion::v4: return "v4";
    case ck::BlockGemmPipelineVersion::v5: return "v5";
    default: return "unknown_version";
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
inline std::string tuple_name()
{
    // For now, just check if it's an empty tuple
    return "EmptyTuple";
}

} // namespace ck_tile::reflect::detail
