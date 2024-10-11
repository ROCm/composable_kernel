// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

namespace ck {
namespace impl {

template <typename Activation>
struct AddActivation
{
    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        Activation{}.template operator()<float>(y, x0 + x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const ck::half_t& x1) const
    {
        float x = x0 + ck::type_convert<float>(x1);
        Activation{}.template operator()<float>(y, x);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<ck::half_t>(ck::half_t& y, const float& x0, const float& x1) const
    {
        float result = 0;
        Activation{}.template operator()<float>(result, x0 + x1);
        y = ck::type_convert<half_t>(result);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<ck::half_t>(ck::half_t& y, const float& x0, const ck::half_t& x1) const
    {
        float result = 0;
        Activation{}.template operator()<float>(result, x0 + x1);
        y = ck::type_convert<half_t>(result);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        float x      = type_convert<float>(x0) + type_convert<float>(x1);
        float result = 0;
        Activation{}.template operator()<float>(result, x);
        y = ck::type_convert<half_t>(result);
    };
};
} // namespace impl
} // namespace ck

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Gelu        = ck::tensor_operation::element_wise::Gelu;
using Relu        = ck::tensor_operation::element_wise::Relu;
using Silu        = ck::tensor_operation::element_wise::Silu;
using Sigmoid     = ck::tensor_operation::element_wise::Sigmoid;

enum class ActivationType
{
    Gelu = 0,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    Sigmoid,
    Identity,
    GeluNoneApproximate,
    GeGluNoneApproximate,
    InvalidType
};
struct GemmBiasAddArgs
{
    const void* mat_a;
    const void* mat_b;
    const void* mat_bias;
    void* mat_c;
    ck::index_t M;
    ck::index_t N;
    ck::index_t K;
};

float gemm_bias_add_fp16(const GemmBiasAddArgs& args,
                         const StreamConfig& config,
                         ActivationType op_type);
