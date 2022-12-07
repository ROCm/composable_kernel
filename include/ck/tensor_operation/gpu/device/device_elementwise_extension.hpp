// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>
#include <array>

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/math_v2.hpp"

#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct NormalizeInInfer
{
    NormalizeInInfer(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T1, typename T2, typename T3, typename T4>
    __host__ __device__ constexpr void operator()(T1& y,
                                                  const T1& x,
                                                  const T2& mean,
                                                  const T2& variance,
                                                  const T3& gamma,
                                                  const T4& beta) const
    {
        static_assert(std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                      "Data type is not supported by this operation!");

        using ck::type_convert;
        using ck::math::sqrt;

        T2 tmp_x, tmp_y;

        tmp_x = type_convert<T2>(x);

        tmp_y = ((tmp_x - mean) / sqrt(variance + type_convert<T2>(epsilon_))) *
                    type_convert<T2>(gamma) +
                type_convert<T2>(beta);
        y = type_convert<T1>(tmp_y);
    };

    double epsilon_;
};

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          index_t Rank>
using DeviceElementwiseForBatchNormInfer = ck::tensor_operation::device::DeviceElementwise<
    ck::Tuple<XDataType, MeanVarDataType, MeanVarDataType, ScaleDataType, BiasDataType>,
    ck::Tuple<YDataType>,
    NormalizeInInfer,
    Rank>;

template <typename XDataType,
          typename YDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          index_t Rank>
using DeviceElementwiseForBatchNormInferPtr =
    std::unique_ptr<DeviceElementwiseForBatchNormInfer<XDataType,
                                                       YDataType,
                                                       ScaleDataType,
                                                       BiasDataType,
                                                       MeanVarDataType,
                                                       Rank>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
