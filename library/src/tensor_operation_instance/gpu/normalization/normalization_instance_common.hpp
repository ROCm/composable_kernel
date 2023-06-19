// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f16_instances =
    // clang-format off
    std::tuple <
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorDim, GammaSrcVectorSize, BetaSrcVectorDim, BetaSrcVectorSize, YDstVectorSize>
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2>,   // irregular size
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4>,   // irregular size
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 16, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 16, 1, 8, 1, 8, 1, 8, 8>
        // clang-format on
        >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f16_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // clang-format on
    >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f32_instances = std::tuple<
    // clang-format off
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorSize, BetaSrcVectorSize, YDstVectorSize>
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2>,   // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 2, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 4, 1, 4, 1, 4, 4>
    // clang-format on
    >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f32_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // clang-format on
    >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f16_f32_f32_f16_instances = std::tuple<
    // clang-format off
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorSize, BetaSrcVectorSize, YDstVectorSize>
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2>,   // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 2, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 4, 1, 4, 1, 4, 4>
    // clang-format on
    >;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_normalization_f16_f32_f32_f16_generic_instance = std::tuple<
    // clang-format off
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
