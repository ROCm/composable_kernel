// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_image_to_column_impl.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

template <ck::index_t NDimSpatial, typename InLayout>
using device_image_to_column_bf16_instances = std::tuple<
    // clang-format off
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|       Slice|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|     Lengths|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |            |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |            |          |       |
        DeviceImageToColumnImpl<NDimSpatial, InLayout,       BF16,        BF16,   256,   128,   128, S<128, 128>, S<16, 16>,      8>
    // clang-format on
    >;

template <ck::index_t NDimSpatial, typename InLayout>
using device_image_to_column_f16_instances = std::tuple<
    // clang-format off
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|       Slice|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|     Lengths|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |            |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |            |          |       |
        DeviceImageToColumnImpl<NDimSpatial, InLayout,        F16,         F16,   256,   128,   128, S<128, 128>, S<16, 16>,      8>
    // clang-format on
    >;

template <ck::index_t NDimSpatial, typename InLayout>
using device_image_to_column_f32_instances = std::tuple<
    // clang-format off
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|       Slice|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|     Lengths|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |            |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |            |          |       |
        DeviceImageToColumnImpl<NDimSpatial, InLayout,        F32,         F32,   256,   128,   128, S<128, 128>, S<16, 16>,      4>
    // clang-format on
    >;

template <ck::index_t NDimSpatial, typename InLayout>
using device_image_to_column_i8_instances = std::tuple<
    // clang-format off
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|       Slice|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|     Lengths|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |            |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |            |          |       |
        DeviceImageToColumnImpl<NDimSpatial, InLayout,     int8_t,      int8_t,   256,   256,   256, S<256, 256>, S<16, 16>,     16>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
