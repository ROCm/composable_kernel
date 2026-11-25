// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdDataDefault = ConvolutionBackwardDataSpecialization::Default;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename DataType,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wavelet_direct_load_xdl_v3_instances = std::tuple<
    // clang-format off
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial,  ALayout, BLayout,    DsLayout, ELayout,   DataType,   DataType,     F32,      DataType, Empty_Tuple,  DataType,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true, 256, 256,    256,   128,    128,   8,   8,   16,   16,       8,       4,       S<16, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<2, 16, 8>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            2,     S<1, 32, 1, 8>, 8>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial,  ALayout, BLayout,    DsLayout, ELayout,   DataType,   DataType,     F32,      DataType, Empty_Tuple,  DataType,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true, 256, 256,    256,   128,     64,   8,   8,   16,   16,       4,       8,       S< 8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<2, 16, 8>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            4,     S<1, 32, 1, 8>, 8>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial,  ALayout, BLayout,    DsLayout, ELayout,   DataType,   DataType,     F32,      DataType, Empty_Tuple,  DataType,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true, 256, 512,    256,   256,     64,   8,   8,   16,   16,       2,      16,       S< 8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<1, 32, 8>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            4,     S<1, 64, 1, 8>, 8>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial,  ALayout, BLayout,    DsLayout, ELayout,   DataType,   DataType,     F32,      DataType, Empty_Tuple,  DataType,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true, 256, 256,    512,    64,     64,   8,   8,   16,   16,       8,       4,       S< 8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4,  8, 8>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            2,     S<1, 64, 1, 4>, 8>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial,  ALayout, BLayout,    DsLayout, ELayout,   DataType,   DataType,     F32,      DataType, Empty_Tuple,  DataType,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true, 256, 256,    256,   128,    128,   8,   8,   16,   16,       8,       4,       S<16, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<2, 16, 8>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            2,     S<1, 32, 1, 8>, 8, DataType, DataType, true>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial,  ALayout, BLayout,    DsLayout, ELayout,   DataType,   DataType,     F32,      DataType, Empty_Tuple,  DataType,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true, 256, 512,    256,   256,     64,   8,   8,   16,   16,       2,      16,       S< 8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4,  8, 8>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            4,     S<1, 64, 1, 8>, 8, DataType, DataType, true>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial,  ALayout, BLayout,    DsLayout, ELayout,   DataType,   DataType,     F32,      DataType, Empty_Tuple,  DataType,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true, 256, 256,    256,   128,     64,   8,   8,   16,   16,       8,       4,       S< 8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<2, 16, 8>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            2,     S<1, 32, 1, 8>, 8, DataType, DataType, true>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial,  ALayout, BLayout,    DsLayout, ELayout,   DataType,   DataType,     F32,      DataType, Empty_Tuple,  DataType,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true, 256, 512,    256,   256,    128,   8,   8,   16,   16,       2,      16,       S<16, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<1, 32, 8>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            8,     S<1, 32, 1,16>, 8, DataType, DataType, true>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename DataType,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wavelet_xdl_v3_instances_part1 = std::tuple<
    // clang-format off
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, DataType, DataType, F32, DataType, Empty_Tuple, DataType, PassThrough, PassThrough, PassThrough, ConvSpec, true, true, 256, 256,  64,  64, 128, 8, 8, 16, 16, 2, 2, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4,  8, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 8, 1, 2, 1, S<1, 64, 1, 4>, 8, DataType, DataType, true, false>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, DataType, DataType, F32, DataType, Empty_Tuple, DataType, PassThrough, PassThrough, PassThrough, ConvSpec, true, true, 256, 256,  64, 128, 128, 8, 8, 16, 16, 2, 4, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<2, 16, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 8, 1, 1, 2, S<1, 32, 1, 8>, 8, DataType, DataType, true, true>,
    DeviceGroupedConvBwdDataMultipleD_Wavelet_Xdl_CShuffleV3<NDimSpatial, ALayout, BLayout, DsLayout, ELayout, DataType, DataType, F32, DataType, Empty_Tuple, DataType, PassThrough, PassThrough, PassThrough, ConvSpec, true, true, 256, 256, 128, 128, 128, 8, 8, 16, 16, 2, 8, S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<2, 16, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 8, 1, 1, 2, S<1, 64, 1, 4>, 8, DataType, DataType, true, true>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
