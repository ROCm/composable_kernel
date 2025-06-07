// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "device_grouped_conv_bwd_weight_dl_v4.hpp"
#include "common.hpp"

#define ENABLE_CONV_FACTORY 1

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

using ALayout = ck::tensor_layout::convolution::GNHWC;
using BLayout = ck::tensor_layout::convolution::GKYXC;
using ELayout = ck::tensor_layout::convolution::GNHWK;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<NDimSpatial,
                                                                 256,
                                                                  ALayout,
                                                                  BLayout,
                                                                  ELayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  OutDataType,
                                                                  AccDataType,
                                                                  S<28, 28>,
                                                                  5,
                                                                  ck::Tuple<S<1,1>, S<1,1>, S<2,2>>,
                                                                  InElementOp,
                                                                  WeiElementOp,
                                                                  OutElementOp,
                                                                  2,  // N batch
                                                                  1,  // NumWavePerTile
                                                                  4,  // InScalarPerVector
                                                                  4,  // OutScalarPerVector
                                                                  2,  // DstScalarPerVector
                                                                  false>;

using DeviceConvBwdWeightFactory = std::tuple<   
    //                                                   NDimSpatial BlockSize InLayout WeiLayout OutLayout  InDataType WeiDataType  OutDataType     AccDatType BlockTileSize FilterSize  FilterParam(dilation, stride, pad)                                        NBatch NumWavePerTile InScalarPerVector OutScalarPerVector DstScalarPerVector  RequirePadding
      ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             2,                2,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 16,    1,             1,                1,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             2,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             2,                1,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1,     4,             8,                8,                 1,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1,     2,             4,                4,                 1,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 16,    1,             1,                1,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1,     4,             8,                4,                 1,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                2,                 2,                 false>

     // 28 x 5 x 1
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>
    
     // 14 x 5 x 1
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>

     // 7 x 5 x 1
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             1,                1,                 8,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             1,                1,                 4,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      5,          ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             1,                1,                 2,                 false>

     // 56 x 5 x 2
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             4,                2,                 2,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             4,                2,                 2,                 false, 2>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             2,                2,                 2,                 false, 2>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false, 2>

     // 14 x 5 x 2
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,      ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                1,                 4,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             2,                1,                 8,                 false>
     , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    5,          ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                1,                 4,                 false>

     // 112 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             4,                4,                 2,                 false, 2>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 4>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 2>
   
    // 56 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             4,                4,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             2,                2,                 2,                 false, 2>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     2,             2,                2,                 2,                 false, 4>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<56, 56>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 2>

    // 28 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                4,                 2,                 false>

    // 14 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             2,                2,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             2,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<14, 14>,    3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             2,                2,                 4,                 false>

    // 7 x 3 x 1
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 8,     1,             1,               1,                 8,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             1,               1,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<7, 7>,      3,          ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             1,               1,                 2,                 false>

    // 112 x 3 x 2
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             4,                2,                 2,                 false, 2>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 2>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<112, 112>,  3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     4,             2,                2,                 2,                 false, 4>

    // 28 x 3 x 3
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2,     1,             4,                2,                 2,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  64,        ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             4,                2,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  128,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             4,                2,                 4,                 false>
    , ck::tensor_operation::device::DeviceGroupedConvBwdWeightDlV4<2,  256,       ALayout,  BLayout,  ELayout,  InDataType, WeiDataType, OutDataType, AccDataType, S<28, 28>,    3,          ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 4,     1,             4,                2,                 4,                 false>
>;

template <ck::index_t NDimSpatial>
using HostConvBwdWeightInstance = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                                     InDataType,
                                                                                     WeiDataType,
                                                                                     OutDataType,
                                                                                     InElementOp,
                                                                                     WeiElementOp,
                                                                                     OutElementOp>;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return 1;
    }

    switch(conv_param.num_dim_spatial_)
    {
    case 1: break;//return !run_grouped_conv_bwd_weight<1>(config, conv_param);
    case 2: return !run_grouped_conv_bwd_weight<2>(config, conv_param);
    case 3: break;//return !run_grouped_conv_bwd_weight<3>(config, conv_param);
    default: break;
    }

    return 1;
}
