// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_streamk_util.hpp"

using F16  = ck_tile::half_t;
using F32  = float;
using BF16 = ck_tile::bf16_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using Mem    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;
using CompV4 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV4>;

using Persistent    = std::true_type;
using NonPersistent = std::false_type;

using I1   = ck_tile::number<1>;
using I2   = ck_tile::number<2>;
using I4   = ck_tile::number<4>;
using I8   = ck_tile::number<8>;
using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I64  = ck_tile::number<64>;
using I128 = ck_tile::number<128>;
using I256 = ck_tile::number<256>;

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename M_MacroTile,
          typename N_MacroTile,
          typename K_MacroTile,
          typename M_Warps,
          typename N_Warps,
          typename K_Warps,
          typename M_MmaTile,
          typename N_MmaTile,
          typename K_MmaTile,
          typename PipelineType,
          typename Persistent>
struct Layouts
{
    // clang-format off
    // Create all combinations of A, B, Acc, C layouts
    //                                      ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent
    using RRR = ::testing::Types<std::tuple<    Row,     Row,     Row, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using RRC = ::testing::Types<std::tuple<    Row,     Row,     Col, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using RCR = ::testing::Types<std::tuple<    Row,     Col,     Row, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using RCC = ::testing::Types<std::tuple<    Row,     Col,     Col, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using CRR = ::testing::Types<std::tuple<    Col,     Row,     Row, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using CRC = ::testing::Types<std::tuple<    Col,     Row,     Col, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using CCR = ::testing::Types<std::tuple<    Col,     Col,     Row, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using CCC = ::testing::Types<std::tuple<    Col,     Col,     Col, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    // clang-format on
};

// clang-format off
// Here we use macros to generate a large number of parameter sets for different test configurations.
// One parameter set is intended to be be implemented per .cpp file to keep the compile time down.
// The naming convention is as follows:
//        __________________________________________________    ____________________________________________________________________________________
//       |                Parameter Name                    |  |                           Parameter Value Type                                     |
// using F16_RRR_Mem_128x128x32_2x2x1_32x32x16_NonPersistent = F16Layouts<I128, I128, I32, I2,  I2,    I1,    I32, I32, I16, Mem, NonPersistent>::RRR;
//        /   |     \         \      \      \       \               |       |     |    |    \     \     \      \     \    \    \         \          \
//     DATA LAYOUT  PIPELINE  MACRO  WARPS   MMA    PERSISTENT    LAYOUT   MACRO MACRO MACRO WARPS WARPS WARPS MMA  MMA  MMA  PIPELINE  PERSISTENT   LAYOUT
//     TYPE         TYPE      TILE   MxNxK   TILE   TYPE          CLASS    TILE  TILE  TILE  M     N     K     TILE TILE TILE TYPE      TYPE
//                            MxNxK          MxNxK                         M      N    K                       M     N   K    
// 
// The example options for each field are:
//  - DATA_TYPE: F16, BF16
//  - LAYOUT: RRR, RRC, RCR, RCC, CRR, CRC, CCR, CCC
//  - PIPELINE_TYPE: Mem, CompV3, CompV4
//  - M_MACRO_TILE: 128, 256, etc
//  - N_MACRO_TILE: 128, 256, etc
//  - K_MACRO_TILE: 32, 64, 128, etc
//  - M_WARPS: 2, 4, 1
//  - N_WARPS: 2, 1, 4
//  - K_WARPS: 1
//  - M_MMA_TILE: 32, 16
//  - N_MMA_TILE: 32, 16
//  - K_MMA_TILE: 16
//  - PERSISTENT_TYPE: NonPersistent, Persistent

// Macro to concatenate the parameter name
// E.g. F16_RRR_Mem_128x128x32_2x2x1_32x32x16_NonPersistent
#define CONCATENATE_PARAM_NAME(DATA_TYPE, LAYOUT, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DATA_TYPE##_##LAYOUT##_##PIPELINE_TYPE##_##M_MACRO_TILE##x##N_MACRO_TILE##x##K_MACRO_TILE##_##M_WARPS##x##N_WARPS##x##K_WARPS##_##M_MMA_TILE##x##N_MMA_TILE##x##K_MMA_TILE##_##PERSISTENT

// Macro to get the parameter value type
// E.g. F16Layouts<I128, I128, I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>::RRR
#define CONCATENATE_PARAM_VALUE(LAYOUTS_CLASS, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PIPELINE_TYPE, PERSISTENT, LAYOUT) \
    LAYOUTS_CLASS<I##M_MACRO_TILE, I##N_MACRO_TILE, I##K_MACRO_TILE, I##M_WARPS, I##N_WARPS, I##K_WARPS, I##M_MMA_TILE, I##N_MMA_TILE, I##K_MMA_TILE, PIPELINE_TYPE, PERSISTENT>::LAYOUT

// Macro to declare a single parameter set, consisting of a parameter name and value type
#define DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, LAYOUT, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    using CONCATENATE_PARAM_NAME(DATA_TYPE, LAYOUT, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) = \
          CONCATENATE_PARAM_VALUE(LAYOUTS_CLASS, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PIPELINE_TYPE, PERSISTENT, LAYOUT);

// Macro to declare all layout combinations for a given set of parameters
#define DECLARE_PARAMS_ALL_LAYOUTS(LAYOUTS_CLASS, DATA_TYPE, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, RRR, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, RRC, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, RCR, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, RCC, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, CRR, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, CRC, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, CCR, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAM(LAYOUTS_CLASS, DATA_TYPE, CCC, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT)

#include "test_gemm_streamk_types_fp16.hpp"
#include "test_gemm_streamk_types_bf16.hpp"
