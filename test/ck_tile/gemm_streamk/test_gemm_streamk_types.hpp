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
    using rrr = ::testing::Types<std::tuple<    Row,     Row,     Row, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using rrc = ::testing::Types<std::tuple<    Row,     Row,     Col, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using rcr = ::testing::Types<std::tuple<    Row,     Col,     Row, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using rcc = ::testing::Types<std::tuple<    Row,     Col,     Col, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using crr = ::testing::Types<std::tuple<    Col,     Row,     Row, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using crc = ::testing::Types<std::tuple<    Col,     Row,     Col, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using ccr = ::testing::Types<std::tuple<    Col,     Col,     Row, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    using ccc = ::testing::Types<std::tuple<    Col,     Col,     Col, ADataType, BDataType, AccDataType, CDataType, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>>;
    // clang-format on
};

#include "test_gemm_streamk_types_fp16.hpp"
#include "test_gemm_streamk_types_bf16.hpp"
