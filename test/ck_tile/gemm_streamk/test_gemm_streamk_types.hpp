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

using I1  = ck_tile::number<1>;
using I2  = ck_tile::number<2>;
using I4  = ck_tile::number<4>;
using I8  = ck_tile::number<8>;
using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I64  = ck_tile::number<64>;
using I128  = ck_tile::number<128>;
using I256 = ck_tile::number<256>;

// clang-format off

namespace detail
{
    template<typename Lhs, typename Rhs>
    struct combine;

    template<typename... Lhs, typename... Rhs>
    struct combine<::testing::Types<Lhs...>, ::testing::Types<Rhs...>>
    {
        using type = ::testing::Types<Lhs..., Rhs...>;
    };

    template<typename Lhs, typename Rhs>
    using combine_t = typename combine<Lhs, Rhs>::type;
}

template<typename ADataType,
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
};

// template<typename M_MacroTile,
//          typename N_MacroTile,
//          typename K_MacroTile,
//          typename M_Warps,
//          typename N_Warps,
//          typename K_Warps,
//          typename M_MmaTile,
//          typename N_MmaTile,
//          typename K_MmaTile,
//          typename PipelineType,
//          typename Persistent>
// struct F16Layouts
// {
//     // For CDNA, we support [A, B, Acc, C] = [f16, f16, f32, f16] and [f16, f16, f32, f32]:
//     using f16_f16_f32_f16 = Layouts<F16, F16, F32, F16, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>;
//     using f16_f16_f32_f32 = Layouts<F16, F16, F32, F16, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>;
//     using rrr = detail::combine_t<typename f16_f16_f32_f16::rrr, typename f16_f16_f32_f32::rrr>;
//     using rrc = detail::combine_t<typename f16_f16_f32_f16::rrc, typename f16_f16_f32_f32::rrc>;
//     using rcr = detail::combine_t<typename f16_f16_f32_f16::rcr, typename f16_f16_f32_f32::rcr>;
//     using rcc = detail::combine_t<typename f16_f16_f32_f16::rcc, typename f16_f16_f32_f32::rcc>;
//     using crr = detail::combine_t<typename f16_f16_f32_f16::crr, typename f16_f16_f32_f32::crr>;
//     using crc = detail::combine_t<typename f16_f16_f32_f16::crc, typename f16_f16_f32_f32::crc>;
//     using ccr = detail::combine_t<typename f16_f16_f32_f16::ccr, typename f16_f16_f32_f32::ccr>;
//     using ccc = detail::combine_t<typename f16_f16_f32_f16::ccc, typename f16_f16_f32_f32::ccc>;
// };

// template<typename PipelineType,
//          typename Persistent>
// struct F16Set
// {
//     // 32x32x16
//     // 2x2x1
//     using f16_128x128x32_2x2x1_32x32x16 =  F16Layouts<I128, I128,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x128x64_2x2x1_32x32x16 =  F16Layouts<I128, I128,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x128x128_2x2x1_32x32x16 = F16Layouts<I128, I128, I128, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x128x32_2x2x1_32x32x16 =  F16Layouts<I256, I128,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x128x64_2x2x1_32x32x16 =  F16Layouts<I256, I128,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x256x32_2x2x1_32x32x16 =  F16Layouts<I128, I256,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x256x64_2x2x1_32x32x16 =  F16Layouts<I128, I256,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x256x32_2x2x1_32x32x16 =  F16Layouts<I256, I256,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x256x64_2x2x1_32x32x16 =  F16Layouts<I256, I256,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;

//     // 32x32x16
//     // 4x4x1
//     using f16_128x128x32_4x1x1_32x32x16 =  F16Layouts<I128, I128,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x128x64_4x1x1_32x32x16 =  F16Layouts<I128, I128,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x128x128_4x1x1_32x32x16 = F16Layouts<I128, I128, I128, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x128x32_4x1x1_32x32x16 =  F16Layouts<I256, I128,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x128x64_4x1x1_32x32x16 =  F16Layouts<I256, I128,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x256x32_4x1x1_32x32x16 =  F16Layouts<I128, I256,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x256x64_4x1x1_32x32x16 =  F16Layouts<I128, I256,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x256x32_4x1x1_32x32x16 =  F16Layouts<I256, I256,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x256x64_4x1x1_32x32x16 =  F16Layouts<I256, I256,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;

//     // 32x32x16
//     // 1x4x1
//     using f16_128x128x32_1x4x1_32x32x16 =  F16Layouts<I128, I128,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x128x64_1x4x1_32x32x16 =  F16Layouts<I128, I128,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x128x128_1x4x1_32x32x16 = F16Layouts<I128, I128, I128, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x128x32_1x4x1_32x32x16 =  F16Layouts<I256, I128,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x128x64_1x4x1_32x32x16 =  F16Layouts<I256, I128,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x256x32_1x4x1_32x32x16 =  F16Layouts<I128, I256,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_128x256x64_1x4x1_32x32x16 =  F16Layouts<I128, I256,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x256x32_1x4x1_32x32x16 =  F16Layouts<I256, I256,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
//     using f16_256x256x64_1x4x1_32x32x16 =  F16Layouts<I256, I256,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;

//     // 16x16x16
//     // 2x2x1
//     using f16_128x128x32_2x2x1_16x16x16 =  F16Layouts<I128, I128,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x128x64_2x2x1_16x16x16 =  F16Layouts<I128, I128,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x128x128_2x2x1_16x16x16 = F16Layouts<I128, I128, I128, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x128x32_2x2x1_16x16x16 =  F16Layouts<I256, I128,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x128x64_2x2x1_16x16x16 =  F16Layouts<I256, I128,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x256x32_2x2x1_16x16x16 =  F16Layouts<I128, I256,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x256x64_2x2x1_16x16x16 =  F16Layouts<I128, I256,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x256x32_2x2x1_16x16x16 =  F16Layouts<I256, I256,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x256x64_2x2x1_16x16x16 =  F16Layouts<I256, I256,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;

//     // 16x16x16
//     // 4x1x1
//     using f16_128x128x32_4x1x1_16x16x16 =  F16Layouts<I128, I128,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x128x64_4x1x1_16x16x16 =  F16Layouts<I128, I128,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x128x128_4x1x1_16x16x16 = F16Layouts<I128, I128, I128, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x128x32_4x1x1_16x16x16 =  F16Layouts<I256, I128,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x128x64_4x1x1_16x16x16 =  F16Layouts<I256, I128,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x256x32_4x1x1_16x16x16 =  F16Layouts<I128, I256,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x256x64_4x1x1_16x16x16 =  F16Layouts<I128, I256,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x256x32_4x1x1_16x16x16 =  F16Layouts<I256, I256,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x256x64_4x1x1_16x16x16 =  F16Layouts<I256, I256,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;

//     // 16x16x16
//     // 1x4x1
//     using f16_128x128x32_1x4x1_16x16x16 =  F16Layouts<I128, I128,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x128x64_1x4x1_16x16x16 =  F16Layouts<I128, I128,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x128x128_1x4x1_16x16x16 = F16Layouts<I128, I128, I128, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x128x32_1x4x1_16x16x16 =  F16Layouts<I256, I128,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x128x64_1x4x1_16x16x16 =  F16Layouts<I256, I128,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x256x32_1x4x1_16x16x16 =  F16Layouts<I128, I256,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_128x256x64_1x4x1_16x16x16 =  F16Layouts<I128, I256,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x256x32_1x4x1_16x16x16 =  F16Layouts<I256, I256,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
//     using f16_256x256x64_1x4x1_16x16x16 =  F16Layouts<I256, I256,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
// };

#include "test_gemm_streamk_types_fp16.hpp"

