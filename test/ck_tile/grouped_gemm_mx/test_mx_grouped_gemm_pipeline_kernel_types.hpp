// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <type_traits>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_mx_grouped_gemm_util.hpp"

using F4    = ck_tile::pk_fp4_t;
using F8    = ck_tile::fp8_t;
using BF8   = ck_tile::bf8_t;
using F16   = ck_tile::half_t;
using F32   = float;
using BF16  = ck_tile::bf16_t;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
using True  = ck_tile::bool_constant<true>;
using False = ck_tile::bool_constant<false>;
using E8M0  = ck_tile::e8m0_t;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

using CompTDMV1 = ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompTDMV1>;
using CompTDMV2 = ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompTDMV2>;
using CompAsync = ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompAsync>;
using CompEightWaves =
    ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompEightWaves>;
using WeightPreshuffle =
    ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::WeightPreshuffle>;

using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I64  = ck_tile::number<64>;
using I128 = ck_tile::number<128>;
using I256 = ck_tile::number<256>;
using I512 = ck_tile::number<512>;

template <ck_tile::index_t N>
using ScaleBS = ck_tile::integral_constant<ck_tile::index_t, N>;

// clang-format off
// MX GEMM kernel types using TDM pipeline with scale support
// Tuple format:
//         ALayout, BLayout, CLayout, ADataType, BDataType, AScaleDataType, BScaleDataType, AccDataType, CDataType,  Persistent, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, N_TileSize, PipelineType
using KernelTypesMxGemmCompTDMWmma = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I128, I32, I32,  CompTDMV1,  ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       BF8,       F8,    E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I128, I32, I32,  CompTDMV1,  ScaleBS<32>>,
    std::tuple<    Row,     Row,     Row,       BF8,       F8,    E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I128, I32, I32,  CompTDMV1,  ScaleBS<32>>,
    std::tuple<    Col,     Row,     Row,       F8,        BF8,   E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I128, I32, I32,  CompTDMV1,  ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I128, I32, I32,  CompTDMV2,  ScaleBS<32>>, 
    std::tuple<    Row,     Col,     Row,       BF8,       F8,    E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I128, I32, I32,  CompTDMV2,  ScaleBS<32>>,
    std::tuple<    Row,     Row,     Row,       BF8,       F8,    E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I128, I32, I32,  CompTDMV2,  ScaleBS<32>>,
    std::tuple<    Col,     Row,     Row,       F8,        BF8,   E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I128, I32, I32,  CompTDMV2,  ScaleBS<32>>
>;

using KernelTypesMxGemmCompAsync = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I256, I16, I16,        CompAsync, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,      False,        I64,  I64,  I256, I16, I16,        CompAsync, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       True,        I64,  I64,  I256, I16, I16,        CompAsync, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       True,        I64,  I64,  I256, I16, I16,        CompAsync, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,      False,       I128, I256,  I128, I16, I16,   CompEightWaves, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,      False,       I128, I256,  I128, I16, I16,   CompEightWaves, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       True,       I128, I256,  I128, I16, I16,   CompEightWaves, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       True,       I128, I256,  I128, I16, I16,   CompEightWaves, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,      False,       I128, I256,  I256, I16, I16, WeightPreshuffle, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,      False,       I128, I512,  I256, I16, I16, WeightPreshuffle, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       True,       I128, I256,  I256, I16, I16, WeightPreshuffle, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       True,       I128, I512,  I256, I16, I16, WeightPreshuffle, ScaleBS<32>>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,      False,       I128, I256,  I256, I16, I16, WeightPreshuffle, ScaleBS<32>, True>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,      False,       I128, I512,  I256, I16, I16, WeightPreshuffle, ScaleBS<32>, True>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,    E8M0,  E8M0,      F32,       F16,       True,       I128, I256,  I256, I16, I16, WeightPreshuffle, ScaleBS<32>, True>,
    std::tuple<    Row,     Col,     Row,       F4,        F4,    E8M0,  E8M0,      F32,       F16,       True,       I128, I512,  I256, I16, I16, WeightPreshuffle, ScaleBS<32>, True>
>;
// clang-format on
