// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <tuple>
#include <type_traits>
#include "gtest/gtest.h"
#include "ck_tile/host.hpp"
#include "test_gemm_streamk_util.hpp"

using F8   = ck_tile::fp8_t;
using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using BF8  = ck_tile::bf8_t;
using F32  = float;

// Layouts
using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// Persistence
using Persistent    = std::true_type;
using NonPersistent = std::false_type;

// Pipelines
using Mem    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::Mem>;
using CompV3 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV3>;
using CompV4 = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV4>;

// Reduction Strategies
using Atomic = ck_tile::integral_constant<ck_tile::StreamKReductionStrategy,
                                          ck_tile::StreamKReductionStrategy::Atomic>;
using Linear = ck_tile::integral_constant<ck_tile::StreamKReductionStrategy,
                                          ck_tile::StreamKReductionStrategy::Linear>;
using Tree   = ck_tile::integral_constant<ck_tile::StreamKReductionStrategy,
                                          ck_tile::StreamKReductionStrategy::Tree>;

using I16  = ck_tile::number<16>;
using I32  = ck_tile::number<32>;
using I128 = ck_tile::number<128>;
using I256 = ck_tile::number<256>;

// clang-format off

// ========================== CompV3 Pipeline ==========================

// Atomics
using KernelTypesStreamKFp16PersistentAtomicCompV3 = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  M_WaveTile  N_WaveTile  K_WaveTile  Persistent    Pipeline  ReductionStrategy

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,       Persistent,   CompV3,   Atomic>
>;

using KernelTypesStreamKBf16PersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic>
>;

using KernelTypesStreamKBf8PersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic>
>;

using KernelTypesStreamKFp8PersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Atomic>
>;

using KernelTypesStreamKFp16NonPersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Atomic>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Atomic>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Atomic>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Atomic>
>;

using KernelTypesStreamKBf16NonPersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic>
>;

using KernelTypesStreamKBf8NonPersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic>
>;

using KernelTypesStreamKFp8NonPersistentAtomicCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic>,
    std::tuple<    Row,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic>,
    std::tuple<    Col,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic>,
    std::tuple<    Col,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Atomic>
>;

// Linear
using KernelTypesStreamKFp16PersistentLinearCompV3 = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  M_WaveTile  N_WaveTile  K_WaveTile  Persistent    Pipeline  ReductionStrategy

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I16,       I16,       I16,     Persistent,   CompV3,   Linear>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Linear>
>;

using KernelTypesStreamKBf16PersistentLinearCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Linear>
>;

using KernelTypesStreamKBf8PersistentLinearCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Linear>
>;

using KernelTypesStreamKFp8PersistentLinearCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Linear>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Linear>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Linear>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Linear>
>;

using KernelTypesStreamKFp16NonPersistentLinearCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Linear>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Linear>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Linear>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Linear>
>;

using KernelTypesStreamKBf16NonPersistentLinearCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Linear>
>;

using KernelTypesStreamKBf8NonPersistentLinearCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Linear>
>;

using KernelTypesStreamKFp8NonPersistentLinearCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Linear>,
    std::tuple<    Row,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Linear>,
    std::tuple<    Col,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Linear>,
    std::tuple<    Col,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Linear>
>;

// Tree
using KernelTypesStreamKFp16PersistentTreeCompV3 = ::testing::Types<
//                ALayout  BLayout  CLayout   ADataType  BDataType  AccDataType  CDataType  M_MacroTile  N_MacroTile  K_MacroTile  M_WaveTile  N_WaveTile  K_WaveTile  Persistent    Pipeline  ReductionStrategy

    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Tree>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Tree>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I16,       I16,       I16,     Persistent,   CompV3,   Tree>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Tree>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,   CompV3,   Tree>
>;

using KernelTypesStreamKBf16PersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree>
>;

using KernelTypesStreamKBf8PersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree>
>;

using KernelTypesStreamKFp8PersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree>,
    std::tuple<    Row,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree>,
    std::tuple<    Col,     Col,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree>,
    std::tuple<    Col,     Row,     Row,       F8,        F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    Persistent,   CompV3,   Tree>
>;

using KernelTypesStreamKFp16NonPersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Tree>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Tree>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Tree>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV3,   Tree>
>;

using KernelTypesStreamKBf16NonPersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,       BF16,      BF16,        F32,       BF16,         I256,        I256,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree>
>;

using KernelTypesStreamKBf8NonPersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Col,     Row,        BF8,      BF8,        F32,       BF16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree>
>;

using KernelTypesStreamKFp8NonPersistentTreeCompV3 = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree>,
    std::tuple<    Row,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree>,
    std::tuple<    Col,     Col,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree>,
    std::tuple<    Col,     Row,     Row,       F8,         F8,        F32,        F16,         I128,        I128,        I32,         I32,       I32,       I16,    NonPersistent,   CompV3,   Tree>
>;

// ============================= Other Pipelines =============================

using KernelTypesStreamKPipelines = ::testing::Types<
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,      Mem,      Atomic>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   Mem,      Tree>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,      Mem,      Linear>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,      CompV4,   Atomic>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     NonPersistent,   CompV4,   Tree>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,       F16,         I256,        I256,        I32,         I32,       I32,       I16,     Persistent,      CompV4,   Linear>
>;
// clang-format on
