// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_wmma_pipeline_util.hpp"

using F16 = ck_tile::half_t;
using F32 = float;
using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    //       ALayout,  BLayout, CLayout, ADataType,  BDataType, AccDataType, CDataType
    std::tuple<Row,    Col,     Row,     F16,        F16,        F32,        F16,     
               ck_tile::number<64>, ck_tile::number<128>, ck_tile::number<32>,   // M,N,K Tile
               ck_tile::number<32>, ck_tile::number<64>, ck_tile::number<32>,    // Warp Tile
               ck_tile::number<16>, ck_tile::number<16>, ck_tile::number<16>    // Wmma Tile
               >     // WmmaPipelineType 
    // std::tuple<Row,    Col,     Col,     F16,        F16,        F32,        F16,     
    //            ck_tile::number<64>, ck_tile::number<128>, ck_tile::number<32>,   // M,N,K Tile
    //            ck_tile::number<32>, ck_tile::number<64>, ck_tile::number<32>,    // Warp Tile
    //            ck_tile::number<16>, ck_tile::number<16>, ck_tile::number<16>
    //            >
//     std::tuple<Col,    Col,     Row,     F16,        F16,        F32,        F16,     
//                ck_tile::number<64>, ck_tile::number<128>, ck_tile::number<32>,   // M,N,K Tile
//                ck_tile::number<32>, ck_tile::number<64>, ck_tile::number<32>,    // Warp Tile
//                ck_tile::number<16>, ck_tile::number<16>, ck_tile::number<16>,
//                Default>,
//     std::tuple<Col,    Row,     Row,     F16,        F16,        F32,        F16,     
//                ck_tile::number<64>, ck_tile::number<128>, ck_tile::number<32>,   // M,N,K Tile
//                ck_tile::number<32>, ck_tile::number<64>, ck_tile::number<32>,    // Warp Tile
//                ck_tile::number<16>, ck_tile::number<16>, ck_tile::number<16>,
//                Default>,
//     std::tuple<Row,    Row,     Row,     F16,        F16,        F32,        F16,     
//                ck_tile::number<64>, ck_tile::number<128>, ck_tile::number<32>,   // M,N,K Tile
//                ck_tile::number<32>, ck_tile::number<64>, ck_tile::number<32>,    // Warp Tile
//                ck_tile::number<16>, ck_tile::number<16>, ck_tile::number<16>,
//                Default>,
//     std::tuple<Col,    Col,     Col,     F16,        F16,        F32,        F16,     
//                ck_tile::number<64>, ck_tile::number<128>, ck_tile::number<32>,   // M,N,K Tile
//                ck_tile::number<32>, ck_tile::number<64>, ck_tile::number<32>,    // Warp Tile
//                ck_tile::number<16>, ck_tile::number<16>, ck_tile::number<16>,
//                Default>,
//     std::tuple<Col,    Row,     Col,     F16,        F16,        F32,        F16,     
//                ck_tile::number<64>, ck_tile::number<128>, ck_tile::number<32>,   // M,N,K Tile
//                ck_tile::number<32>, ck_tile::number<64>, ck_tile::number<32>,    // Warp Tile
//                ck_tile::number<16>, ck_tile::number<16>, ck_tile::number<16>,
//                Default>,
//     std::tuple<Row,    Row,     Col,     F16,        F16,        F32,        F16,     
//                ck_tile::number<64>, ck_tile::number<128>, ck_tile::number<32>,   // M,N,K Tile
//                ck_tile::number<32>, ck_tile::number<64>, ck_tile::number<32>,    // Warp Tile
//                ck_tile::number<16>, ck_tile::number<16>, ck_tile::number<16>,
//                Default>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileWmmaPipeline, KernelTypes);

#include "test_wmma_pipeline_ut_cases.inc"
