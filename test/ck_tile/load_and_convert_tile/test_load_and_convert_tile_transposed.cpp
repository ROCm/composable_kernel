// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_load_and_convert_tile_util.hpp"

using TestTypes = ::testing::Types<std::tuple<ck_tile::half_t, ck_tile::half_t, std::true_type>,
                                   std::tuple<ck_tile::half_t, ck_tile::fp8_t, std::true_type>,
                                   std::tuple<ck_tile::fp8_t, ck_tile::half_t, std::true_type>,
                                   std::tuple<ck_tile::bf16_t, ck_tile::bf16_t, std::true_type>,
                                   std::tuple<ck_tile::bf16_t, ck_tile::fp8_t, std::true_type>,
                                   std::tuple<ck_tile::fp8_t, ck_tile::bf16_t, std::true_type>,
                                   std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, std::true_type>>;

TYPED_TEST_SUITE(TestLoadAndConvert, TestTypes);

TYPED_TEST(TestLoadAndConvert, TestTransposed) { this->RunTest(); }
