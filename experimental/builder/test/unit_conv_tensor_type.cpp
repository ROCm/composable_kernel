// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>

#include "ck_tile/builder/factory/helpers/conv_tensor_type.hpp"

namespace {

namespace ckb = ck_tile::builder;
using ck_tile::builder::factory::internal::DataTypeToCK;

TEST(ConvTensorType, AssignsTypesForFP16)
{
    using CKType = DataTypeToCK<ckb::DataType::FP16>::type;
    EXPECT_TRUE((std::is_same_v<CKType, ck::half_t>));
}

TEST(ConvTensorType, AssignsTypesForBF16)
{
    using CKType = DataTypeToCK<ckb::DataType::BF16>::type;
    EXPECT_TRUE((std::is_same_v<CKType, ck::bhalf_t>));
}

TEST(ConvTensorType, AssignsTypesForFP32)
{
    using CKType = DataTypeToCK<ckb::DataType::FP32>::type;
    EXPECT_TRUE((std::is_same_v<CKType, float>));
}

TEST(ConvTensorType, AssignsTypesForINT32)
{
    using CKType = DataTypeToCK<ckb::DataType::INT32>::type;
    EXPECT_TRUE((std::is_same_v<CKType, int32_t>));
}

TEST(ConvTensorType, AssignsTypesForI8)
{
    using CKType = DataTypeToCK<ckb::DataType::I8>::type;
    EXPECT_TRUE((std::is_same_v<CKType, int8_t>));
}

TEST(ConvTensorType, AssignsTypesForFP8)
{
    using CKType = DataTypeToCK<ckb::DataType::FP8>::type;
    EXPECT_TRUE((std::is_same_v<CKType, ck::f8_t>));
}

} // namespace
