// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>

#include "ck_tile/builder/factory/helpers/conv_tensor_type.hpp"

namespace {

namespace ckb = ck_tile::builder;
using ck_tile::builder::factory::internal::ConvTensorTypes;

TEST(ConvTensorType, AssignsTypesForFP16)
{
    using Types = ConvTensorTypes<ckb::DataType::FP16>;

    EXPECT_TRUE((std::is_same_v<Types::ADataType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<Types::BDataType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<Types::EDataType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<Types::AccDataType, float>));
    EXPECT_TRUE((std::is_same_v<Types::AComputeType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<Types::BComputeType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<Types::CShuffleDataType, ck::half_t>));
}

TEST(ConvTensorType, AssignsTypesForBF16)
{
    using Types = ConvTensorTypes<ckb::DataType::BF16>;

    EXPECT_TRUE((std::is_same_v<Types::ADataType, ck::bhalf_t>));
    EXPECT_TRUE((std::is_same_v<Types::BDataType, ck::bhalf_t>));
    EXPECT_TRUE((std::is_same_v<Types::EDataType, ck::bhalf_t>));
    EXPECT_TRUE((std::is_same_v<Types::AccDataType, float>));
    EXPECT_TRUE((std::is_same_v<Types::AComputeType, ck::bhalf_t>));
    EXPECT_TRUE((std::is_same_v<Types::BComputeType, ck::bhalf_t>));
    EXPECT_TRUE((std::is_same_v<Types::CShuffleDataType, ck::bhalf_t>));
}

TEST(ConvTensorType, AssignsTypesForFP32)
{
    using Types = ConvTensorTypes<ckb::DataType::FP32>;

    EXPECT_TRUE((std::is_same_v<Types::ADataType, float>));
    EXPECT_TRUE((std::is_same_v<Types::BDataType, float>));
    EXPECT_TRUE((std::is_same_v<Types::EDataType, float>));
    EXPECT_TRUE((std::is_same_v<Types::AccDataType, float>));
    EXPECT_TRUE((std::is_same_v<Types::AComputeType, float>));
    EXPECT_TRUE((std::is_same_v<Types::BComputeType, float>));
    EXPECT_TRUE((std::is_same_v<Types::CShuffleDataType, float>));
}

TEST(ConvTensorType, AssignsTypesForI8)
{
    using Types = ConvTensorTypes<ckb::DataType::I8>;

    EXPECT_TRUE((std::is_same_v<Types::ADataType, int8_t>));
    EXPECT_TRUE((std::is_same_v<Types::BDataType, int8_t>));
    EXPECT_TRUE((std::is_same_v<Types::EDataType, int8_t>));
    EXPECT_TRUE((std::is_same_v<Types::AccDataType, int32_t>));
    EXPECT_TRUE((std::is_same_v<Types::AComputeType, int8_t>));
    EXPECT_TRUE((std::is_same_v<Types::BComputeType, int8_t>));
    EXPECT_TRUE((std::is_same_v<Types::CShuffleDataType, int8_t>));
}

TEST(ConvTensorType, AssignsTypesForFP8)
{
    using Types = ConvTensorTypes<ckb::DataType::FP8>;

    EXPECT_TRUE((std::is_same_v<Types::ADataType, ck::f8_t>));
    EXPECT_TRUE((std::is_same_v<Types::BDataType, ck::f8_t>));
    EXPECT_TRUE((std::is_same_v<Types::EDataType, ck::f8_t>));
    EXPECT_TRUE((std::is_same_v<Types::AccDataType, float>));
    EXPECT_TRUE((std::is_same_v<Types::AComputeType, ck::f8_t>));
    EXPECT_TRUE((std::is_same_v<Types::BComputeType, ck::f8_t>));
    EXPECT_TRUE((std::is_same_v<Types::CShuffleDataType, ck::f8_t>));
}

} // namespace
