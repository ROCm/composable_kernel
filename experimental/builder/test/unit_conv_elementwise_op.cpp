// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>

#include "ck_tile/builder/factory/helpers/conv_elementwise_op.hpp"

namespace {

using ::ck_tile::builder::factory::internal::ElementwiseOps;
using enum ::ck_tile::builder::ElementwiseOperation;

TEST(ConvElementwiseOp, AssignsOpsForPassThrough)
{
    using Ops = ElementwiseOps<PASS_THROUGH>;

    EXPECT_TRUE(
        (std::is_same_v<Ops::AElementwiseOp, ck::tensor_operation::element_wise::PassThrough>));
    EXPECT_TRUE(
        (std::is_same_v<Ops::BElementwiseOp, ck::tensor_operation::element_wise::PassThrough>));
    EXPECT_TRUE(
        (std::is_same_v<Ops::CDEElementwiseOp, ck::tensor_operation::element_wise::PassThrough>));
}

TEST(ConvElementwiseOp, AssignsOpsForScale)
{
    using Ops = ElementwiseOps<SCALE>;

    EXPECT_TRUE(
        (std::is_same_v<Ops::AElementwiseOp, ck::tensor_operation::element_wise::PassThrough>));
    EXPECT_TRUE(
        (std::is_same_v<Ops::BElementwiseOp, ck::tensor_operation::element_wise::PassThrough>));
    EXPECT_TRUE((std::is_same_v<Ops::CDEElementwiseOp, ck::tensor_operation::element_wise::Scale>));
}

} // namespace
