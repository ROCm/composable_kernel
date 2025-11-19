// Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ck_tile/builder/testing/tensor_memory_manager.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ::testing::IsNull;

struct ConvSignature
{
    int spatial_dim;
    ckb::ConvDirection direction;
    ckb::GroupConvLayout layout;
    ckb::DataType data_type;
    ckb::ElementwiseOperation elementwise_operation;
};
static_assert(ckb::ConvSignatureDescriptor<ConvSignature>);

TEST(TensorMemoryManagerTest, BuffersInitializedToNull)
{
    constexpr ConvSignature signature = {
        .spatial_dim           = 2,
        .direction             = ckb::ConvDirection::FORWARD,
        .layout                = ckb::GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
        .data_type             = ckb::DataType::FP16,
        .elementwise_operation = ckb::ElementwiseOperation::PASS_THROUGH,
    };
    static_assert(ckb::ValidConvSignature<signature>);

    ckt::TensorMemoryManager<signature> manager;

    EXPECT_THAT(manager.input_buf.get(), IsNull());
    EXPECT_THAT(manager.weight_buf.get(), IsNull());
    EXPECT_THAT(manager.output_buf.get(), IsNull());
}

} // namespace
