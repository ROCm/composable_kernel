// Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ck_tile/builder/testing/tensor_memory_manager.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/types.hpp"

namespace {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::testing;

using ::testing::IsNull;

struct ConvSignature
{
    int spatial_dim;
    ckb::ConvDirection direction;
    ckb::GroupConvLayout layout;
    ckb::DataType data_type;
    ckb::ElementwiseOperation elementwise_operation;
    ckb::GroupConvDeviceOp device_operation;
};
static_assert(ckb::ConvSignatureDescriptor<ConvSignature>);

TEST(TensorMemoryManagerTest, BuffersInitializedToNull)
{
    constexpr ConvSignature kSignature = {
        .spatial_dim           = 2,
        .direction             = ckb::ConvDirection::FORWARD,
        .layout                = ckb::GroupConvLayout2D::NHWGC_GKYXC_NHWGK,
        .data_type             = ckb::DataType::FP16,
        .elementwise_operation = ckb::ElementwiseOperation::PASS_THROUGH,
        .device_operation =
            ckb::FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3,
    };
    static_assert(ckb::ValidConvSignature<kSignature>);

    ckt::TensorMemoryManager<kSignature> manager;

    EXPECT_THAT(manager.input_buf.get(), IsNull());
    EXPECT_THAT(manager.weight_buf.get(), IsNull());
    EXPECT_THAT(manager.output_buf.get(), IsNull());
}

} // namespace
