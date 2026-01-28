// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "impl/conv_signature_types.hpp"
#include "testing_utils.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/tensor_foreach.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::NotNull;

constexpr auto SIGNATURE =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr ckt::Args<SIGNATURE> ARGS = {
    .lengths =
        {
            .batch_size      = 17,
            .groups          = 5,
            .input_channels  = 13,
            .output_channels = 44,
            .image =
                {
                    .width  = 99,
                    .height = 125,
                },
            .filter =
                {
                    .width  = 9,
                    .height = 4,
                },
        },
    .filter_strides     = {.width = 1, .height = 1},
    .filter_dilation    = {.width = 1, .height = 1},
    .input_left_pad     = {.width = 0, .height = 0},
    .input_right_pad    = {.width = 0, .height = 0},
    .a_elementwise_op   = {},
    .b_elementwise_op   = {},
    .cde_elementwise_op = {},
};

using Inputs        = ckt::Inputs<SIGNATURE>;
using Outputs       = ckt::Outputs<SIGNATURE>;
using UniqueInputs  = ckt::UniqueInputs<SIGNATURE>;
using UniqueOutputs = ckt::UniqueOutputs<SIGNATURE>;

static_assert(ckt::ValidUniqueInputs<SIGNATURE>);
static_assert(ckt::ValidUniqueOutputs<SIGNATURE>);
static_assert(ckt::TensorReflectable<Inputs, SIGNATURE>);
static_assert(ckt::TensorReflectable<Outputs, SIGNATURE>);

TEST(ConvFwdTesting, MakeDescriptors)
{
    const auto get_lengths = [](const auto& descriptor) {
        const auto lengths = descriptor.get_lengths();
        // Google Test cannot print std::span, so turn it into a vector for
        // legibility.
        return std::vector(lengths.begin(), lengths.end());
    };

    EXPECT_THAT(get_lengths(ARGS.make_input_descriptor()), ElementsAreArray({5, 17, 13, 125, 99}));
    EXPECT_THAT(get_lengths(ARGS.make_weight_descriptor()), ElementsAreArray({5, 44, 13, 4, 9}));
    EXPECT_THAT(get_lengths(ARGS.make_output_descriptor()), ElementsAreArray({5, 17, 44, 122, 91}));
}

TEST(ConvFwdTesting, Alloc)
{
    auto inputs  = alloc_inputs(ARGS);
    auto outputs = alloc_outputs(ARGS);

    EXPECT_THAT(inputs.get().input, NotNull());
    EXPECT_THAT(inputs.get().weight, NotNull());
    EXPECT_THAT(outputs.get().output, NotNull());
}

TEST(ConvFwdTesting, Validate)
{
    auto a = alloc_outputs(ARGS);
    auto b = alloc_outputs(ARGS);

    // Positive test
    {
        ckt::Outputs<SIGNATURE>::reflect(
            ARGS,
            [&]([[maybe_unused]] std::string_view name,
                const auto& desc,
                void* ckt::Outputs<SIGNATURE>::*ptr) {
                ckt::clear_tensor_buffer(
                    desc, a.get().*ptr, ck::type_convert<ck::bhalf_t, float>(123));
                ckt::clear_tensor_buffer(
                    desc, b.get().*ptr, ck::type_convert<ck::bhalf_t, float>(123));
            });

        const auto report = ckt::validate(ARGS, a.get(), b.get());
        EXPECT_THAT(report.get_errors().size(), Eq(0));
    }

    // Negative test
    {
        size_t field_count = 0;
        ckt::Outputs<SIGNATURE>::reflect(
            ARGS,
            [&]([[maybe_unused]] std::string_view name,
                const auto& desc,
                void* ckt::Outputs<SIGNATURE>::*ptr) {
                ++field_count;
                ckt::clear_tensor_buffer(
                    desc, a.get().*ptr, ck::type_convert<ck::bhalf_t, float>(2));
                ckt::clear_tensor_buffer(
                    desc, b.get().*ptr, ck::type_convert<ck::bhalf_t, float>(1));
            });

        const auto report = ckt::validate(ARGS, a.get(), b.get());
        EXPECT_THAT(report.get_errors().size(), Eq(field_count));
    }
}
