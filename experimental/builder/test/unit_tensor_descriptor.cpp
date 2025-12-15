// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ::testing::ElementsAreArray;
using ::testing::Eq;

TEST(TensorDescriptor, Basic)
{
    constexpr auto dt                = ckb::DataType::FP16;
    constexpr size_t rank            = 3;
    std::array<size_t, rank> lengths = {123, 456, 789};
    std::array<size_t, rank> strides = {456 * 789, 789, 1};

    ckt::TensorDescriptor<dt, rank> descriptor(lengths, strides);

    EXPECT_THAT(descriptor.get_lengths(), ElementsAreArray(lengths));
    EXPECT_THAT(descriptor.get_strides(), ElementsAreArray(strides));
}

TEST(TensorDescriptor, ComputeSize)
{
    constexpr auto dt             = ckb::DataType::FP32;
    constexpr size_t rank         = 3;
    std::array<size_t, 3> lengths = {305, 130, 924};
    std::array<size_t, 3> strides = {1000 * 1000, 1, 1000};

    ckt::TensorDescriptor<dt, rank> descriptor(lengths, strides);

    // Compute the location of the last item in memory, then add one
    // to get the minimum size.
    size_t expected_size  = 1;
    size_t expected_numel = 1;
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        expected_size += (lengths[i] - 1) * strides[i];
        expected_numel *= lengths[i];
    }

    EXPECT_THAT(descriptor.get_element_size(), Eq(expected_numel));
    EXPECT_THAT(descriptor.get_element_space_size(), Eq(expected_size));
    EXPECT_THAT(descriptor.get_element_space_size_in_bytes(),
                Eq(expected_size * ckt::data_type_sizeof(dt)));
}
