// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ::testing::ElementsAreArray;
using ::testing::Ge;

TEST(TensorDescriptor, Basic)
{
    constexpr auto dt           = ckb::DataType::FP16;
    std::vector<size_t> lengths = {123, 456, 789};
    std::vector<size_t> strides = {456 * 789, 789, 1};

    ckt::TensorDescriptor<dt> descriptor(lengths, strides);

    EXPECT_THAT(descriptor.get_lengths(), ElementsAreArray(lengths));
    EXPECT_THAT(descriptor.get_strides(), ElementsAreArray(strides));
}

TEST(TensorDescriptor, ComputeSize)
{
    constexpr auto dt           = ckb::DataType::FP32;
    std::vector<size_t> lengths = {305, 130, 924};
    std::vector<size_t> strides = {1000 * 1000, 1, 1000};

    ckt::TensorDescriptor<dt> descriptor(lengths, strides);

    // Compute the location of the last item in memory, then add one
    // to get the minimum size.
    size_t expected_size = 1;
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        expected_size += (lengths[i] - 1) * strides[i];
    }

    EXPECT_THAT(descriptor.get_element_space_size(), Ge(expected_size));
    EXPECT_THAT(descriptor.get_element_space_size_in_bytes(),
                Ge(expected_size * ckt::data_type_sizeof(dt)));
}
