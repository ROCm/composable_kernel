// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/testing/tensor_foreach.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <functional>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ::testing::Each;
using ::testing::Eq;

TEST(TensorForeach, CalculateOffset)
{
    EXPECT_THAT(ckt::calculate_offset(ckt::Extent{1, 2, 3}, ckt::Extent{100, 10, 1}), Eq(123));
    EXPECT_THAT(ckt::calculate_offset(ckt::Extent{523, 266, 263}, ckt::Extent{1, 545, 10532}),
                Eq(2915409));
    EXPECT_THAT(ckt::calculate_offset(ckt::Extent{}, ckt::Extent{}), Eq(0));
    // Note: >4 GB overflow test
    EXPECT_THAT(ckt::calculate_offset(ckt::Extent{8, 2, 5, 7, 0, 4, 1, 3, 6, 9},
                                      ckt::Extent{1'000,
                                                  1'000'000,
                                                  10'000'000,
                                                  1'000'000'000,
                                                  1,
                                                  10'000,
                                                  100,
                                                  10,
                                                  100'000'000,
                                                  100'000}),
                Eq(size_t{7'652'948'130}));
}

TEST(TensorForeach, VisitsCorrectCount)
{
    // tensor_foreach should visit every index exactly once.
    // This test checks that the count is at least correct.

    const ckt::Extent shape = {10, 20, 30};

    auto d_count = ckt::alloc_buffer(sizeof(uint64_t));
    ckt::check_hip(hipMemset(d_count.get(), 0, sizeof(uint64_t)));

    ckt::tensor_foreach(shape, [count = d_count.get()]([[maybe_unused]] const auto& index) {
        atomicAdd(reinterpret_cast<uint64_t*>(count), 1);
    });

    uint64_t actual;
    ckt::check_hip(hipMemcpy(&actual, d_count.get(), sizeof(uint64_t), hipMemcpyDeviceToHost));

    const auto expected = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    EXPECT_THAT(actual, Eq(expected));
}

TEST(TensorForeach, VisitsEveryIndex)
{
    const ckt::Extent shape = {5, 6, 7, 8, 9, 10, 11};
    const auto total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    // We know this is correct due to testing in unit_tensor_descriptor.cpp
    const auto stride = ckt::PackedRightLayout{}(shape);

    auto d_output = ckt::alloc_buffer(sizeof(uint32_t) * total);
    ckt::check_hip(hipMemset(d_output.get(), 0, sizeof(uint32_t) * total));

    ckt::tensor_foreach(shape, [output = d_output.get(), stride](const auto& index) {
        // We know this is correct due to the CalculateOffset test.
        auto offset = ckt::calculate_offset(index, stride);

        // Use atomic add so that we can check that every index is visited exactly once.
        atomicAdd(&reinterpret_cast<uint32_t*>(output)[offset], 1);
    });

    std::vector<uint32_t> actual(total);
    ckt::check_hip(
        hipMemcpy(actual.data(), d_output.get(), sizeof(uint32_t) * total, hipMemcpyDeviceToHost));

    EXPECT_THAT(actual, Each(Eq(1)));
}
