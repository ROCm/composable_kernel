// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace {

using namespace ck_tile;

constexpr long_index_t kIndexMax = std::numeric_limits<index_t>::max();

// Only a positive size of 2^31 or more is clamped to the index_t range.
static_assert(!detail::needs_element_space_clamp(0));
static_assert(!detail::needs_element_space_clamp(1));
static_assert(!detail::needs_element_space_clamp(kIndexMax));
static_assert(detail::needs_element_space_clamp(kIndexMax + 1));
static_assert(detail::needs_element_space_clamp(long_index_t{1} << 32));
static_assert(detail::needs_element_space_clamp((long_index_t{1} << 32) + 5));
static_assert(detail::needs_element_space_clamp(std::numeric_limits<long_index_t>::max()));
// A zero-length dimension makes the size negative; it is kept as it is.
static_assert(!detail::needs_element_space_clamp(-1));
static_assert(!detail::needs_element_space_clamp(-(long_index_t{1} << 32)));
static_assert(!detail::needs_element_space_clamp(std::numeric_limits<long_index_t>::min()));

index_t element_space_size(index_t rows, index_t cols, index_t row_stride)
{
    return make_naive_tensor_descriptor(make_tuple(rows, cols), make_tuple(row_stride, 1))
        .get_element_space_size();
}

index_t element_space_size_with_offset(index_t rows, index_t cols, index_t row_stride)
{
    return make_naive_tensor_descriptor_with_offset(
               make_tuple(rows, cols), make_tuple(row_stride, 1), 0)
        .get_element_space_size();
}

TEST(ElementSpaceClamp, NaiveDescriptor)
{
    for(auto size : {element_space_size, element_space_size_with_offset})
    {
        EXPECT_EQ(size(128, 64, 256), 127 * 256 + 64);
        // (65536 - 1) * 32768 + 32768 = 2^31: one past the index_t range.
        EXPECT_EQ(size(65536, 32768, 32768), kIndexMax);
        // 2^31 - 1, the index_t maximum itself, is not clamped.
        EXPECT_EQ(size(65536, 32767, 32768), kIndexMax);
        // 2^32 and beyond, so the low 32 bits alone would look small.
        EXPECT_EQ(size(131072, 32768, 32768), kIndexMax);
        EXPECT_EQ(size(131073, 4, 32768), kIndexMax);
        // A zero-length row dimension: (0 - 1) * 256 + 4 = -252.
        EXPECT_EQ(size(0, 4, 256), -252);
    }
}

} // namespace
