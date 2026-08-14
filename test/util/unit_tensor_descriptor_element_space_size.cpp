// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression tests for GetElementSpaceSize() of a naive tensor descriptor.
//
// A dynamic (runtime) descriptor whose (length-1)*stride product exceeds
// INT32_MAX must compute the element space size in 64-bit. If the multiply is
// performed in index_t (int32) before widening, the product wraps and the
// reported element space size is far too small, which later leads to an
// undersized workspace allocation and an out-of-bounds device write
// (e.g. grouped conv bwd-weight with K*C > INT32_MAX).
//
// A fully static (Number<>) descriptor must keep a compile-time-constant
// element space size so TensorDescriptor::IsKnownAtCompileTime() stays true.

#include <limits>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

using ck::index_t;
using ck::long_index_t;
using ck::make_naive_tensor_descriptor;
using ck::make_tuple;
using ck::Number;

__global__ void
element_space_size_kernel(long_index_t* out, index_t length0, index_t length1, index_t stride0)
{
    const auto desc =
        make_naive_tensor_descriptor(make_tuple(length0, length1), make_tuple(stride0, index_t{1}));
    out[0] = desc.GetElementSpaceSize();
}

static long_index_t device_element_space_size(index_t length0, index_t length1, index_t stride0)
{
    long_index_t* d_out = nullptr;
    HIP_CHECK_ERROR(hipMalloc(&d_out, sizeof(long_index_t)));
    element_space_size_kernel<<<dim3(1), dim3(1), 0, nullptr>>>(d_out, length0, length1, stride0);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    long_index_t h_out = 0;
    HIP_CHECK_ERROR(hipMemcpy(&h_out, d_out, sizeof(long_index_t), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipFree(d_out));
    return h_out;
}

// Dynamic descriptor with (length0-1)*stride0 > INT32_MAX must not overflow,
// on both host and device.
TEST(TensorDescriptorElementSpaceSize, DynamicDoesNotOverflowInt32)
{
    // length {65537, 65537}, stride {65537, 1}
    // element_space_size = 1 + (65537-1)*65537 + (65537-1)*1 = 4,295,098,369
    const index_t length0 = 65537;
    const index_t length1 = 65537;
    const index_t stride0 = 65537;

    const long_index_t expected = 1 + static_cast<long_index_t>(length0 - 1) * stride0 +
                                  static_cast<long_index_t>(length1 - 1);

    ASSERT_GT(expected, static_cast<long_index_t>(std::numeric_limits<int32_t>::max()))
        << "test case must actually exceed INT32_MAX to exercise the overflow path";

    const auto desc =
        make_naive_tensor_descriptor(make_tuple(length0, length1), make_tuple(stride0, index_t{1}));
    EXPECT_EQ(desc.GetElementSpaceSize(), expected) << "host element space size overflowed";
    EXPECT_EQ(device_element_space_size(length0, length1, stride0), expected)
        << "device element space size overflowed";
}

// Products that fit in int32 must be unchanged by the widening fix.
TEST(TensorDescriptorElementSpaceSize, BoundaryBelowInt32IsUnchanged)
{
    // 46340 = floor(sqrt(INT32_MAX)); 46340*46340 = 2,147,395,600 < INT32_MAX.
    const index_t length0 = 46341;
    const index_t length1 = 46340;
    const index_t stride0 = 46340;

    const long_index_t expected = 1 + static_cast<long_index_t>(length0 - 1) * stride0 +
                                  static_cast<long_index_t>(length1 - 1);

    const auto desc =
        make_naive_tensor_descriptor(make_tuple(length0, length1), make_tuple(stride0, index_t{1}));
    EXPECT_EQ(desc.GetElementSpaceSize(), expected);
}

// A fully static descriptor must remain known at compile time.
TEST(TensorDescriptorElementSpaceSize, StaticDescriptorStaysCompileTime)
{
    const auto desc = make_naive_tensor_descriptor(make_tuple(Number<8>{}, Number<260>{}),
                                                   make_tuple(Number<260>{}, Number<1>{}));
    static_assert(decltype(desc)::IsKnownAtCompileTime(),
                  "static tensor descriptor must keep a compile-time-constant element space size");
    EXPECT_TRUE(decltype(desc)::IsKnownAtCompileTime());
    EXPECT_EQ(desc.GetElementSpaceSize(), 1 + (8 - 1) * 260 + (260 - 1));
}
