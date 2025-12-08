// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include "gpu_reference_utils.hpp"

using namespace ck;
using ck::test::ConvKernelType;
using ck::test::ConvParams;

TEST(GpuReferenceConvBwdWeight, Conv2DFP16Small)
{
    auto params = test::conv_test_shapes::get_2d_small();
    bool result =
        test::test_conv_gpu_ref<2, half_t, half_t, half_t>(params, ConvKernelType::BackwardWeight);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdWeight, Conv2DFP32Medium)
{
    auto params = test::conv_test_shapes::get_2d_medium();
    bool result =
        test::test_conv_gpu_ref<2, float, float, float>(params, ConvKernelType::BackwardWeight);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdWeight, Conv1DFP16)
{
    auto params = test::conv_test_shapes::get_1d();
    bool result =
        test::test_conv_gpu_ref<1, half_t, half_t, half_t>(params, ConvKernelType::BackwardWeight);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdWeight, Conv3DFP16Small)
{
    auto params = test::conv_test_shapes::get_3d_small();
    bool result =
        test::test_conv_gpu_ref<3, half_t, half_t, half_t>(params, ConvKernelType::BackwardWeight);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdWeight, Conv2DFP16Stride2)
{
    auto params = test::conv_test_shapes::get_2d_stride2();
    bool result =
        test::test_conv_gpu_ref<2, half_t, half_t, half_t>(params, ConvKernelType::BackwardWeight);
    EXPECT_TRUE(result);
}
