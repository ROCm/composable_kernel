// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include "gpu_reference_utils.hpp"

using namespace ck;
using ck::test::ConvKernelType;
using ck::test::ConvParams;

TEST(GpuReferenceConvFwd, Conv2DFP16Small)
{
    auto params = test::conv_test_shapes::get_2d_small();
    bool result =
        test::test_conv_gpu_ref<2, half_t, half_t, half_t>(params, ConvKernelType::Forward);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwd, Conv2DFP32Medium)
{
    auto params = test::conv_test_shapes::get_2d_medium();
    bool result = test::test_conv_gpu_ref<2, float, float, float>(params, ConvKernelType::Forward);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwd, Conv1DFP16)
{
    auto params = test::conv_test_shapes::get_1d();
    bool result =
        test::test_conv_gpu_ref<1, half_t, half_t, half_t>(params, ConvKernelType::Forward);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwd, Conv3DFP16Small)
{
    auto params = test::conv_test_shapes::get_3d_small();
    bool result =
        test::test_conv_gpu_ref<3, half_t, half_t, half_t>(params, ConvKernelType::Forward);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwd, Conv2DFP16Stride2)
{
    auto params = test::conv_test_shapes::get_2d_stride2();
    bool result =
        test::test_conv_gpu_ref<2, half_t, half_t, half_t>(params, ConvKernelType::Forward);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwd, Conv2DFP16GroupedG2)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result =
        test::test_conv_gpu_ref<2, half_t, half_t, half_t>(params, ConvKernelType::Forward);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwd, Conv2DFP32GroupedG4)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g4();
    bool result = test::test_conv_gpu_ref<2, float, float, float>(params, ConvKernelType::Forward);
    EXPECT_TRUE(result);
}

// Test case from profiler that's failing - GNHWC layout
// Params: {NDimSpatial=2, G=2, N=32, K=128, C=256, filter={1,1}, input={7,7}, output={4,4},
// stride={2,2}, dilation={1,1}, pad={0,0}}
TEST(GpuReferenceConvFwd, Conv2DFP32ProfilerGNHWC)
{
    using namespace ck::test;
    ConvParams<2> params;
    params.N              = 32;
    params.K              = 128;
    params.C              = 256;
    params.G              = 2;
    params.filter_spatial = {1, 1};
    params.input_spatial  = {7, 7};
    params.output_spatial = {4, 4};
    params.strides        = {2, 2};
    params.dilations      = {1, 1};
    params.pads           = {0, 0};

    bool result = test::test_conv_gpu_ref<2, float, float, float>(params, ConvKernelType::Forward);
    EXPECT_TRUE(result);
}
