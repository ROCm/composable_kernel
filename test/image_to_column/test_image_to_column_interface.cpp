// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_image_to_column_impl.hpp"

#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

#include <gtest/gtest.h>

using DataType = float;
using InLayout = ck::tensor_layout::convolution::GNWC;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

template <ck::index_t ScalarPerVector, bool IsCPacked>
class TestImageToColumnInterface : public ::testing::Test
{
    protected:
    static constexpr ck::index_t NDimSpatial = 1;

    // clang-format off
    using DeviceImgToColInstance = ck::tensor_operation::device::DeviceImageToColumnImpl
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|    Thread|         Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|   Cluster|            Per|
        //#####################|    Spatial|         |           |            |      |      |      |   Lengths|         Vector|
        //#####################|           |         |           |            |      |      |      |          |               |
                              < NDimSpatial, InLayout,   DataType,    DataType,   256,   128,   128, S<16, 16>,ScalarPerVector>;
    // clang-format on

    ck::utils::conv::ConvParam conv_param;

    bool Run()
    {

        const auto N = conv_param.N_;
        const auto C = conv_param.C_;
        const auto FakeC =
            conv_param.C_ / 2; // Fake C to simulate the behavior that C is not packed

        const ck::index_t NDoHoWo =
            N *
            ck::accumulate_n<ck::index_t>(
                conv_param.output_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());
        const ck::index_t CZYX =
            C *
            ck::accumulate_n<ck::index_t>(
                conv_param.filter_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());

        const auto in_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);
        const auto out_desc = HostTensorDescriptor({NDoHoWo, CZYX});

        std::array<ck::index_t, NDimSpatial> input_spatial_lengths{};
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths{};
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> input_g_n_c_wis_strides{};
        std::array<ck::index_t, 2> output_m_k_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto copy = [](const auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };

        copy(conv_param.input_spatial_lengths_, input_spatial_lengths);
        copy(conv_param.filter_spatial_lengths_, filter_spatial_lengths);
        copy(conv_param.output_spatial_lengths_, output_spatial_lengths);
        copy(in_desc.GetStrides(), input_g_n_c_wis_strides);
        copy(out_desc.GetStrides(), output_m_k_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        auto img2col  = DeviceImgToColInstance{};
        auto argument = img2col.MakeArgument(nullptr,
                                             nullptr,
                                             N,
                                             IsCPacked ? C : FakeC,
                                             input_spatial_lengths,
                                             filter_spatial_lengths,
                                             output_spatial_lengths,
                                             input_g_n_c_wis_strides,
                                             output_m_k_strides,
                                             conv_filter_strides,
                                             conv_filter_dilations,
                                             input_left_pads,
                                             input_right_pads);

        return img2col.IsSupportedArgument(argument);
    }
};

class TestImageToColumnInterface1ScalarPerVector : public TestImageToColumnInterface<1, true>
{
};

class TestImageToColumnInterface4ScalarPerVector : public TestImageToColumnInterface<4, true>
{
};

class TestImageToColumnInterface4ScalarPerVectorFakeC : public TestImageToColumnInterface<4, false>
{
};

TEST_F(TestImageToColumnInterface1ScalarPerVector, X1ScalarPerVector)
{
    // vector load C * X % ScalarPerVector
    this->conv_param  = {1, 1, 1, 1, 1, {3}, {3}, {1}, {1}, {0}, {0}};
    bool is_supported = this->Run();
    EXPECT_TRUE(is_supported);
    // vector load C * left_pad_x % ScalarPerVector
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {1}, {3}, {0}};
    is_supported     = this->Run();
    EXPECT_TRUE(is_supported);
    // vector load C * right_pad_x % ScalarPerVector
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {1}, {0}, {3}};
    is_supported     = this->Run();
    EXPECT_TRUE(is_supported);

    // vector load C % ScalarPerVector, right_pad and stride
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {2}, {1}, {0}, {3}};
    is_supported     = this->Run();
    EXPECT_TRUE(is_supported);
    // vector load C % ScalarPerVector, left_pad and stride
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {2}, {1}, {3}, {0}};
    is_supported     = this->Run();
    EXPECT_TRUE(is_supported);
    // vector load C % ScalarPerVector, dilation
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {2}, {0}, {0}};
    is_supported     = this->Run();
    EXPECT_TRUE(is_supported);

    // C = 4
    this->conv_param = {1, 1, 1, 1, 4, {3}, {3}, {1}, {1}, {3}, {3}};
    is_supported     = this->Run();
    EXPECT_TRUE(is_supported);
}

TEST_F(TestImageToColumnInterface4ScalarPerVector, X4ScalarPerVector)
{
    // vector load C * X % ScalarPerVector
    this->conv_param  = {1, 1, 1, 1, 1, {3}, {3}, {1}, {1}, {0}, {0}};
    bool is_supported = this->Run();
    EXPECT_FALSE(is_supported);
    // vector load C * left_pad_x % ScalarPerVector
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {1}, {3}, {0}};
    is_supported     = this->Run();
    EXPECT_FALSE(is_supported);
    // vector load C * right_pad_x % ScalarPerVector
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {1}, {0}, {3}};
    is_supported     = this->Run();
    EXPECT_FALSE(is_supported);

    // vector load C % ScalarPerVector, right_pad and stride
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {2}, {1}, {0}, {3}};
    is_supported     = this->Run();
    EXPECT_FALSE(is_supported);
    // vector load C % ScalarPerVector, left_pad and stride
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {2}, {1}, {3}, {0}};
    is_supported     = this->Run();
    EXPECT_FALSE(is_supported);
    // vector load C % ScalarPerVector, dilation
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {2}, {0}, {0}};
    is_supported     = this->Run();
    EXPECT_FALSE(is_supported);

    // C = 4
    this->conv_param = {1, 1, 1, 1, 4, {3}, {3}, {1}, {1}, {3}, {3}};
    is_supported     = this->Run();
    EXPECT_TRUE(is_supported);
}

TEST_F(TestImageToColumnInterface4ScalarPerVectorFakeC, X4ScalarPerVectorFakeC)
{
    // C = 3
    this->conv_param  = {1, 1, 1, 1, 3, {4}, {3}, {1}, {1}, {0}, {0}};
    bool is_supported = this->Run();
    EXPECT_FALSE(is_supported);
    // C = 4
    this->conv_param = {1, 1, 1, 1, 8, {4}, {3}, {1}, {1}, {0}, {0}};
    is_supported     = this->Run();
    EXPECT_TRUE(is_supported);
}
