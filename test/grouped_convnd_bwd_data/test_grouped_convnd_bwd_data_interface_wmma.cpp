// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle.hpp"

#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

#include <gtest/gtest.h>

using DataType    = ck::half_t;
using AccDataType = float;
using Pass        = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using ConvBackwardDataSpecialization =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization;

static constexpr auto ConvBwdDataDefault   = ConvBackwardDataSpecialization::Default;
static constexpr auto Filter1x1Stride1Pad0 = ConvBackwardDataSpecialization::Filter1x1Stride1Pad0;

template <typename Tuple, ConvBackwardDataSpecialization ConvSpec>
class TestGroupedConvndBwdData : public ::testing::Test
{
    protected:
    static constexpr ck::index_t NDimSpatial = 2;

    using OutLayout = std::tuple_element_t<0, Tuple>;
    using WeiLayout = std::tuple_element_t<1, Tuple>;
    using InLayout  = std::tuple_element_t<2, Tuple>;

    // clang-format off
    using GroupedConvBwdDataDeviceInstance = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle
            //|    NumDim|        A|         B|          Ds|       E|        AData|        BData|    AccData|          CShuffle|     DsData|       EData|           A|           B|          CDE|       ConvForward| Block|  MPer|  NPer| K0Per| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
            //|   Spatial|   Layout|    Layout|      Layout|  Layout|         Type|         Type|       Type|          DataType|       Type|        Type| Elementwise| Elementwise|  Elementwise|    Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
            //|          |         |          |            |        |             |             |           |                  |           |            |   Operation|   Operation|    Operation|                  |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
            //|          |         |          |            |        |             |             |           |                  |           |            |            |            |             |                  |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
            < NDimSpatial,OutLayout, WeiLayout, ck::Tuple<>, InLayout,       DataType,  DataType, AccDataType,          DataType,  ck::Tuple<>,   DataType,        Pass,        Pass,        Pass,         ConvSpec, 64,    32,    64,     8,  8,    16,   16,       1,       4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,      S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>;
    // clang-format on

    ck::utils::conv::ConvParam conv_param;

    template <ck::index_t NDimSpatial>
    bool Run()
    {

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        std::array<ck::index_t, NDimSpatial + 3> out_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> out_strides{};
        std::array<ck::index_t, NDimSpatial + 3> wei_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> wei_strides{};
        std::array<ck::index_t, NDimSpatial + 3> in_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> in_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(out_g_n_k_wos_desc.GetLengths(), out_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), out_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), wei_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), wei_strides);
        copy(in_g_n_c_wis_desc.GetLengths(), in_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), in_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        auto conv = GroupedConvBwdDataDeviceInstance{};

        auto argument = conv.MakeArgument(nullptr,
                                          nullptr,
                                          std::array<const void*, 0>{},
                                          nullptr,
                                          out_lengths,
                                          out_strides,
                                          wei_lengths,
                                          wei_strides,
                                          {},
                                          {},
                                          in_lengths,
                                          in_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          Pass{},
                                          Pass{},
                                          Pass{});
        return conv.IsSupportedArgument(argument);
    }
};

using GNHWC = ck::tensor_layout::convolution::GNHWC;
using NHWGC = ck::tensor_layout::convolution::NHWGC;

using GKYXC = ck::tensor_layout::convolution::GKYXC;

using GNHWK = ck::tensor_layout::convolution::GNHWK;
using NHWGK = ck::tensor_layout::convolution::NHWGK;

using KernelTypes =
    ::testing::Types<std::tuple<GNHWK, GKYXC, GNHWC>, std::tuple<NHWGK, GKYXC, NHWGC>>;

template <typename Tuple>
class TestGroupedConvndBwdDataDefault : public TestGroupedConvndBwdData<Tuple, ConvBwdDataDefault>
{
};

template <typename Tuple>
class TestGroupedConvndBwdDataFilter1x1
    : public TestGroupedConvndBwdData<Tuple, Filter1x1Stride1Pad0>
{
};

TYPED_TEST_SUITE(TestGroupedConvndBwdDataDefault, KernelTypes);
TYPED_TEST_SUITE(TestGroupedConvndBwdDataFilter1x1, KernelTypes);

TYPED_TEST(TestGroupedConvndBwdDataFilter1x1, SpecializationCheck)
{
    // Check filter 3,3 instead of 1,1
    this->conv_param  = {2, 2, 4, 192, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    bool is_supported = this->template Run<2>();
    EXPECT_FALSE(is_supported);

    // Check strides 2,2 instead of 1,1
    this->conv_param = {2, 2, 4, 192, 192, {1, 1}, {28, 28}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    is_supported     = this->template Run<2>();
    EXPECT_FALSE(is_supported);

    // Check with pad
    this->conv_param = {2, 2, 4, 192, 192, {1, 1}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    is_supported     = this->template Run<2>();
    EXPECT_FALSE(is_supported);

    // Supported version
    this->conv_param = {2, 2, 4, 192, 192, {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    is_supported     = this->template Run<2>();
    EXPECT_TRUE(is_supported);
}

TYPED_TEST(TestGroupedConvndBwdDataDefault, VectorLoadCheck)
{
    // vector load for A
    this->conv_param  = {2, 2, 128, 129, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    bool is_supported = this->template Run<2>();
    EXPECT_FALSE(is_supported);
    // vector load for B, E, Ds
    this->conv_param = {2, 2, 128, 128, 257, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    is_supported     = this->template Run<2>();
    EXPECT_FALSE(is_supported);
}
