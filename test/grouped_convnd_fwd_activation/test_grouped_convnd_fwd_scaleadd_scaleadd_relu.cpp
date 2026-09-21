// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "profiler/profile_grouped_conv_fwd_scaleadd_scaleadd_relu_impl.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using I8   = int8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

template <typename Tuple>
class TestGroupedConvndFwdScaleAddScaleAddRelu : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, Tuple>;
    using WeiDataType = std::tuple_element_t<1, Tuple>;
    using OutDataType = std::tuple_element_t<2, Tuple>;
    using InLayout    = std::tuple_element_t<3, Tuple>;
    using WeiLayout   = std::tuple_element_t<4, Tuple>;
    using OutLayout   = std::tuple_element_t<5, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(auto& param : conv_params)
        {
            if(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a")
            {
                if(std::is_same<InDataType, ck::f8_t>::value ||
                   std::is_same<InDataType, ck::bf8_t>::value)
                {
                    printf("Skipping FP8 / BF8 tests on CDNA1/2.\n");
                    continue;
                }
            }
            pass = pass && ck::profiler::profile_grouped_conv_fwd_scaleadd_scaleadd_relu_impl<
                               NDimSpatial,
                               InLayout,
                               WeiLayout,
                               OutLayout,
                               InDataType,
                               WeiDataType,
                               OutDataType,
                               ck::tensor_operation::element_wise::ScaleAddScaleAddRelu,
                               InDataType,
                               InDataType>(true,  // do_verification
                                           1,     // init_method: integer value
                                           false, // do_log
                                           true,  // time_kernel
                                           param);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;
using CombConvScaleAddScaleAddReluKernelTypes3d =
    ::testing::Types<std::tuple<F16, F16, F16, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<BF16, BF16, BF16, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<I8, I8, I8, NDHWGC, GKZYXC, NDHWGK>>;

template <typename Tuple>
class TestGroupedConvndFwdScaleAddScaleAddRelu3d
    : public TestGroupedConvndFwdScaleAddScaleAddRelu<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwdScaleAddScaleAddRelu3d,
                 CombConvScaleAddScaleAddReluKernelTypes3d);

TYPED_TEST(TestGroupedConvndFwdScaleAddScaleAddRelu3d, Test3D)
{
    this->conv_params.clear();

    this->conv_params.push_back(
        {3, 3, 5, 96, 200, {1, 1, 1}, {37, 37, 16}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {5, 5, 5}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    this->template Run<3>();
}
