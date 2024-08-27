// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_pool2d_fwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

template <typename Tuple>
class TestMaxPool2dFwd : public ::testing::Test
{
    protected:
    using InDataType      = std::tuple_element_t<0, Tuple>;
    using OutDataType     = std::tuple_element_t<1, Tuple>;
    using ComputeDataType = std::tuple_element_t<2, Tuple>;
    using IndexDataType   = std::tuple_element_t<3, Tuple>;
    static constexpr bool ReturnIndex = std::tuple_element_t<4, Tuple>::value;

    std::vector<PoolingParam> params;

    void Run()
    {
        for(auto param : params)
        {
            // max pool
            bool success =
                ck::profiler::profile_pool2d_fwd_impl<InDataType,
                                                      OutDataType,
                                                      ComputeDataType,
                                                      IndexDataType,
                                                      ck::tensor_layout::convolution::NHWC,
                                                      ck::tensor_layout::convolution::NHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      ReturnIndex>(true,
                                                             2,
                                                             false,
                                                             false,
                                                             param.length_,
                                                             param.window_spatial_lengths_,
                                                             param.window_strides_,
                                                             param.window_dilations_,
                                                             param.input_left_pads_,
                                                             param.input_right_pads_);
            EXPECT_TRUE(success);
        }
    }
};

using true_t  = std::integral_constant<bool, true>;
using false_t = std::integral_constant<bool, false>;

using KernelTypes = std::conditional_t<
    CK_ENABLE_FP16,
    ::testing::Types<std::tuple<F16, F16, F32, I32, true_t>,
                     std::tuple<F16, F16, F32, I32, false_t>,
                     std::tuple<F32, F32, F32, I32, true_t>,
                     std::tuple<F32, F32, F32, I32, false_t>>,
    ::testing::Types<std::tuple<F32, F32, F32, I32, true_t>,
                     std::tuple<F32, F32, F32, I32, false_t>>
>;

TYPED_TEST_SUITE(TestMaxPool2dFwd, KernelTypes);
TYPED_TEST(TestMaxPool2dFwd, Test_Pool)
{
    // length, window_length, window_stride, window_dilation, left_pad, right_pad
    this->params = {{{1, 1, 1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                    {{2, 16, 64, 64}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                    {{2, 16, 64, 64}, {4, 4}, {4, 4}, {2, 2}, {0, 0}, {0, 0}},
                    {{2, 32, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1}}};

    this->Run();
}
