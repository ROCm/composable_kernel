// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_pool2d_fwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

template <typename Tuple>
class TestAvgPool2dFwd : public ::testing::Test
{
    protected:
    using InDataType      = std::tuple_element_t<0, Tuple>;
    using OutDataType     = std::tuple_element_t<1, Tuple>;
    using ComputeDataType = std::tuple_element_t<2, Tuple>;
    using IndexDataType   = std::tuple_element_t<3, Tuple>;

    std::vector<PoolingParam> params;

    void Run()
    {
        for(auto param : params)
        {
            // avg pool
            bool success =
                ck::profiler::profile_pool2d_fwd_impl<InDataType,
                                                      OutDataType,
                                                      ComputeDataType,
                                                      IndexDataType,
                                                      ck::tensor_layout::convolution::NHWC,
                                                      ck::tensor_layout::convolution::NHWC,
                                                      ck::ReduceTensorOp::AVG,
                                                      false,
                                                      false>(true,
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

using KernelTypes = std::conditional_t<
    CK_ENABLE_FP16 && CK_ENABLE_BF16 && CK_ENABLE_INT8 && CK_ENABLE_FP8,
    ::testing::Types<std::tuple<F8, F8, F32, I32>,
                     std::tuple<F8, F8, F32, I32>,
                     std::tuple<I8, I8, F32, I32>,
                     std::tuple<I8, I8, F32, I32>,
                     std::tuple<F16, F16, F32, I32>,
                     std::tuple<F16, F16, F32, I32>,
                     std::tuple<BF16, BF16, F32, I32>,
                     std::tuple<BF16, BF16, F32, I32>,
                     std::tuple<F32, F32, F32, I32>,
                     std::tuple<F32, F32, F32, I32>>,
    ::testing::Types<std::tuple<F32, F32, F32, I32>, std::tuple<F32, F32, F32, I32>>>;

TYPED_TEST_SUITE(TestAvgPool2dFwd, KernelTypes);
TYPED_TEST(TestAvgPool2dFwd, Test_Pool)
{
    // length, window_length, window_stride, window_dilation, left_pad, right_pad
    this->params = {{{1, 1, 1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                    {{2, 16, 64, 64}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                    {{2, 16, 64, 64}, {4, 4}, {4, 4}, {2, 2}, {0, 0}, {0, 0}},
                    {{2, 32, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1}}};

    this->Run();
}
