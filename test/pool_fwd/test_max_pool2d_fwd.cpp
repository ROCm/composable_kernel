// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      false>(true,
                                                             2,
                                                             false,
                                                             false,
                                                             param.length_,
                                                             param.window_spatial_lengths_,
                                                             param.window_strides_,
                                                             param.input_left_pads_,
                                                             param.input_right_pads_);
            EXPECT_TRUE(success);

            // max pool + index
            success = ck::profiler::profile_pool2d_fwd_impl<InDataType,
                                                            OutDataType,
                                                            ComputeDataType,
                                                            IndexDataType,
                                                            ck::ReduceTensorOp::MAX,
                                                            false,
                                                            true>(true,
                                                                  2,
                                                                  false,
                                                                  false,
                                                                  param.length_,
                                                                  param.window_spatial_lengths_,
                                                                  param.window_strides_,
                                                                  param.input_left_pads_,
                                                                  param.input_right_pads_);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes =
    ::testing::Types<std::tuple<F16, F16, F16, I32>, std::tuple<F32, F32, F32, I32>>;

TYPED_TEST_SUITE(TestMaxPool2dFwd, KernelTypes);
TYPED_TEST(TestMaxPool2dFwd, Test_Pool)
{
    // length, window_length, window_stride, left_pad, right_pad
    this->params = {{{1, 1, 1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
                    {{2, 16, 64, 64}, {64, 64}, {1, 1}, {0, 0}, {0, 0}},
                    {{2, 32, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {1, 1}}};

    this->Run();
}
