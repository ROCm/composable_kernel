// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/impl/device_pool2d_fwd_nhwc_nhwc.hpp"
#include "profiler/profile_pool2d_fwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

// Padding expands a 2x2 input to a 3x3 output. All nine output elements must
// fit in the descriptor's buffer bound, even though the input has only four.
TEST(AvgPool2dFwdDescriptor, OutputLargerThanInput)
{
    using DevicePool =
        ck::tensor_operation::device::DevicePool2dFwd_NHWC_NHWC<float,
                                                                float,
                                                                int32_t,
                                                                float,
                                                                ck::ReduceTensorOp::AVG,
                                                                false,
                                                                256,
                                                                256,
                                                                1,
                                                                1,
                                                                1,
                                                                1>;
    const auto descriptors =
        DevicePool::MakeABGridDescriptor_A_M_K_B_M({1, 1, 2, 2}, // input NCHW lengths
                                                   {1, 1, 3, 3}, // output NCHW lengths
                                                   {4, 1, 2, 1}, // input strides for NHWC storage
                                                   {9, 1, 3, 1}, // output strides for NHWC storage
                                                   {2, 2},       // window
                                                   {1, 1},       // stride
                                                   {1, 1},       // dilation
                                                   {1, 1},       // left padding
                                                   {1, 1});      // right padding

    EXPECT_EQ(descriptors[ck::Number<0>{}].GetElementSpaceSize(), 4);
    EXPECT_EQ(descriptors[ck::Number<1>{}].GetElementSpaceSize(), 9);
}

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

template <typename Tuple>
class TestAvgPool2dFwd : public ::testing::Test
{
    protected:
    using InDataType      = std::tuple_element_t<0, Tuple>;
    using OutDataType     = std::tuple_element_t<1, Tuple>;
    using ComputeDataType = std::tuple_element_t<2, Tuple>;
    using IndexDataType   = std::tuple_element_t<3, Tuple>;

    static std::vector<PoolingParam> params;

    void Run()
    {
        for(size_t i = 0; i < this->params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = this->params[i];
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
                                                             param.input_right_pads_,
                                                             instance_index);
            EXPECT_TRUE(success);
        }
    }
};

template <typename T>
std::vector<PoolingParam> TestAvgPool2dFwd<T>::params = {
    {{{1, 1, 1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
     {{2, 16, 64, 64}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
     {{2, 16, 64, 64}, {4, 4}, {4, 4}, {2, 2}, {0, 0}, {0, 0}},
     {{2, 32, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1}}}};

using AvgPool2D_F32_Types =
    ::testing::Types<std::tuple<F32, F32, F32, I32>, std::tuple<F32, F32, F32, I32>>;
using AvgPool2D_F16_Types =
    ::testing::Types<std::tuple<F16, F16, F32, I32>, std::tuple<F16, F16, F32, I32>>;
using AvgPool2D_BF16_Types =
    ::testing::Types<std::tuple<I8, I8, F32, I32>, std::tuple<BF16, BF16, F32, I32>>;
using AvgPool2D_I8_Types =
    ::testing::Types<std::tuple<I8, I8, F32, I32>, std::tuple<I8, I8, F32, I32>>;
using AvgPool2D_F8_Types =
    ::testing::Types<std::tuple<F8, F8, F32, I32>, std::tuple<F8, F8, F32, I32>>;

template <typename TType>
class AvgPool2D_F32 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP32)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_F32 tests because CK_ENABLE_FP32 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_F16 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP16)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_F16 tests because CK_ENABLE_FP16 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_BF16 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_BF16)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_BF16 tests because CK_ENABLE_BF16 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_I8 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_INT8)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_I8 tests because CK_ENABLE_INT8 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_F8 : public TestAvgPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP8)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_F8 tests because CK_ENABLE_FP8 is "
                            "not enabled";
        }
    }
};

TYPED_TEST_SUITE(AvgPool2D_F32, AvgPool2D_F32_Types);
TYPED_TEST_SUITE(AvgPool2D_F16, AvgPool2D_F16_Types);
TYPED_TEST_SUITE(AvgPool2D_BF16, AvgPool2D_BF16_Types);
TYPED_TEST_SUITE(AvgPool2D_I8, AvgPool2D_I8_Types);
TYPED_TEST_SUITE(AvgPool2D_F8, AvgPool2D_F8_Types);

TYPED_TEST(AvgPool2D_F32, AvgPool2D_F32_Test) { this->Run(); }
TYPED_TEST(AvgPool2D_F16, AvgPool2D_F16_Test) { this->Run(); }
TYPED_TEST(AvgPool2D_BF16, AvgPool2D_BF16_Test) { this->Run(); }
TYPED_TEST(AvgPool2D_I8, AvgPool2D_I8_Test) { this->Run(); }
TYPED_TEST(AvgPool2D_F8, AvgPool2D_F8_Test) { this->Run(); }
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
