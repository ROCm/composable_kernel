#include <half.hpp>
#include <iostream>
#include <tuple>
#include <vector>
#include "gtest/gtest.h"

#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "ck/library/utility/conv_util.hpp"
#include "conv_util.hpp"

namespace {

template <typename T>
bool test_conv2d_nhwc_instances(const std::vector<test::conv::DeviceConvFwdNoOpPtr>& conv_ptrs)
{
    using namespace std::placeholders;
    using namespace ck::utils;

    conv::ConvParams params;
    params.num_dim_spatial_        = 2;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3, 3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{71, 71};
    params.conv_filter_strides_    = std::vector<ck::index_t>{2, 2};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{1, 1};
    params.input_right_pads_       = std::vector<ck::index_t>{1, 1};

    conv::ConvFwdOpInstance<T, T, T> conv_instance(params);

    auto reference_conv_fwd_fun =
        std::bind(conv::run_reference_convolution_forward<2, T, T, T>, params, _1, _2, _3);
    OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    return run_engine.Test(conv_ptrs);
}

} // anonymous namespace

TEST(Conv2DFwdNHWC, TestConv2D)
{
    using namespace std::placeholders;
    using namespace ck::utils;

    ck::utils::conv::ConvParams params;
    params.N_                     = 2;
    params.K_                     = 16;
    params.C_                     = 4;
    params.input_spatial_lengths_ = std::vector<ck::index_t>{16, 16};
    params.conv_filter_strides_   = std::vector<ck::index_t>{1, 1};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<2>(conv_ptrs);
    conv::ConvFwdOpInstance<float, float, float> conv_instance(params);

    auto reference_conv_fwd_fun = std::bind(
        conv::run_reference_convolution_forward<2, float, float, float>, params, _1, _2, _3);
    OpInstanceRunEngine<float, float, float> run_engine(conv_instance, reference_conv_fwd_fun);
    run_engine.SetAtol(1e-5);
    run_engine.SetRtol(1e-4);
    EXPECT_TRUE(run_engine.Test(conv_ptrs));
}

TEST(Conv2DFwdNHWC, Bf16Instances)
{
    EXPECT_TRUE(test_conv2d_nhwc_instances<ck::bhalf_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>::Get<2>()));
}

TEST(Conv2DFwdNHWC, F16Instances)
{
    EXPECT_TRUE(test_conv2d_nhwc_instances<ck::half_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::half_t, ck::half_t, ck::half_t>::Get<2>()));
}

TEST(Conv2DFwdNHWC, BF32Instances)
{
    EXPECT_TRUE(test_conv2d_nhwc_instances<float>(
        ck::utils::conv::ConvolutionFwdInstances<float, float, float>::Get<2>()));
}

TEST(Conv2DFwdNHWC, F32Instances)
{
    EXPECT_TRUE(test_conv2d_nhwc_instances<float>(
        ck::utils::conv::ConvolutionFwdInstances<float, float, float>::Get<2>()));
}

TEST(Conv2DFwdNHWC, Int8Instances)
{
    EXPECT_TRUE(test_conv2d_nhwc_instances<int8_t>(
        ck::utils::conv::ConvolutionFwdInstances<int8_t, int8_t, int8_t>::Get<2>()));
}
