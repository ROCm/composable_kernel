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
class Conv2dFwdNHWCInstances : public ::testing::Test
{
    public:
    bool test_conv2d_nhwc_instances(const std::vector<test::conv::DeviceConvFwdNoOpPtr>& conv_ptrs,
                                    const ck::utils::conv::ConvParams& params)
    {
        using namespace std::placeholders;
        using namespace ck::utils;

        conv::ConvFwdOpInstance<T, T, T> conv_instance(params);
        auto reference_conv_fwd_fun =
            std::bind(conv::run_reference_convolution_forward<2, T, T, T>, params, _1, _2, _3);
        OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
        return run_engine.Test(conv_ptrs);
    }

    bool test_default()
    {
        return test_conv2d_nhwc_instances(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<2>(), params_default_);
    }

    bool test_filter1x1_stride1_pad0()
    {
        return test_conv2d_nhwc_instances(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<2>(),
            params_filter1x1_stride1_pad0_);
    }

    bool test_filter1x1_pad0()
    {
        return test_conv2d_nhwc_instances(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<2>(),
            params_filter1x1_pad0_);
    }

    bool test_oddC()
    {
        return test_conv2d_nhwc_instances(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<2>(), params_oddC_);
    }

    static inline ck::utils::conv::ConvParams params_default_;
    static inline ck::utils::conv::ConvParams params_filter1x1_stride1_pad0_{
        2, 4, 256, 128, {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    static inline ck::utils::conv::ConvParams params_filter1x1_pad0_{
        2, 4, 256, 128, {1, 1}, {28, 28}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    static inline ck::utils::conv::ConvParams params_oddC_{
        2, 4, 256, 125, {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
};

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

using Conv2dFwdNHWCInstancesTypes = ::testing::Types<ck::bhalf_t, /*ck::half_t,*/ float, int8_t>;
TYPED_TEST_SUITE(Conv2dFwdNHWCInstances, Conv2dFwdNHWCInstancesTypes);

TYPED_TEST(Conv2dFwdNHWCInstances, conv_spec_default) { EXPECT_TRUE(this->test_default()); }

TYPED_TEST(Conv2dFwdNHWCInstances, conv_spec_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0());
}

TYPED_TEST(Conv2dFwdNHWCInstances, conv_spec_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0());
}

TYPED_TEST(Conv2dFwdNHWCInstances, conv_spec_oddC) { EXPECT_TRUE(this->test_oddC()); }

// Workaround for linker error:
// ld.lld: error: undefined symbol: _ZTIDF16_

namespace {

class Conv2dFwdNHWCInstancesF16 : public ::testing::Test
{

    using T = ck::half_t;

    public:
    bool test_conv2d_nhwc_instances(const std::vector<test::conv::DeviceConvFwdNoOpPtr>& conv_ptrs,
                                    const ck::utils::conv::ConvParams& params)
    {
        using namespace std::placeholders;
        using namespace ck::utils;

        conv::ConvFwdOpInstance<T, T, T> conv_instance(params);
        auto reference_conv_fwd_fun =
            std::bind(conv::run_reference_convolution_forward<2, T, T, T>, params, _1, _2, _3);
        OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
        return run_engine.Test(conv_ptrs);
    }

    bool test_default()
    {
        return test_conv2d_nhwc_instances(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<2>(), params_default_);
    }

    bool test_filter1x1_stride1_pad0()
    {
        return test_conv2d_nhwc_instances(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<2>(),
            params_filter1x1_stride1_pad0_);
    }

    bool test_filter1x1_pad0()
    {
        return test_conv2d_nhwc_instances(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<2>(),
            params_filter1x1_pad0_);
    }

    bool test_oddC()
    {
        return test_conv2d_nhwc_instances(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<2>(), params_oddC_);
    }

    protected:
    static inline ck::utils::conv::ConvParams params_default_;
    static inline ck::utils::conv::ConvParams params_filter1x1_stride1_pad0_{
        2, 4, 256, 128, {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
    static inline ck::utils::conv::ConvParams params_filter1x1_pad0_{
        2, 4, 256, 128, {1, 1}, {28, 28}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    static inline ck::utils::conv::ConvParams params_oddC_{
        2, 4, 256, 125, {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}};
};

} // namespace

TEST_F(Conv2dFwdNHWCInstancesF16, conv_spec_default) { EXPECT_TRUE(test_default()); }

TEST_F(Conv2dFwdNHWCInstancesF16, conv_spec_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(test_filter1x1_stride1_pad0());
}

TEST_F(Conv2dFwdNHWCInstancesF16, conv_spec_filter1x1_pad0) { EXPECT_TRUE(test_filter1x1_pad0()); }

TEST_F(Conv2dFwdNHWCInstancesF16, conv_spec_oddC) { EXPECT_TRUE(test_oddC()); }
