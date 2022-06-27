// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/conv_util.hpp"
#include "test/convnd_fwd/conv_util.hpp"

namespace {

class Conv1dFwdNWCInstances : public ::testing::Test
{
    public:
    template <typename T>
    bool test_conv1d_nwc_instances(const std::vector<test::conv::DeviceConvFwdNoOpPtr>& conv_ptrs,
                                   const ck::utils::conv::ConvParams& params)
    {
        using namespace std::placeholders;
        using namespace ck::utils;
        namespace ctl = ck::tensor_layout::convolution;

        conv::ConvFwdOpInstance<T,
                                T,
                                T,
                                ctl::NWC,
                                ctl::KXC,
                                ctl::NWK,
                                ck::tensor_operation::element_wise::PassThrough,
                                ck::tensor_operation::element_wise::PassThrough,
                                ck::tensor_operation::element_wise::PassThrough,
                                FillUniformDistributionIntegerValue<T>,
                                FillUniformDistributionIntegerValue<T>>
            conv_instance(params,
                          true,
                          FillUniformDistributionIntegerValue<T>{},
                          FillUniformDistributionIntegerValue<T>{});
        auto reference_conv_fwd_fun =
            std::bind(conv::run_reference_convolution_forward<1, T, T, T>, params, _1, _2, _3);
        OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
        run_engine.SetAtol(atol_);
        run_engine.SetRtol(rtol_);
        return run_engine.Test(conv_ptrs);
    }

    template <typename T>
    bool test_default()
    {
        return test_conv1d_nwc_instances<T>(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<1>(), params_default_);
    }

    template <typename T>
    bool test_filter1x1_stride1_pad0()
    {
        return test_conv1d_nwc_instances<T>(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<1>(),
            params_filter1x1_stride1_pad0_);
    }

    template <typename T>
    bool test_filter1x1_pad0()
    {
        return test_conv1d_nwc_instances<T>(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<1>(),
            params_filter1x1_pad0_);
    }

    static inline ck::utils::conv::ConvParams params_default_{
        1, 4, 256, 64, {3}, {71}, {2}, {2}, {2}, {2}};
    static inline ck::utils::conv::ConvParams params_filter1x1_stride1_pad0_{
        1, 4, 256, 64, {1}, {28}, {1}, {1}, {0}, {0}};
    static inline ck::utils::conv::ConvParams params_filter1x1_pad0_{
        1, 4, 256, 64, {1}, {28}, {2}, {1}, {0}, {0}};

    private:
    double atol_{1e-5};
    double rtol_{1e-4};
};

} // anonymous namespace

TEST(Conv1DFwdNWC, IntegerValues)
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;
    using T       = float;

    ck::utils::conv::ConvParams params{1, 4, 256, 64, {3}, {36}, {1}, {2}, {2}, {2}};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<1, T, T, T, T>(conv_ptrs);
    conv::ConvFwdOpInstance<T,
                            T,
                            T,
                            ctl::NWC,
                            ctl::KXC,
                            ctl::NWK,
                            ck::tensor_operation::element_wise::PassThrough,
                            ck::tensor_operation::element_wise::PassThrough,
                            ck::tensor_operation::element_wise::PassThrough,
                            FillUniformDistributionIntegerValue<T>,
                            FillUniformDistributionIntegerValue<T>>
        conv_instance(params,
                      true,
                      FillUniformDistributionIntegerValue<T>{},
                      FillUniformDistributionIntegerValue<T>{});

    auto reference_conv_fwd_fun =
        std::bind(conv::run_reference_convolution_forward<1, T, T, T>, params, _1, _2, _3);
    OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    run_engine.SetAtol(1e-5);
    run_engine.SetRtol(1e-4);
    EXPECT_TRUE(run_engine.Test(conv_ptrs));
}

TEST(Conv1DFwdNWC, FloatingPointValues)
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;
    using T       = ck::half_t;

    ck::utils::conv::ConvParams params{1, 4, 256, 64, {3}, {36}, {1}, {2}, {2}, {2}};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<1, T, T, T, float>(conv_ptrs);
    conv::ConvFwdOpInstance<T,
                            T,
                            T,
                            ctl::NWC,
                            ctl::KXC,
                            ctl::NWK,
                            ck::tensor_operation::element_wise::PassThrough,
                            ck::tensor_operation::element_wise::PassThrough,
                            ck::tensor_operation::element_wise::PassThrough,
                            FillUniformDistribution<T>,
                            FillUniformDistribution<T>>
        conv_instance(params, true, FillUniformDistribution<T>{}, FillUniformDistribution<T>{});

    auto reference_conv_fwd_fun =
        std::bind(conv::run_reference_convolution_forward<1, T, T, T>, params, _1, _2, _3);
    OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    run_engine.SetAtol(0.1);
    run_engine.SetRtol(1e-2);
    EXPECT_TRUE(run_engine.Test(conv_ptrs));
}

TEST_F(Conv1dFwdNWCInstances, BF16_default) { EXPECT_TRUE(this->test_default<ck::bhalf_t>()); }
TEST_F(Conv1dFwdNWCInstances, BF16_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0<ck::bhalf_t>());
}
TEST_F(Conv1dFwdNWCInstances, BF16_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0<ck::bhalf_t>());
}

TEST_F(Conv1dFwdNWCInstances, F16_default) { EXPECT_TRUE(this->test_default<ck::half_t>()); }
TEST_F(Conv1dFwdNWCInstances, F16_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0<ck::half_t>());
}
TEST_F(Conv1dFwdNWCInstances, F16_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0<ck::half_t>());
}

TEST_F(Conv1dFwdNWCInstances, F32_default) { EXPECT_TRUE(this->test_default<float>()); }
TEST_F(Conv1dFwdNWCInstances, F32_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0<float>());
}
TEST_F(Conv1dFwdNWCInstances, F32_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0<float>());
}

TEST_F(Conv1dFwdNWCInstances, I8_default) { EXPECT_TRUE(this->test_default<int8_t>()); }
TEST_F(Conv1dFwdNWCInstances, I8_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0<int8_t>());
}
TEST_F(Conv1dFwdNWCInstances, I8_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0<int8_t>());
}
