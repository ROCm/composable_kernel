// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/conv_util.hpp"

#include "test/convnd_fwd/conv_util.hpp"

namespace {

class Conv3dFwdNDHWCInstances : public ::testing::Test
{
    public:
    template <typename T>
    bool test_conv3d_nwc_instances(const std::vector<test::conv::DeviceConvFwdNoOpPtr>& conv_ptrs,
                                   const ck::utils::conv::ConvParams& params)
    {
        using namespace std::placeholders;
        using namespace ck::utils;
        namespace ctl = ck::tensor_layout::convolution;

        conv::ConvFwdOpInstance<T,
                                T,
                                T,
                                ctl::NDHWC,
                                ctl::KZYXC,
                                ctl::NDHWK,
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
            std::bind(conv::run_reference_convolution_forward<3, T, T, T>, params, _1, _2, _3);
        OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
        run_engine.SetAtol(atol_);
        run_engine.SetRtol(rtol_);
        return run_engine.Test(conv_ptrs);
    }

    template <typename T>
    bool test_default()
    {
        return test_conv3d_nwc_instances<T>(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<3>(), params_default_);
    }

    template <typename T>
    bool test_filter1x1_stride1_pad0()
    {
        return test_conv3d_nwc_instances<T>(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<3>(),
            params_filter1x1_stride1_pad0_);
    }

    template <typename T>
    bool test_filter1x1_pad0()
    {
        return test_conv3d_nwc_instances<T>(
            ck::utils::conv::ConvolutionFwdInstances<T, T, T>::template Get<3>(),
            params_filter1x1_pad0_);
    }

    static inline ck::utils::conv::ConvParams params_default_{
        3, 4, 256, 64, {3, 3, 3}, {28, 28, 28}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    static inline ck::utils::conv::ConvParams params_filter1x1_stride1_pad0_{
        3, 4, 256, 64, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    static inline ck::utils::conv::ConvParams params_filter1x1_pad0_{
        3, 4, 256, 64, {1, 1, 1}, {28, 28, 28}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};

    private:
    double atol_{1e-5};
    double rtol_{1e-4};
};

} // anonymous namespace

TEST(Conv3DFwdNDHWC, IntegerValues)
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;
    using T       = float;

    ck::utils::conv::ConvParams params{
        3, 4, 256, 64, {3, 3, 3}, {18, 18, 18}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3, T, T, T, T>(conv_ptrs);
    conv::ConvFwdOpInstance<T,
                            T,
                            T,
                            ctl::NDHWC,
                            ctl::KZYXC,
                            ctl::NDHWK,
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
        std::bind(conv::run_reference_convolution_forward<3, T, T, T>, params, _1, _2, _3);
    OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    run_engine.SetAtol(1e-5);
    run_engine.SetRtol(1e-3);
    EXPECT_TRUE(run_engine.Test(conv_ptrs));
}

TEST(Conv3DFwdNDHWC, FloatingPointValues)
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;
    using T       = ck::half_t;

    ck::utils::conv::ConvParams params{
        3, 4, 256, 64, {3, 3, 3}, {18, 18, 18}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3, T, T, T, float>(conv_ptrs);
    conv::ConvFwdOpInstance<T,
                            T,
                            T,
                            ctl::NDHWC,
                            ctl::KZYXC,
                            ctl::NDHWK,
                            ck::tensor_operation::element_wise::PassThrough,
                            ck::tensor_operation::element_wise::PassThrough,
                            ck::tensor_operation::element_wise::PassThrough,
                            FillUniformDistribution<T>,
                            FillUniformDistribution<T>>
        conv_instance(params, true, FillUniformDistribution<T>{}, FillUniformDistribution<T>{});

    auto reference_conv_fwd_fun =
        std::bind(conv::run_reference_convolution_forward<3, T, T, T>, params, _1, _2, _3);
    OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    run_engine.SetAtol(1e-3);
    run_engine.SetRtol(1e-3);
    EXPECT_TRUE(run_engine.Test(conv_ptrs));
}

TEST(Conv3DFwdNDHWC, InputOver2GB)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using namespace ck::utils;
    using T = float;

    // >2GB Input
    conv::ConvParams params;
    params.num_dim_spatial_        = 3;
    params.N_                      = 2;
    params.K_                      = 16;
    params.C_                      = 32;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{32, 1000, 1000};
    params.conv_filter_strides_    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads_       = std::vector<ck::index_t>{1, 1, 1};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3, T, T, T, T>(conv_ptrs);
    auto arg = conv_ptrs.back()->MakeArgumentPointer(nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     params.N_,
                                                     params.K_,
                                                     params.C_,
                                                     params.input_spatial_lengths_,
                                                     params.filter_spatial_lengths_,
                                                     params.GetOutputSpatialLengths(),
                                                     params.conv_filter_strides_,
                                                     params.conv_filter_dilations_,
                                                     params.input_left_pads_,
                                                     params.input_right_pads_,
                                                     PassThrough{},
                                                     PassThrough{},
                                                     PassThrough{});
    EXPECT_FALSE(conv_ptrs.back()->IsSupportedArgument(arg.get()));
}

TEST(Conv3DFwdNDHWC, FiltersOver2GB)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using namespace ck::utils;
    using T = float;

    // >2GB Filters
    conv::ConvParams params;
    params.num_dim_spatial_        = 3;
    params.N_                      = 2;
    params.K_                      = 16;
    params.C_                      = 32;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{4, 1000, 1000};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{16, 16, 16};
    params.conv_filter_strides_    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads_       = std::vector<ck::index_t>{1, 1, 1};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3, T, T, T, T>(conv_ptrs);
    auto arg = conv_ptrs.back()->MakeArgumentPointer(nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     params.N_,
                                                     params.K_,
                                                     params.C_,
                                                     params.input_spatial_lengths_,
                                                     params.filter_spatial_lengths_,
                                                     params.GetOutputSpatialLengths(),
                                                     params.conv_filter_strides_,
                                                     params.conv_filter_dilations_,
                                                     params.input_left_pads_,
                                                     params.input_right_pads_,
                                                     PassThrough{},
                                                     PassThrough{},
                                                     PassThrough{});
    EXPECT_FALSE(conv_ptrs.back()->IsSupportedArgument(arg.get()));
}

TEST(Conv3DFwdNDHWC, OutputOver2GB)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using namespace ck::utils;
    using T = float;

    // >2GB Output
    conv::ConvParams params;
    params.num_dim_spatial_        = 3;
    params.N_                      = 2;
    params.K_                      = 16;
    params.C_                      = 2;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{1, 1, 1};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{1000, 1000, 30};
    params.conv_filter_strides_    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{2, 2, 2};
    params.input_right_pads_       = std::vector<ck::index_t>{2, 2, 2};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3, T, T, T, T>(conv_ptrs);
    auto arg = conv_ptrs.back()->MakeArgumentPointer(nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     params.N_,
                                                     params.K_,
                                                     params.C_,
                                                     params.input_spatial_lengths_,
                                                     params.filter_spatial_lengths_,
                                                     params.GetOutputSpatialLengths(),
                                                     params.conv_filter_strides_,
                                                     params.conv_filter_dilations_,
                                                     params.input_left_pads_,
                                                     params.input_right_pads_,
                                                     PassThrough{},
                                                     PassThrough{},
                                                     PassThrough{});
    EXPECT_FALSE(conv_ptrs.back()->IsSupportedArgument(arg.get()));
}

TEST_F(Conv3dFwdNDHWCInstances, BF16_default) { EXPECT_TRUE(this->test_default<ck::bhalf_t>()); }
TEST_F(Conv3dFwdNDHWCInstances, BF16_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0<ck::bhalf_t>());
}
TEST_F(Conv3dFwdNDHWCInstances, BF16_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0<ck::bhalf_t>());
}

TEST_F(Conv3dFwdNDHWCInstances, F16_default) { EXPECT_TRUE(this->test_default<ck::half_t>()); }
TEST_F(Conv3dFwdNDHWCInstances, F16_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0<ck::half_t>());
}
TEST_F(Conv3dFwdNDHWCInstances, F16_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0<ck::half_t>());
}

TEST_F(Conv3dFwdNDHWCInstances, F32_default) { EXPECT_TRUE(this->test_default<float>()); }
TEST_F(Conv3dFwdNDHWCInstances, F32_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0<float>());
}
TEST_F(Conv3dFwdNDHWCInstances, F32_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0<float>());
}

TEST_F(Conv3dFwdNDHWCInstances, I8_default) { EXPECT_TRUE(this->test_default<int8_t>()); }
TEST_F(Conv3dFwdNDHWCInstances, I8_filter1x1_stride1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_stride1_pad0<int8_t>());
}
TEST_F(Conv3dFwdNDHWCInstances, I8_filter1x1_pad0)
{
    EXPECT_TRUE(this->test_filter1x1_pad0<int8_t>());
}
