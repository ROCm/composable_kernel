#include <half.hpp>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "gtest/gtest.h"

#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "library/include/ck/library/utility/conv_util.hpp"
#include "conv_util.hpp"

namespace {

template <typename T>
bool test_conv3d_ndhwc_instances(const std::vector<test::conv::DeviceConvFwdNoOpPtr>& conv_ptrs)
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;

    conv::ConvParams params;
    params.N_                      = 64;
    params.num_dim_spatial_        = 3;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3, 3, 2};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{32, 32, 2};
    params.conv_filter_strides_    = std::vector<ck::index_t>{2, 2, 2};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads_       = std::vector<ck::index_t>{1, 1, 1};

    conv::ConvFwdOpInstance<T, T, T, ctl::NDHWC, ctl::KZYXC, ctl::NDHWK> conv_instance(params);

    auto reference_conv_fwd_fun =
        std::bind(conv::run_reference_convolution_forward<3, T, T, T>, params, _1, _2, _3);
    OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    return run_engine.Test(conv_ptrs);
}

} // anonymous namespace

TEST(Conv3DFwdNDHWC, TestConv3D)
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;

    conv::ConvParams params;
    params.num_dim_spatial_        = 3;
    params.N_                      = 2;
    params.K_                      = 16;
    params.C_                      = 4;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{16, 16, 16};
    params.conv_filter_strides_    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads_       = std::vector<ck::index_t>{1, 1, 1};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3>(conv_ptrs);
    conv::ConvFwdOpInstance<float, float, float, ctl::NDHWC, ctl::KZYXC, ctl::NDHWK> conv_instance(
        params);

    auto reference_conv_fwd_fun = std::bind(
        conv::run_reference_convolution_forward<3, float, float, float>, params, _1, _2, _3);
    OpInstanceRunEngine<float, float, float> run_engine(conv_instance, reference_conv_fwd_fun);
    run_engine.SetAtol(1e-5);
    run_engine.SetRtol(1e-4);
    EXPECT_TRUE(run_engine.Test(conv_ptrs));
}

TEST(Conv3DFwdNDHWC, InputOver2GB)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using namespace ck::utils;

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
    test::conv::get_test_convolution_fwd_instance<3>(conv_ptrs);

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
    test::conv::get_test_convolution_fwd_instance<3>(conv_ptrs);

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
    test::conv::get_test_convolution_fwd_instance<3>(conv_ptrs);
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

TEST(Conv3DFwdNDHWC, Bf16Instances)
{
    EXPECT_TRUE(test_conv3d_ndhwc_instances<ck::bhalf_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>::Get<3>()));
}

TEST(Conv3DFwdNDHWC, F16Instances)
{
    EXPECT_TRUE(test_conv3d_ndhwc_instances<ck::half_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::half_t, ck::half_t, ck::half_t>::Get<3>()));
}

TEST(Conv3DFwdNDHWC, F32Instances)
{
    EXPECT_TRUE(test_conv3d_ndhwc_instances<float>(
        ck::utils::conv::ConvolutionFwdInstances<float, float, float>::Get<3>()));
}

TEST(Conv3DFwdNDHWC, Int8Instances)
{
    EXPECT_TRUE(test_conv3d_ndhwc_instances<int8_t>(
        ck::utils::conv::ConvolutionFwdInstances<int8_t, int8_t, int8_t>::Get<3>()));
}
