#include <half.hpp>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "conv_fwd_util.hpp"
#include "conv_util.hpp"

namespace {

bool test_conv3d_ndhwc()
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;

    conv::ConvParams params;
    params.num_dim_spatial        = 3;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 4;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{16, 16, 16};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3>(conv_ptrs);
    conv::ConvFwdOpInstance<float, float, float, ctl::NDHWC, ctl::KZYXC, ctl::NDHWK> conv_instance(
        params);

    auto reference_conv_fwd_fun = std::bind(
        conv::run_reference_convolution_forward<3, float, float, float>, params, _1, _2, _3);
    OpInstanceRunEngine<float, float, float> run_engine(conv_instance, reference_conv_fwd_fun);
    run_engine.SetAtol(1e-5);
    run_engine.SetRtol(1e-4);
    return run_engine.Test(conv_ptrs);
}

bool test_conv3d_ndhwc_2gb_input()
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using namespace ck::utils;

    // >2GB Input
    conv::ConvParams params;
    params.num_dim_spatial        = 3;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 32;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{32, 1000, 1000};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3>(conv_ptrs);

    auto arg = conv_ptrs.back()->MakeArgumentPointer(nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     params.N,
                                                     params.K,
                                                     params.C,
                                                     params.input_spatial_lengths,
                                                     params.filter_spatial_lengths,
                                                     params.GetOutputSpatialLengths(),
                                                     params.conv_filter_strides,
                                                     params.conv_filter_dilations,
                                                     params.input_left_pads,
                                                     params.input_right_pads,
                                                     PassThrough{},
                                                     PassThrough{},
                                                     PassThrough{});
    return !(conv_ptrs.back()->IsSupportedArgument(arg.get()));
}

bool test_conv3d_ndhwc_2gb_filters()
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using namespace ck::utils;

    // >2GB Filters
    conv::ConvParams params;
    params.num_dim_spatial        = 3;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 32;
    params.filter_spatial_lengths = std::vector<ck::index_t>{4, 1000, 1000};
    params.input_spatial_lengths  = std::vector<ck::index_t>{16, 16, 16};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3>(conv_ptrs);

    auto arg = conv_ptrs.back()->MakeArgumentPointer(nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     params.N,
                                                     params.K,
                                                     params.C,
                                                     params.input_spatial_lengths,
                                                     params.filter_spatial_lengths,
                                                     params.GetOutputSpatialLengths(),
                                                     params.conv_filter_strides,
                                                     params.conv_filter_dilations,
                                                     params.input_left_pads,
                                                     params.input_right_pads,
                                                     PassThrough{},
                                                     PassThrough{},
                                                     PassThrough{});
    return !(conv_ptrs.back()->IsSupportedArgument(arg.get()));
}

bool test_conv3d_ndhwc_2gb_output()
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using namespace ck::utils;

    // >2GB Output
    conv::ConvParams params;
    params.num_dim_spatial        = 3;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 2;
    params.filter_spatial_lengths = std::vector<ck::index_t>{1, 1, 1};
    params.input_spatial_lengths  = std::vector<ck::index_t>{1000, 1000, 30};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{2, 2, 2};
    params.input_right_pads       = std::vector<ck::index_t>{2, 2, 2};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<3>(conv_ptrs);
    auto arg = conv_ptrs.back()->MakeArgumentPointer(nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     params.N,
                                                     params.K,
                                                     params.C,
                                                     params.input_spatial_lengths,
                                                     params.filter_spatial_lengths,
                                                     params.GetOutputSpatialLengths(),
                                                     params.conv_filter_strides,
                                                     params.conv_filter_dilations,
                                                     params.input_left_pads,
                                                     params.input_right_pads,
                                                     PassThrough{},
                                                     PassThrough{},
                                                     PassThrough{});
    return !(conv_ptrs.back()->IsSupportedArgument(arg.get()));
}

template <typename T>
bool test_conv3d_ndhwc_instances(const std::vector<test::conv::DeviceConvFwdNoOpPtr>& conv_ptrs)
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;

    conv::ConvParams params;
    params.N                      = 64;
    params.num_dim_spatial        = 3;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3, 2};
    params.input_spatial_lengths  = std::vector<ck::index_t>{32, 32, 2};
    params.conv_filter_strides    = std::vector<ck::index_t>{2, 2, 2};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    conv::ConvFwdOpInstance<T, T, T, ctl::NDHWC, ctl::KZYXC, ctl::NDHWK> conv_instance(params);

    auto reference_conv_fwd_fun =
        std::bind(conv::run_reference_convolution_forward<3, T, T, T>, params, _1, _2, _3);
    OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    return run_engine.Test(conv_ptrs);
}

bool test_conv3d_ndhwc_bf16_instances()
{
    return test_conv3d_ndhwc_instances<ck::bhalf_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>::Get<3>());
}

bool test_conv3d_ndhwc_f16_instances()
{
    return test_conv3d_ndhwc_instances<ck::half_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::half_t, ck::half_t, ck::half_t>::Get<3>());
}

bool test_conv3d_ndhwc_f32_instances()
{
    return test_conv3d_ndhwc_instances<float>(
        ck::utils::conv::ConvolutionFwdInstances<float, float, float>::Get<3>());
}

bool test_conv3d_ndhwc_int8_instances()
{
    return test_conv3d_ndhwc_instances<int8_t>(
        ck::utils::conv::ConvolutionFwdInstances<int8_t, int8_t, int8_t>::Get<3>());
}

} // anonymous namespace

int main()
{
    bool res{true};
    res = test_conv3d_ndhwc();
    std::cout << "test_conv3d_ndhwc ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = test_conv3d_ndhwc_2gb_input();
    std::cout << "\ntest_conv3d_ndhwc_2gb_input ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv3d_ndhwc_2gb_filters();
    std::cout << "\ntest_conv3d_ndhwc_2gb_filters ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv3d_ndhwc_2gb_output();
    std::cout << "\ntest_conv3d_ndhwc_2gb_output ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;

    res = test_conv3d_ndhwc_bf16_instances();
    std::cout << "\ntest_conv3d_ndhwc_bf16_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv3d_ndhwc_f16_instances();
    std::cout << "\ntest_conv3d_ndhwc_f16_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv3d_ndhwc_f32_instances();
    std::cout << "\ntest_conv3d_ndhwc_f32_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv3d_ndhwc_int8_instances();
    std::cout << "\ntest_conv3d_ndhwc_int8_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;

    return res ? 0 : 1;
}
