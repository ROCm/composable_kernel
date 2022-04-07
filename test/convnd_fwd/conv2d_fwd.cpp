#include <half.hpp>
#include <iostream>
#include <tuple>
#include <vector>

#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "conv_fwd_util.hpp"
#include "conv_util.hpp"
#include "host_tensor.hpp"
#include "tensor_layout.hpp"
#include "check_err.hpp"

// Forward declarations for conv instances.
using DeviceConvFwdNoOpPtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough>;

namespace {

bool test_conv2d_nhwc()
{
    bool res{true};
    ck::utils::conv::ConvParams params;
    params.N                     = 2;
    params.K                     = 16;
    params.C                     = 4;
    params.input_spatial_lengths = std::vector<ck::index_t>{16, 16};
    params.conv_filter_strides   = std::vector<ck::index_t>{1, 1};

    auto host_tensors            = ck::utils::conv::get_host_tensors(params);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& host_output   = std::get<2>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    ck::utils::conv::run_reference_convolution_forward<2>(params, input, weights, host_output);
    test::conv::RunConv<2>(params, input, weights, device_output);
    res = res &&
          ck::utils::check_err(
              device_output.mData, host_output.mData, "Error: incorrect results!", 1e-5f, 1e-4f);

    return res;
}

template <typename T>
bool test_conv2d_nhwc_instances(const std::vector<DeviceConvFwdNoOpPtr>& conv_ptrs)
{
    ck::utils::conv::ConvParams params;
    params.num_dim_spatial        = 2;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{71, 71};
    params.conv_filter_strides    = std::vector<ck::index_t>{2, 2};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1};

    ck::utils::conv::ConvFwdOpInstance<T, T, T> conv_instance(params);
    using namespace std::placeholders;

    auto reference_conv_fwd_fun = std::bind(
        ck::utils::conv::run_reference_convolution_forward<2, T, T, T>, params, _1, _2, _3);
    ck::utils::OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    return run_engine.Test(conv_ptrs);
}

bool test_conv2d_nhwc_bf16_instances()
{
    return test_conv2d_nhwc_instances<ck::bhalf_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>::Get<2>());
}

bool test_conv2d_nhwc_f16_instances()
{
    return test_conv2d_nhwc_instances<ck::half_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::half_t, ck::half_t, ck::half_t>::Get<2>());
}

bool test_conv2d_nhwc_f32_instances()
{
    return test_conv2d_nhwc_instances<float>(
        ck::utils::conv::ConvolutionFwdInstances<float, float, float>::Get<2>());
}

bool test_conv2d_nhwc_int8_instances()
{
    return test_conv2d_nhwc_instances<int8_t>(
        ck::utils::conv::ConvolutionFwdInstances<int8_t, int8_t, int8_t>::Get<2>());
}

} // anonymous namespace

int main()
{
    bool res{true};
    res = test_conv2d_nhwc();
    std::cout << "test_conv2d_nhwc ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = test_conv2d_nhwc_bf16_instances();
    std::cout << "\ntest_conv2d_nhwc_bf16_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv2d_nhwc_f16_instances();
    std::cout << "\ntest_conv2d_nhwc_f16_instances ....." << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv2d_nhwc_f32_instances();
    std::cout << "\ntest_conv2d_nhwc_f32_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv2d_nhwc_int8_instances();
    std::cout << "\ntest_conv2d_nhwc_int8_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;

    return res ? 0 : 1;
}
