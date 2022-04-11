#include <half.hpp>
#include <iostream>
#include <stdexcept>
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

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_fwd_instance {

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace device_conv2d_fwd_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

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

    auto host_tensors =
        ck::utils::conv::get_host_tensors<T,
                                          T,
                                          T,
                                          ck::tensor_layout::convolution::NHWC,
                                          ck::tensor_layout::convolution::KYXC,
                                          ck::tensor_layout::convolution::NHWK>(params);
    const Tensor<T>& input   = std::get<0>(host_tensors);
    const Tensor<T>& weights = std::get<1>(host_tensors);
    Tensor<T>& host_output   = std::get<2>(host_tensors);
    Tensor<T>& device_output = std::get<3>(host_tensors);

    ck::utils::conv::run_reference_convolution_forward<2>(params, input, weights, host_output);
    return ck::utils::conv::run_convolution_forward_instances<2>(
        params, conv_ptrs, input, weights, device_output, host_output);
}

bool test_conv2d_nhwc_bf16_instances()
{
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv2d_fwd_instance::
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(conv_ptrs);
    return test_conv2d_nhwc_instances<ck::bhalf_t>(conv_ptrs);
}

bool test_conv2d_nhwc_f16_instances()
{
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv2d_fwd_instance::
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
    ck::tensor_operation::device::device_conv2d_fwd_instance::
        add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
    return test_conv2d_nhwc_instances<ck::half_t>(conv_ptrs);
}

bool test_conv2d_nhwc_f32_instances()
{
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv2d_fwd_instance::
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(conv_ptrs);
    return test_conv2d_nhwc_instances<float>(conv_ptrs);
}

bool test_conv2d_nhwc_int8_instances()
{
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv2d_fwd_instance::
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(conv_ptrs);
    return test_conv2d_nhwc_instances<int8_t>(conv_ptrs);
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
