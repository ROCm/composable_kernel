#include <half.hpp>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "conv_test_util.hpp"
#include "host_tensor.hpp"
#include "tensor_layout.hpp"
#include "test_util.hpp"

// Forward declarations for conv instances.
using DeviceConvFwdNoOpPtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv3d_fwd_instance {

void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace device_conv3d_fwd_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace {

bool TestConv3DNDHWC()
{
    bool res{true};
    ck::conv_util::ConvParams params;
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

    auto host_tensors            = test::conv::GetHostTensors<float,
                                                   float,
                                                   float,
                                                   ck::tensor_layout::convolution::NDHWC,
                                                   ck::tensor_layout::convolution::KZYXC,
                                                   ck::tensor_layout::convolution::NDHWK>(params);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& host_output   = std::get<2>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    test::conv::RunReferenceConv<3>(params, input, weights, host_output);
    test::conv::RunConv<3>(params, input, weights, device_output);
    res = res &&
          test::check_err(
              device_output.mData, host_output.mData, "Error: incorrect results!", 1e-5f, 1e-4f);

    return res;
}

bool TestConv3DNDHWC2GBInput()
{
    // >2GB Input
    ck::conv_util::ConvParams params;
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

    auto host_tensors =
        test::conv::GetHostTensors<float,
                                   float,
                                   float,
                                   ck::tensor_layout::convolution::NDHWC,
                                   ck::tensor_layout::convolution::KZYXC,
                                   ck::tensor_layout::convolution::NDHWK>(params, false);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    try
    {
        test::conv::RunConv<3>(params, input, weights, device_output);
    }
    catch(const std::runtime_error& err)
    {
        std::string err_msg{"Error! device_conv with the specified compilation parameters does "
                            "not support this Conv problem"};
        if(err.what() != err_msg)
        {
            return false;
        }
        return true;
    }
    std::cout << "Error: Failure checking oversized tensor!" << std::endl;
    return false;
}

bool TestConv3DNDHWC2GBFilters()
{
    // >2GB Filters
    ck::conv_util::ConvParams params;
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

    auto host_tensors =
        test::conv::GetHostTensors<float,
                                   float,
                                   float,
                                   ck::tensor_layout::convolution::NDHWC,
                                   ck::tensor_layout::convolution::KZYXC,
                                   ck::tensor_layout::convolution::NDHWK>(params, false);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    try
    {
        test::conv::RunConv<3>(params, input, weights, device_output);
    }
    catch(const std::runtime_error& err)
    {
        std::string err_msg{"Error! device_conv with the specified compilation parameters does "
                            "not support this Conv problem"};
        if(err.what() != err_msg)
        {
            return false;
        }
        return true;
    }
    std::cout << "Error: Failure checking oversized tensor!" << std::endl;
    return false;
}

bool TestConv3DNDHWC2GBOutput()
{
    // >2GB Output
    ck::conv_util::ConvParams params;
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

    auto host_tensors =
        test::conv::GetHostTensors<float,
                                   float,
                                   float,
                                   ck::tensor_layout::convolution::NDHWC,
                                   ck::tensor_layout::convolution::KZYXC,
                                   ck::tensor_layout::convolution::NDHWK>(params, false);
    const Tensor<float>& input   = std::get<0>(host_tensors);
    const Tensor<float>& weights = std::get<1>(host_tensors);
    Tensor<float>& device_output = std::get<3>(host_tensors);

    try
    {
        test::conv::RunConv<3>(params, input, weights, device_output);
    }
    catch(const std::runtime_error& err)
    {
        std::string err_msg{"Error! device_conv with the specified compilation parameters does "
                            "not support this Conv problem"};
        if(err.what() != err_msg)
        {
            return false;
        }
        return true;
    }
    std::cout << "Error: Failure checking oversized tensor!" << std::endl;
    return false;
}

template <typename T>
bool TestConv3DNDHWCInstances(const std::vector<DeviceConvFwdNoOpPtr>& conv_ptrs)
{
    ck::conv_util::ConvParams params;
    params.num_dim_spatial        = 3;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3, 2};
    params.input_spatial_lengths  = std::vector<ck::index_t>{71, 71, 2};
    params.conv_filter_strides    = std::vector<ck::index_t>{2, 2, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    auto host_tensors        = test::conv::GetHostTensors<T,
                                                   T,
                                                   T,
                                                   ck::tensor_layout::convolution::NDHWC,
                                                   ck::tensor_layout::convolution::KZYXC,
                                                   ck::tensor_layout::convolution::NDHWK>(params);
    const Tensor<T>& input   = std::get<0>(host_tensors);
    const Tensor<T>& weights = std::get<1>(host_tensors);
    Tensor<T>& host_output   = std::get<2>(host_tensors);
    Tensor<T>& device_output = std::get<3>(host_tensors);

    test::conv::RunReferenceConv<3>(params, input, weights, host_output);
    return test::conv::RunConvInstances<3>(
        params, conv_ptrs, input, weights, device_output, host_output);
}

bool TestConv3DNDHWCBF16Instances()
{
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv3d_fwd_instance::
        add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(conv_ptrs);
    return TestConv3DNDHWCInstances<ck::bhalf_t>(conv_ptrs);
}

bool TestConv3DNDHWCF16Instances()
{
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv3d_fwd_instance::
        add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f16_instances(conv_ptrs);
    return TestConv3DNDHWCInstances<ck::half_t>(conv_ptrs);
}

bool TestConv3DNDHWCF32Instances()
{
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv3d_fwd_instance::
        add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f32_instances(conv_ptrs);
    return TestConv3DNDHWCInstances<float>(conv_ptrs);
}

bool TestConv3DNDHWCInt8Instances()
{
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    ck::tensor_operation::device::device_conv3d_fwd_instance::
        add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_int8_instances(conv_ptrs);
    return TestConv3DNDHWCInstances<int8_t>(conv_ptrs);
}

} // anonymous namespace

int main()
{
    bool res{true};
    res = TestConv3DNDHWC();
    std::cout << "TestConv3DNDHWC ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = TestConv3DNDHWC2GBInput();
    std::cout << "\nTestConv3DNDHWC2GBInput ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv3DNDHWC2GBFilters();
    std::cout << "\nTestConv3DNDHWC2GBFilters ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv3DNDHWC2GBOutput();
    std::cout << "\nTestConv3DNDHWC2GBOutput ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = TestConv3DNDHWCBF16Instances();
    std::cout << "\nTestConv3DNDHWCBF16Instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = TestConv3DNDHWCF16Instances();
    std::cout << "\nTestConv3DNDHWCF16Instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = TestConv3DNDHWCF32Instances();
    std::cout << "\nTestConv3DNDHWCF32Instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = TestConv3DNDHWCInt8Instances();
    std::cout << "\nTestConv3DNDHWCInt8Instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;

    return 0;
}
