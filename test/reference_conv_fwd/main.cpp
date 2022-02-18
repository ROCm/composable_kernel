#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <half.hpp>
#include <numeric>
#include <type_traits>
#include <vector>

#include "config.hpp"
#include "conv_utils.hpp"
#include "element_wise_operation.hpp"
#include "host_tensor.hpp"
#include "reference_conv_fwd.hpp"
#include "tensor_layout.hpp"
#include "test_util.hpp"

namespace {
using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t NDim,
          typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float,
          typename InLayout    = ck::tensor_layout::convolution::NHWC,
          typename WeiLayout   = ck::tensor_layout::convolution::KYXC,
          typename OutLayout   = ck::tensor_layout::convolution::NHWK>
Tensor<OutDataType> RunReferenceConv(const ck::conv_util::ConvParams& params)
{
    std::vector<std::size_t> input_dims{static_cast<std::size_t>(params.N),
                                        static_cast<std::size_t>(params.C)};
    input_dims.insert(std::end(input_dims),
                      std::begin(params.input_spatial_lengths),
                      std::end(params.input_spatial_lengths));

    std::vector<std::size_t> filter_dims{static_cast<std::size_t>(params.K),
                                         static_cast<std::size_t>(params.C)};
    filter_dims.insert(std::end(filter_dims),
                       std::begin(params.filter_spatial_lengths),
                       std::end(params.filter_spatial_lengths));

    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();
    std::vector<std::size_t> output_dims{static_cast<std::size_t>(params.N),
                                         static_cast<std::size_t>(params.K)};
    output_dims.insert(std::end(output_dims),
                       std::begin(output_spatial_lengths),
                       std::end(output_spatial_lengths));

    Tensor<InDataType> input(ck::conv_util::GetHostTensorDescriptor(input_dims, InLayout{}));
    Tensor<WeiDataType> weights(ck::conv_util::GetHostTensorDescriptor(filter_dims, WeiLayout{}));
    Tensor<OutDataType> host_output(
        ck::conv_util::GetHostTensorDescriptor(output_dims, OutLayout{}));

    // init
    std::iota(input.begin(), input.end(), InDataType(0.f));
    std::fill(weights.begin(), weights.end(), WeiDataType(0.5f));
    std::fill(host_output.begin(), host_output.end(), OutDataType(0.f));

    auto ref_conv     = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                 WeiDataType,
                                                                 OutDataType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 OutElementOp,
                                                                 NDim>();
    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(input,
                                              weights,
                                              host_output,
                                              params.conv_filter_strides,
                                              params.conv_filter_dilations,
                                              params.input_left_pads,
                                              params.input_right_pads,
                                              InElementOp{},
                                              WeiElementOp{},
                                              OutElementOp{});

    ref_invoker.Run(ref_argument);
    return host_output;
}

bool TestConv2DNHWC()
{
    bool res{true};
    ck::conv_util::ConvParams params;
    params.N                      = 1;
    params.K                      = 1;
    params.C                      = 2;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{6, 6};
    params.conv_filter_strides    = std::vector<ck::index_t>{1, 1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1};
    params.input_left_pads        = std::vector<ck::index_t>{0, 0};
    params.input_right_pads       = std::vector<ck::index_t>{0, 0};

    auto out_tensor = RunReferenceConv<2>(params);
    std::vector<std::size_t> ref_dims{1, 1, 4, 4};
    std::vector<float> ref_data{130.5,
                                148.5,
                                166.5,
                                184.5,
                                238.5,
                                256.5,
                                274.5,
                                292.5,
                                346.5,
                                364.5,
                                382.5,
                                400.5,
                                454.5,
                                472.5,
                                490.5,
                                508.5};
    res = res && test_util::check_err(out_tensor.mDesc.GetLengths(),
                                      ref_dims,
                                      "Error: wrong output tensor dimensions!");
    res = res && test_util::check_err(out_tensor.mData, ref_data, "Error: incorrect results!");

    params.N                      = 1;
    params.K                      = 2;
    params.C                      = 2;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{12, 12};
    params.conv_filter_strides    = std::vector<ck::index_t>{2, 2};
    params.conv_filter_dilations  = std::vector<ck::index_t>{2, 2};
    params.input_left_pads        = std::vector<ck::index_t>{1, 1};
    params.input_right_pads       = std::vector<ck::index_t>{1, 1};

    out_tensor = RunReferenceConv<2>(params);
    ref_dims   = std::vector<std::size_t>{1, 2, 5, 5};
    ref_data   = std::vector<float>{
        210.,  210.,  327.,   327.,   351.,   351.,   375.,   375.,   399.,   399.,
        459.,  459.,  706.5,  706.5,  742.5,  742.5,  778.5,  778.5,  814.5,  814.5,
        747.,  747.,  1138.5, 1138.5, 1174.5, 1174.5, 1210.5, 1210.5, 1246.5, 1246.5,
        1035., 1035., 1570.5, 1570.5, 1606.5, 1606.5, 1642.5, 1642.5, 1678.5, 1678.5,
        1323., 1323., 2002.5, 2002.5, 2038.5, 2038.5, 2074.5, 2074.5, 2110.5, 2110.5};
    res = res && test_util::check_err(out_tensor.mDesc.GetLengths(),
                                      ref_dims,
                                      "Error: wrong output tensor dimensions!");
    res = res && test_util::check_err(out_tensor.mData, ref_data, "Error: incorrect results!");

    return res;
}

bool TestConv1DNHWC()
{
    bool res{true};
    ck::conv_util::ConvParams params;
    params.spatial_dims           = 1;
    params.N                      = 1;
    params.K                      = 1;
    params.C                      = 2;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{6};
    params.conv_filter_strides    = std::vector<ck::index_t>{1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1};
    params.input_left_pads        = std::vector<ck::index_t>{0};
    params.input_right_pads       = std::vector<ck::index_t>{0};

    auto out_tensor = RunReferenceConv<1,
                                       float,
                                       float,
                                       float,
                                       ck::tensor_layout::convolution::NWC,
                                       ck::tensor_layout::convolution::KXC,
                                       ck::tensor_layout::convolution::NWK>(params);
    std::vector<std::size_t> ref_dims{1, 1, 4};
    std::vector<float> ref_data{7.5, 13.5, 19.5, 25.5};
    res = res && test_util::check_err(out_tensor.mDesc.GetLengths(),
                                      ref_dims,
                                      "Error: wrong output tensor dimensions!");
    res = res && test_util::check_err(out_tensor.mData, ref_data, "Error: incorrect results!");

    params.spatial_dims           = 1;
    params.N                      = 1;
    params.K                      = 2;
    params.C                      = 2;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{12};
    params.conv_filter_strides    = std::vector<ck::index_t>{2};
    params.conv_filter_dilations  = std::vector<ck::index_t>{2};
    params.input_left_pads        = std::vector<ck::index_t>{1};
    params.input_right_pads       = std::vector<ck::index_t>{1};

    out_tensor = RunReferenceConv<1,
                                  float,
                                  float,
                                  float,
                                  ck::tensor_layout::convolution::NWC,
                                  ck::tensor_layout::convolution::KXC,
                                  ck::tensor_layout::convolution::NWK>(params);
    ref_dims   = std::vector<std::size_t>{1, 2, 5};
    ref_data   = std::vector<float>{9., 9., 19.5, 19.5, 31.5, 31.5, 43.5, 43.5, 55.5, 55.5};
    res        = res && test_util::check_err(out_tensor.mDesc.GetLengths(),
                                      ref_dims,
                                      "Error: wrong output tensor dimensions!");
    res = res && test_util::check_err(out_tensor.mData, ref_data, "Error: incorrect results!");

    return res;
}

} // anonymous namespace

int main(void)
{
    bool res{true};
    res = TestConv2DNHWC();
    std::cout << "TestConv2DNHWC ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = TestConv1DNHWC();
    std::cout << "TestConv1DNHWC ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    return 0;
}
