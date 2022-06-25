// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/conv_util.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

namespace {
using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t NDim,
          typename InDataType    = float,
          typename WeiDataType   = float,
          typename OutDataType   = float,
          typename InLayout      = ck::tensor_layout::convolution::NHWC,
          typename WeiLayout     = ck::tensor_layout::convolution::KYXC,
          typename OutLayout     = ck::tensor_layout::convolution::NHWK,
          typename FillInputOp   = ck::utils::FillMonotonicSeq<InDataType>,
          typename FillWeightsOp = ck::utils::FillConstant<WeiDataType>>
Tensor<OutDataType>
run_reference_convolution_forward(const ck::utils::conv::ConvParams& params,
                                  const FillInputOp& fill_input_op     = FillInputOp{},
                                  const FillWeightsOp& fill_weights_op = FillWeightsOp{0.5f})
{
    std::vector<std::size_t> input_dims{static_cast<std::size_t>(params.N_),
                                        static_cast<std::size_t>(params.C_)};
    input_dims.insert(std::end(input_dims),
                      std::begin(params.input_spatial_lengths_),
                      std::end(params.input_spatial_lengths_));

    std::vector<std::size_t> filter_dims{static_cast<std::size_t>(params.K_),
                                         static_cast<std::size_t>(params.C_)};
    filter_dims.insert(std::end(filter_dims),
                       std::begin(params.filter_spatial_lengths_),
                       std::end(params.filter_spatial_lengths_));

    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();
    std::vector<std::size_t> output_dims{static_cast<std::size_t>(params.N_),
                                         static_cast<std::size_t>(params.K_)};
    output_dims.insert(std::end(output_dims),
                       std::begin(output_spatial_lengths),
                       std::end(output_spatial_lengths));

    Tensor<InDataType> input(ck::utils::conv::get_host_tensor_descriptor(input_dims, InLayout{}));
    Tensor<WeiDataType> weights(
        ck::utils::conv::get_host_tensor_descriptor(filter_dims, WeiLayout{}));
    Tensor<OutDataType> host_output(
        ck::utils::conv::get_host_tensor_descriptor(output_dims, OutLayout{}));

    fill_input_op(input.begin(), input.end());
    fill_weights_op(weights.begin(), weights.end());
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
                                              params.conv_filter_strides_,
                                              params.conv_filter_dilations_,
                                              params.input_left_pads_,
                                              params.input_right_pads_,
                                              InElementOp{},
                                              WeiElementOp{},
                                              OutElementOp{});

    ref_invoker.Run(ref_argument);
    return host_output;
}

} // anonymous namespace

TEST(ReferenceConvolutionFWD, Conv2DNHWC)
{
    ck::utils::conv::ConvParams params;
    params.N_                      = 1;
    params.K_                      = 1;
    params.C_                      = 2;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3, 3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{6, 6};
    params.conv_filter_strides_    = std::vector<ck::index_t>{1, 1};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{0, 0};
    params.input_right_pads_       = std::vector<ck::index_t>{0, 0};

    auto out_tensor = run_reference_convolution_forward<2>(params);
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
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor.mData, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv2DNHWCStridesDilationsPadding)
{
    ck::utils::conv::ConvParams params;
    params.N_                      = 1;
    params.K_                      = 2;
    params.C_                      = 2;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3, 3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{12, 12};
    params.conv_filter_strides_    = std::vector<ck::index_t>{2, 2};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{2, 2};
    params.input_left_pads_        = std::vector<ck::index_t>{1, 1};
    params.input_right_pads_       = std::vector<ck::index_t>{1, 1};

    auto out_tensor                   = run_reference_convolution_forward<2>(params);
    std::vector<std::size_t> ref_dims = std::vector<std::size_t>{1, 2, 5, 5};
    std::vector<float> ref_data{
        210.,  210.,  327.,   327.,   351.,   351.,   375.,   375.,   399.,   399.,
        459.,  459.,  706.5,  706.5,  742.5,  742.5,  778.5,  778.5,  814.5,  814.5,
        747.,  747.,  1138.5, 1138.5, 1174.5, 1174.5, 1210.5, 1210.5, 1246.5, 1246.5,
        1035., 1035., 1570.5, 1570.5, 1606.5, 1606.5, 1642.5, 1642.5, 1678.5, 1678.5,
        1323., 1323., 2002.5, 2002.5, 2038.5, 2038.5, 2074.5, 2074.5, 2110.5, 2110.5};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor.mData, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv1DNWC)
{
    ck::utils::conv::ConvParams params;
    params.num_dim_spatial_        = 1;
    params.N_                      = 1;
    params.K_                      = 1;
    params.C_                      = 2;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{6};
    params.conv_filter_strides_    = std::vector<ck::index_t>{1};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1};
    params.input_left_pads_        = std::vector<ck::index_t>{0};
    params.input_right_pads_       = std::vector<ck::index_t>{0};

    auto out_tensor =
        run_reference_convolution_forward<1,
                                          float,
                                          float,
                                          float,
                                          ck::tensor_layout::convolution::NWC,
                                          ck::tensor_layout::convolution::KXC,
                                          ck::tensor_layout::convolution::NWK>(params);
    std::vector<std::size_t> ref_dims{1, 1, 4};
    std::vector<float> ref_data{7.5, 13.5, 19.5, 25.5};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor.mData, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv1DNWCStridesDilationsPadding)
{
    ck::utils::conv::ConvParams params;
    params.num_dim_spatial_        = 1;
    params.N_                      = 1;
    params.K_                      = 2;
    params.C_                      = 2;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{12};
    params.conv_filter_strides_    = std::vector<ck::index_t>{2};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{2};
    params.input_left_pads_        = std::vector<ck::index_t>{1};
    params.input_right_pads_       = std::vector<ck::index_t>{1};

    auto out_tensor =
        run_reference_convolution_forward<1,
                                          float,
                                          float,
                                          float,
                                          ck::tensor_layout::convolution::NWC,
                                          ck::tensor_layout::convolution::KXC,
                                          ck::tensor_layout::convolution::NWK>(params);
    std::vector<std::size_t> ref_dims{1, 2, 5};
    std::vector<float> ref_data{9., 9., 19.5, 19.5, 31.5, 31.5, 43.5, 43.5, 55.5, 55.5};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor.mData, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv1DNWCSameOutputSize)
{
    ck::utils::conv::ConvParams params;
    params.num_dim_spatial_        = 1;
    params.N_                      = 2;
    params.K_                      = 16;
    params.C_                      = 4;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{16};
    params.conv_filter_strides_    = std::vector<ck::index_t>{1};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1};
    params.input_left_pads_        = std::vector<ck::index_t>{1};
    params.input_right_pads_       = std::vector<ck::index_t>{1};

    auto out_tensor2 = run_reference_convolution_forward<1,
                                                         float,
                                                         float,
                                                         float,
                                                         ck::tensor_layout::convolution::NWC,
                                                         ck::tensor_layout::convolution::KXC,
                                                         ck::tensor_layout::convolution::NWK>(
        params, ck::utils::FillMonotonicSeq<float>{0.f, 0.1f});

    std::vector<std::size_t> ref_dims{2, 16, 16};
    std::vector<float> ref_data{
        1.4,       1.4,       1.4,       1.4,       1.4,       1.4,       1.4,       1.4,
        1.4,       1.4,       1.4,       1.4,       1.4,       1.4,       1.4,       1.4,
        3.3,       3.3,       3.3,       3.3,       3.3,       3.3,       3.3,       3.3,
        3.3,       3.3,       3.3,       3.3,       3.3,       3.3,       3.3,       3.3,
        5.7,       5.7,       5.7,       5.7,       5.7,       5.7,       5.7,       5.7,
        5.7,       5.7,       5.7,       5.7,       5.7,       5.7,       5.7,       5.7,
        8.1,       8.1,       8.1,       8.1,       8.1,       8.1,       8.1,       8.1,
        8.1,       8.1,       8.1,       8.1,       8.1,       8.1,       8.1,       8.1,
        10.5,      10.5,      10.5,      10.5,      10.5,      10.5,      10.5,      10.5,
        10.5,      10.5,      10.5,      10.5,      10.5,      10.5,      10.5,      10.5,
        12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001,
        12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001, 12.900001,
        15.3,      15.3,      15.3,      15.3,      15.3,      15.3,      15.3,      15.3,
        15.3,      15.3,      15.3,      15.3,      15.3,      15.3,      15.3,      15.3,
        17.7,      17.7,      17.7,      17.7,      17.7,      17.7,      17.7,      17.7,
        17.7,      17.7,      17.7,      17.7,      17.7,      17.7,      17.7,      17.7,
        20.1,      20.1,      20.1,      20.1,      20.1,      20.1,      20.1,      20.1,
        20.1,      20.1,      20.1,      20.1,      20.1,      20.1,      20.1,      20.1,
        22.5,      22.5,      22.5,      22.5,      22.5,      22.5,      22.5,      22.5,
        22.5,      22.5,      22.5,      22.5,      22.5,      22.5,      22.5,      22.5,
        24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002,
        24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002, 24.900002,
        27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001,
        27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001, 27.300001,
        29.7,      29.7,      29.7,      29.7,      29.7,      29.7,      29.7,      29.7,
        29.7,      29.7,      29.7,      29.7,      29.7,      29.7,      29.7,      29.7,
        32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002,
        32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002, 32.100002,
        34.5,      34.5,      34.5,      34.5,      34.5,      34.5,      34.5,      34.5,
        34.5,      34.5,      34.5,      34.5,      34.5,      34.5,      34.5,      34.5,
        23.8,      23.8,      23.8,      23.8,      23.8,      23.8,      23.8,      23.8,
        23.8,      23.8,      23.8,      23.8,      23.8,      23.8,      23.8,      23.8,
        27.,       27.,       27.,       27.,       27.,       27.,       27.,       27.,
        27.,       27.,       27.,       27.,       27.,       27.,       27.,       27.,
        41.7,      41.7,      41.7,      41.7,      41.7,      41.7,      41.7,      41.7,
        41.7,      41.7,      41.7,      41.7,      41.7,      41.7,      41.7,      41.7,
        44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002,
        44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002, 44.100002,
        46.5,      46.5,      46.5,      46.5,      46.5,      46.5,      46.5,      46.5,
        46.5,      46.5,      46.5,      46.5,      46.5,      46.5,      46.5,      46.5,
        48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998,
        48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998, 48.899998,
        51.3,      51.3,      51.3,      51.3,      51.3,      51.3,      51.3,      51.3,
        51.3,      51.3,      51.3,      51.3,      51.3,      51.3,      51.3,      51.3,
        53.7,      53.7,      53.7,      53.7,      53.7,      53.7,      53.7,      53.7,
        53.7,      53.7,      53.7,      53.7,      53.7,      53.7,      53.7,      53.7,
        56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002,
        56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002, 56.100002,
        58.5,      58.5,      58.5,      58.5,      58.5,      58.5,      58.5,      58.5,
        58.5,      58.5,      58.5,      58.5,      58.5,      58.5,      58.5,      58.5,
        60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998,
        60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998, 60.899998,
        63.3,      63.3,      63.3,      63.3,      63.3,      63.3,      63.3,      63.3,
        63.3,      63.3,      63.3,      63.3,      63.3,      63.3,      63.3,      63.3,
        65.7,      65.7,      65.7,      65.7,      65.7,      65.7,      65.7,      65.7,
        65.7,      65.7,      65.7,      65.7,      65.7,      65.7,      65.7,      65.7,
        68.1,      68.1,      68.1,      68.1,      68.1,      68.1,      68.1,      68.1,
        68.1,      68.1,      68.1,      68.1,      68.1,      68.1,      68.1,      68.1,
        70.5,      70.5,      70.5,      70.5,      70.5,      70.5,      70.5,      70.5,
        70.5,      70.5,      70.5,      70.5,      70.5,      70.5,      70.5,      70.5,
        72.9,      72.9,      72.9,      72.9,      72.9,      72.9,      72.9,      72.9,
        72.9,      72.9,      72.9,      72.9,      72.9,      72.9,      72.9,      72.9,
        49.4,      49.4,      49.4,      49.4,      49.4,      49.4,      49.4,      49.4,
        49.4,      49.4,      49.4,      49.4,      49.4,      49.4,      49.4,      49.4};
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor2.mDesc.GetLengths(), ref_dims, "Error: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(out_tensor2.mData, ref_data, "Error: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv3DNCDHW)
{
    ck::utils::conv::ConvParams params;
    params.num_dim_spatial_        = 3;
    params.N_                      = 1;
    params.K_                      = 1;
    params.C_                      = 2;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{6, 6, 6};
    params.conv_filter_strides_    = std::vector<ck::index_t>{1, 1, 1};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{0, 0, 0};
    params.input_right_pads_       = std::vector<ck::index_t>{0, 0, 0};

    auto out_tensor = run_reference_convolution_forward<3,
                                                        float,
                                                        float,
                                                        float,
                                                        ck::tensor_layout::convolution::NCDHW,
                                                        ck::tensor_layout::convolution::KCZYX,
                                                        ck::tensor_layout::convolution::NKDHW>(
        params, ck::utils::FillMonotonicSeq<float>{0.f, 0.1f});
    std::vector<std::size_t> ref_dims{1, 1, 4, 4, 4};
    std::vector<float> ref_data{
        407.7,     410.40002, 413.09998, 415.80002, 423.90002, 426.6,     429.30002, 432.,
        440.1,     442.80002, 445.5,     448.2,     456.30002, 459.,      461.7,     464.40002,
        504.90002, 507.6,     510.30002, 513.,      521.1,     523.8,     526.5,     529.2001,
        537.3,     540.,      542.7001,  545.4,     553.5,     556.2001,  558.9,     561.6,
        602.10004, 604.8,     607.5,     610.2,     618.3,     621.,      623.7,     626.4,
        634.5,     637.2,     639.9,     642.60004, 650.7,     653.4,     656.10004, 658.8,
        699.3,     702.,      704.7,     707.4,     715.5,     718.2,     720.9,     723.60004,
        731.7,     734.4001,  737.10004, 739.8,     747.9001,  750.60004, 753.3,     756.};
    EXPECT_TRUE(ck::utils::check_err(out_tensor.mDesc.GetLengths(),
                                     ref_dims,
                                     "Error [case 1]: wrong output tensor dimensions!"));
    EXPECT_TRUE(
        ck::utils::check_err(out_tensor.mData, ref_data, "Error [case 1]: incorrect results!"));
}

TEST(ReferenceConvolutionFWD, Conv3DNCDHWStridesDilations)
{
    ck::utils::conv::ConvParams params;
    params.num_dim_spatial_        = 3;
    params.N_                      = 1;
    params.K_                      = 2;
    params.C_                      = 2;
    params.filter_spatial_lengths_ = std::vector<ck::index_t>{3, 3, 3};
    params.input_spatial_lengths_  = std::vector<ck::index_t>{12, 12, 12};
    params.conv_filter_strides_    = std::vector<ck::index_t>{3, 3, 3};
    params.conv_filter_dilations_  = std::vector<ck::index_t>{1, 1, 1};
    params.input_left_pads_        = std::vector<ck::index_t>{0, 0, 0};
    params.input_right_pads_       = std::vector<ck::index_t>{0, 0, 0};

    auto out_tensor = run_reference_convolution_forward<3,
                                                        float,
                                                        float,
                                                        float,
                                                        ck::tensor_layout::convolution::NCDHW,
                                                        ck::tensor_layout::convolution::KCZYX,
                                                        ck::tensor_layout::convolution::NKDHW>(
        params, ck::utils::FillMonotonicSeq<float>{0.f, 0.1f});
    std::vector<std::size_t> ref_dims{1, 2, 4, 4, 4};
    std::vector<float> ref_data{
        2756.7002, 2764.7998, 2772.9001, 2781.,     2853.9001, 2862.,     2870.1,    2878.2002,
        2951.1,    2959.2002, 2967.2998, 2975.4001, 3048.2998, 3056.4001, 3064.5,    3072.6,
        3923.1,    3931.2,    3939.2998, 3947.4,    4020.2998, 4028.4001, 4036.5002, 4044.5999,
        4117.5,    4125.6,    4133.7,    4141.8,    4214.7,    4222.8,    4230.9004, 4239.,
        5089.5,    5097.5996, 5105.7,    5113.8,    5186.7,    5194.8,    5202.9,    5211.,
        5283.9004, 5292.,     5300.0996, 5308.2,    5381.0996, 5389.2,    5397.3,    5405.4004,
        6255.9004, 6264.0005, 6272.1,    6280.2,    6353.1,    6361.2,    6369.301,  6377.4,
        6450.301,  6458.4,    6466.5,    6474.6,    6547.5,    6555.6,    6563.699,  6571.801,
        2756.7002, 2764.7998, 2772.9001, 2781.,     2853.9001, 2862.,     2870.1,    2878.2002,
        2951.1,    2959.2002, 2967.2998, 2975.4001, 3048.2998, 3056.4001, 3064.5,    3072.6,
        3923.1,    3931.2,    3939.2998, 3947.4,    4020.2998, 4028.4001, 4036.5002, 4044.5999,
        4117.5,    4125.6,    4133.7,    4141.8,    4214.7,    4222.8,    4230.9004, 4239.,
        5089.5,    5097.5996, 5105.7,    5113.8,    5186.7,    5194.8,    5202.9,    5211.,
        5283.9004, 5292.,     5300.0996, 5308.2,    5381.0996, 5389.2,    5397.3,    5405.4004,
        6255.9004, 6264.0005, 6272.1,    6280.2,    6353.1,    6361.2,    6369.301,  6377.4,
        6450.301,  6458.4,    6466.5,    6474.6,    6547.5,    6555.6,    6563.699,  6571.801};
    EXPECT_TRUE(ck::utils::check_err(out_tensor.mDesc.GetLengths(),
                                     ref_dims,
                                     "Error [case 2]: wrong output tensor dimensions!"));
    EXPECT_TRUE(ck::utils::check_err(
        out_tensor.mData, ref_data, "Error [case 2]: incorrect results!", 1e-4f, 1e-6f));
}
