// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "gpu_reference_utils.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

using namespace ck;
using ck::test::ConvKernelType;

// ==================== D Tensor (Bias) Tests ====================

template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
bool test_conv_gpu_ref_with_bias(const ck::utils::conv::ConvParam& params)
{
    using tensor_operation::element_wise::AddClamp;

    // Create tensor descriptors
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(params);
    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(params);
    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(params);

    // Create tensors
    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    Tensor<OutDataType> output(out_g_n_k_wos_desc);
    Tensor<OutDataType> bias(out_g_n_k_wos_desc); // Same shape as output

    // Allocate device memory
    DeviceMem input_dev(input.mData.size() * sizeof(InDataType));
    DeviceMem weight_dev(weight.mData.size() * sizeof(WeiDataType));
    DeviceMem bias_dev(bias.mData.size() * sizeof(OutDataType));
    DeviceMem output_dev(output.mData.size() * sizeof(OutDataType));

    // Initialize and copy tensors
    test::initialize_and_copy_tensor(input, input_dev);
    test::initialize_and_copy_tensor(weight, weight_dev);
    test::initialize_and_copy_tensor(bias, bias_dev);

    // Test with AddClamp (bias operation with clamping)
    AddClamp out_element_op(0.0f, 6.0f); // Clamp between 0 and 6

    return test::test_conv_fwd_with_d_tensor_impl<NDimSpatial,
                                                  InDataType,
                                                  WeiDataType,
                                                  OutDataType,
                                                  InLayout,
                                                  WeiLayout,
                                                  OutLayout>(
        params, input, weight, bias, input_dev, weight_dev, bias_dev, output_dev, out_element_op);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP16Bias)
{
    auto params = test::conv_test_shapes::get_2d_small();
    bool result = test_conv_gpu_ref_with_bias<2,
                                              half_t,
                                              half_t,
                                              half_t,
                                              tensor_layout::convolution::GNCHW,
                                              tensor_layout::convolution::GKCYX,
                                              tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP32Bias)
{
    auto params = test::conv_test_shapes::get_2d_medium();
    bool result = test_conv_gpu_ref_with_bias<2,
                                              float,
                                              float,
                                              float,
                                              tensor_layout::convolution::GNCHW,
                                              tensor_layout::convolution::GKCYX,
                                              tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwdMultiABD, Conv3DFP32Bias)
{
    auto params = test::conv_test_shapes::get_3d_small();
    bool result = test_conv_gpu_ref_with_bias<3,
                                              float,
                                              float,
                                              float,
                                              tensor_layout::convolution::GNCDHW,
                                              tensor_layout::convolution::GKCZYX,
                                              tensor_layout::convolution::GNKDHW>(params);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP16GroupedG2Bias)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test_conv_gpu_ref_with_bias<2,
                                              half_t,
                                              half_t,
                                              half_t,
                                              tensor_layout::convolution::GNCHW,
                                              tensor_layout::convolution::GKCYX,
                                              tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP32GroupedG4Bias)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g4();
    bool result = test_conv_gpu_ref_with_bias<2,
                                              float,
                                              float,
                                              float,
                                              tensor_layout::convolution::GNCHW,
                                              tensor_layout::convolution::GKCYX,
                                              tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

// ==================== D Tensor (Bilinear) Tests ====================

template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
bool test_conv_gpu_ref_with_bilinear(const ck::utils::conv::ConvParam& params)
{
    using tensor_operation::element_wise::Bilinear;

    // Create tensor descriptors
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(params);
    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(params);
    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(params);

    // Create tensors
    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    Tensor<OutDataType> output(out_g_n_k_wos_desc);
    Tensor<OutDataType> d_tensor(out_g_n_k_wos_desc); // Same shape as output

    // Allocate device memory
    DeviceMem input_dev(input.mData.size() * sizeof(InDataType));
    DeviceMem weight_dev(weight.mData.size() * sizeof(WeiDataType));
    DeviceMem d_dev(d_tensor.mData.size() * sizeof(OutDataType));
    DeviceMem output_dev(output.mData.size() * sizeof(OutDataType));

    // Initialize and copy tensors
    test::initialize_and_copy_tensor(input, input_dev);
    test::initialize_and_copy_tensor(weight, weight_dev);
    test::initialize_and_copy_tensor(d_tensor, d_dev);

    // Test with Bilinear: y = alpha * conv_result + beta * d_tensor
    Bilinear out_element_op(1.5f, 0.5f); // alpha=1.5, beta=0.5

    return test::test_conv_fwd_with_d_tensor_impl<NDimSpatial,
                                                  InDataType,
                                                  WeiDataType,
                                                  OutDataType,
                                                  InLayout,
                                                  WeiLayout,
                                                  OutLayout>(
        params, input, weight, d_tensor, input_dev, weight_dev, d_dev, output_dev, out_element_op);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP16Bilinear)
{
    auto params = test::conv_test_shapes::get_2d_small();
    bool result = test_conv_gpu_ref_with_bilinear<2,
                                                  half_t,
                                                  half_t,
                                                  half_t,
                                                  tensor_layout::convolution::GNCHW,
                                                  tensor_layout::convolution::GKCYX,
                                                  tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP32Bilinear)
{
    auto params = test::conv_test_shapes::get_2d_medium();
    bool result = test_conv_gpu_ref_with_bilinear<2,
                                                  float,
                                                  float,
                                                  float,
                                                  tensor_layout::convolution::GNCHW,
                                                  tensor_layout::convolution::GKCYX,
                                                  tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP16GroupedG2Bilinear)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test_conv_gpu_ref_with_bilinear<2,
                                                  half_t,
                                                  half_t,
                                                  half_t,
                                                  tensor_layout::convolution::GNCHW,
                                                  tensor_layout::convolution::GKCYX,
                                                  tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

// ==================== Multiple A/B (ScaleAdd) Tests ====================

template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
bool test_conv_gpu_ref_with_scaleadd(const ck::utils::conv::ConvParam& params)
{
    using tensor_operation::element_wise::ScaleAdd;

    // Create tensor descriptors
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(params);
    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(params);
    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(params);

    // Create tensors
    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    Tensor<OutDataType> output(out_g_n_k_wos_desc);
    Tensor<InDataType> a_extra(in_g_n_c_wis_desc);  // Extra A tensor (same shape as input)
    Tensor<WeiDataType> b_extra(wei_g_k_c_xs_desc); // Extra B tensor (same shape as weight)

    // Allocate device memory
    DeviceMem input_dev(input.mData.size() * sizeof(InDataType));
    DeviceMem weight_dev(weight.mData.size() * sizeof(WeiDataType));
    DeviceMem a_extra_dev(a_extra.mData.size() * sizeof(InDataType));
    DeviceMem b_extra_dev(b_extra.mData.size() * sizeof(WeiDataType));
    DeviceMem output_dev(output.mData.size() * sizeof(OutDataType));

    // Initialize and copy tensors
    test::initialize_and_copy_tensor(input, input_dev);
    test::initialize_and_copy_tensor(weight, weight_dev);
    test::initialize_and_copy_tensor(a_extra, a_extra_dev);
    test::initialize_and_copy_tensor(b_extra, b_extra_dev);

    // Test with ScaleAdd: in_out = scale * in_0 + in_1, wei_out = scale * wei_0 + wei_1
    ScaleAdd in_element_op(2.0f);  // scale factor for input
    ScaleAdd wei_element_op(1.5f); // scale factor for weight

    return test::test_conv_fwd_with_multi_ab_impl<NDimSpatial,
                                                  InDataType,
                                                  WeiDataType,
                                                  OutDataType,
                                                  InLayout,
                                                  WeiLayout,
                                                  OutLayout>(params,
                                                             input,
                                                             weight,
                                                             a_extra,
                                                             b_extra,
                                                             input_dev,
                                                             weight_dev,
                                                             a_extra_dev,
                                                             b_extra_dev,
                                                             output_dev,
                                                             in_element_op,
                                                             wei_element_op);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP16ScaleAdd)
{
    auto params = test::conv_test_shapes::get_2d_small();
    bool result = test_conv_gpu_ref_with_scaleadd<2,
                                                  half_t,
                                                  half_t,
                                                  half_t,
                                                  tensor_layout::convolution::GNCHW,
                                                  tensor_layout::convolution::GKCYX,
                                                  tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP32ScaleAdd)
{
    auto params = test::conv_test_shapes::get_2d_medium();
    bool result = test_conv_gpu_ref_with_scaleadd<2,
                                                  float,
                                                  float,
                                                  float,
                                                  tensor_layout::convolution::GNCHW,
                                                  tensor_layout::convolution::GKCYX,
                                                  tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvFwdMultiABD, Conv2DFP16GroupedG2ScaleAdd)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test_conv_gpu_ref_with_scaleadd<2,
                                                  half_t,
                                                  half_t,
                                                  half_t,
                                                  tensor_layout::convolution::GNCHW,
                                                  tensor_layout::convolution::GKCYX,
                                                  tensor_layout::convolution::GNKHW>(params);
    EXPECT_TRUE(result);
}
