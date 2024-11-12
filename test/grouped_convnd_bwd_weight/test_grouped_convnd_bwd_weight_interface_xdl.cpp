// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/device_memory.hpp"

#include <gtest/gtest.h>

namespace ctl = ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using InDataType  = ck::bhalf_t;
using WeiDataType = float;
using OutDataType = ck::bhalf_t;
using AccDataType = float;
template <ck::index_t... Is>

using S = ck::Sequence<Is...>;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename InputLay, typename WeightLay, typename OutputLay>
struct CommonLayoutSetting
{
    using InputLayout  = InputLay;
    using WeightLayout = WeightLay;
    using OutputLayout = OutputLay;
};

template <ck::index_t NDimSpatial>
struct CommonLayoutSettingSelector
    : CommonLayoutSetting<ck::tuple_element_t<NDimSpatial - 1,
                                              ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                                        ck::tensor_layout::convolution::GNHWC,
                                                        ck::tensor_layout::convolution::GNDHWC>>,
                          ck::tuple_element_t<NDimSpatial - 1,
                                              ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                                        ck::tensor_layout::convolution::GKYXC,
                                                        ck::tensor_layout::convolution::GKZYXC>>,
                          ck::tuple_element_t<NDimSpatial - 1,
                                              ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                                        ck::tensor_layout::convolution::GNHWK,
                                                        ck::tensor_layout::convolution::GNDHWK>>>
{
};

template <ck::index_t NDimSpatial>
using InputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::InputLayout;

template <ck::index_t NDimSpatial>
using WeightLayout = typename CommonLayoutSettingSelector<NDimSpatial>::WeightLayout;

template <ck::index_t NDimSpatial>
using OutputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::OutputLayout;

class TestGroupedConvndBwdWeight : public ::testing::Test
{
    protected:
    ck::utils::conv::ConvParam conv_param;

    template <ck::index_t NDimSpatial>
    void RunReference(Tensor<InDataType>& in,
                      Tensor<WeiDataType>& wei_host_result,
                      Tensor<OutDataType>& out)
    {
        auto ref_conv     = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                           InDataType,
                                                                           WeiDataType,
                                                                           OutDataType,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           PassThrough>{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei_host_result,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{},
                                                  {},
                                                  {},
                                                  {});

        ref_invoker.Run(ref_argument);
    }

    template <ck::index_t NDimSpatial>
    bool PerformConvWeight(ck::index_t split_k)
    {
        bool passed{true};

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
                InputLayout<NDimSpatial>>(conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
                WeightLayout<NDimSpatial>>(conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
                OutputLayout<NDimSpatial>>(conv_param);

        Tensor<InDataType> in(in_g_n_c_wis_desc);
        Tensor<WeiDataType> wei_host_result(wei_g_k_c_xs_desc);
        Tensor<WeiDataType> wei_device_result(wei_g_k_c_xs_desc);
        Tensor<OutDataType> out(out_g_n_k_wos_desc);

        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});

        DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
        DeviceMem wei_device_buf(sizeof(WeiDataType) *
                                 wei_device_result.mDesc.GetElementSpaceSize());
        DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

        in_device_buf.ToDevice(in.mData.data());
        out_device_buf.ToDevice(out.mData.data());

        // init to 0
        wei_device_buf.SetZero();

        std::array<ck::index_t, NDimSpatial + 3> input_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> input_strides{};
        std::array<ck::index_t, NDimSpatial + 3> filter_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> weights_strides{};
        std::array<ck::index_t, NDimSpatial + 3> output_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> output_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto range_copy = [](const auto& from, auto to) { std::copy(begin(from), end(from), to); };

        range_copy(in_g_n_c_wis_desc.GetLengths(), begin(input_lengths));
        range_copy(in_g_n_c_wis_desc.GetStrides(), begin(input_strides));
        range_copy(wei_g_k_c_xs_desc.GetLengths(), begin(filter_lengths));
        range_copy(wei_g_k_c_xs_desc.GetStrides(), begin(weights_strides));
        range_copy(out_g_n_k_wos_desc.GetLengths(), begin(output_lengths));
        range_copy(out_g_n_k_wos_desc.GetStrides(), begin(output_strides));
        range_copy(conv_param.conv_filter_strides_, begin(conv_filter_strides));
        range_copy(conv_param.conv_filter_dilations_, begin(conv_filter_dilations));
        range_copy(conv_param.input_left_pads_, begin(input_left_pads));
        range_copy(conv_param.input_right_pads_, begin(input_right_pads));

        RunReference<NDimSpatial>(in, wei_host_result, out);

        using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffle<
            NDimSpatial,
            ck::tuple_element_t<NDimSpatial - 1,
                                ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                          ck::tensor_layout::convolution::GNHWC,
                                          ck::tensor_layout::convolution::GNDHWC>>,
            ck::tuple_element_t<NDimSpatial - 1,
                                ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                          ck::tensor_layout::convolution::GKYXC,
                                          ck::tensor_layout::convolution::GKZYXC>>,
            ck::tuple_element_t<NDimSpatial - 1,
                                ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                          ck::tensor_layout::convolution::GNHWK,
                                          ck::tensor_layout::convolution::GNDHWK>>,
            InDataType,           // InDataType
            WeiDataType,          // WeiDataType
            OutDataType,          // OutDataType
            AccDataType,          // AccDataType
            PassThrough,          // InElementwiseOperation
            PassThrough,          // WeiElementwiseOperation
            PassThrough,          // OutElementwiseOperation
            ConvBwdWeightDefault, // ConvolutionBackwardWeightSpecialization
            64,                   // BlockSize
            16,                   // MPerBlock
            16,                   // NPerBlock
            32,                   // K0PerBlock
            8,                    // K1
            16,                   // MPerXdl
            16,                   // NPerXdl
            1,                    // MXdlPerWave
            1,                    // NXdlPerWave
            S<4, 16, 1>,          // ABlockTransferThreadClusterLengths_K0_M_K1
            S<2, 0, 1>,           // ABlockTransferThreadClusterArrangeOrder
            S<1, 0, 2>,           // ABlockTransferSrcAccessOrder
            1,                    // ABlockTransferSrcVectorDim
            1,                    // ABlockTransferSrcScalarPerVector
            4,                    // ABlockTransferDstScalarPerVector_K1
            false,                // ABlockLdsAddExtraM
            S<4, 16, 1>,          // BBlockTransferThreadClusterLengths_K0_N_K1
            S<2, 0, 1>,           // BBlockTransferThreadClusterArrangeOrder
            S<1, 0, 2>,           // BBlockTransferSrcAccessOrder
            1,                    // BBlockTransferSrcVectorDim
            1,                    // BBlockTransferSrcScalarPerVector
            4,                    // BBlockTransferDstScalarPerVector_K1
            false,                // BBlockLdsAddExtraN
            1,                    // CShuffleMXdlPerWavePerShuffle
            1,                    // CShuffleNXdlPerWavePerShuffle
            S<1, 8, 1, 8>,        // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
            1>;                   // CBlockTransferScalarPerVector_NWaveNPerXdl

        auto conv_ptr = DeviceOp{};
        auto argument =
            conv_ptr.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                  static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                  static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                  input_lengths,
                                  input_strides,
                                  filter_lengths,
                                  weights_strides,
                                  output_lengths,
                                  output_strides,
                                  conv_filter_strides,
                                  conv_filter_dilations,
                                  input_left_pads,
                                  input_right_pads,
                                  PassThrough{},
                                  PassThrough{},
                                  PassThrough{},
                                  split_k);

        auto invoker_ptr = conv_ptr.MakeInvoker();

        if(conv_ptr.IsSupportedArgument(argument))
        {
            float avg_time = invoker_ptr.Run(argument, StreamConfig{nullptr, false});
            wei_device_buf.FromDevice(wei_device_result.mData.data());
            passed &= ck::utils::check_err(
                wei_device_result.mData, wei_host_result.mData, "Error: incorrect results!");

            std::size_t flop = conv_param.GetFlops() +
                               3 * conv_param.GetOutputByte<WeiDataType>() / sizeof(WeiDataType);
            std::size_t num_bytes = conv_param.GetByte<InDataType, WeiDataType, OutDataType>() +
                                    conv_param.GetOutputByte<WeiDataType>();

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, "
                      << "split_k " << split_k << std::endl;
        }
        return passed;
    }

    template <ck::index_t NDimSpatial>
    void Run()
    {
        bool pass = true;

        for(auto split_k : {1, 2})
        {
            pass = pass && PerformConvWeight<NDimSpatial>(split_k);
            EXPECT_TRUE(pass);
        }
    }
};

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_1_Filter_1x1)
{
    this->conv_param = {
        1, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<1>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_1_Filter_3x3)
{
    this->conv_param = {
        1, 2, 4, 192, 192, {3, 3, 3}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<1>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_2_Filter_1x1)
{
    this->conv_param = {
        2, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<2>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_2_Filter_3x3)
{
    this->conv_param = {
        2, 2, 4, 192, 192, {3, 3, 3}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<2>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_3_Filter_1x1)
{
    this->conv_param = {
        3, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<3>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_3_Filter_3x3)
{
    this->conv_param = {
        3, 2, 4, 192, 192, {3, 3, 3}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<3>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_1_Stride_1x1)
{
    this->conv_param = {
        1, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<1>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_1_Stride_2x2)
{
    this->conv_param = {
        1, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<1>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_2_Stride_1x1)
{
    this->conv_param = {
        2, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<2>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_2_Stride_2x2)
{
    this->conv_param = {
        2, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<2>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_3_Stride_1x1)
{
    this->conv_param = {
        3, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<3>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_3_Stride_2x2)
{
    this->conv_param = {
        3, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    this->template Run<3>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_1_WithPadding)
{
    this->conv_param = {
        1, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->template Run<1>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_2_WithPadding)
{
    this->conv_param = {
        2, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->template Run<2>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_3_WithPadding)
{
    this->conv_param = {
        3, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->template Run<3>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_1_SupportedVersion)
{
    this->conv_param = {
        1, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->template Run<1>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_2_SupportedVersion)
{
    this->conv_param = {
        2, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->template Run<2>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_3_SupportedVersion)
{
    this->conv_param = {
        3, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    this->template Run<3>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_1_VectorLoadForA)
{
    this->conv_param = {1, 2, 128, 129, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    this->template Run<1>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_2_VectorLoadForA)
{
    this->conv_param = {2, 2, 128, 129, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    this->template Run<2>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_1_VectorLoadForB_E_DS)
{
    this->conv_param = {1, 2, 128, 128, 257, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    this->template Run<1>();
}

TEST_F(TestGroupedConvndBwdWeight, TestGroupedConvndBwdWeight_NDimSpatial_2_VectorLoadForB_E_DS)
{
    this->conv_param = {2, 2, 128, 128, 257, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}};
    this->template Run<2>();
}
