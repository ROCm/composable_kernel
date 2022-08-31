// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

template <ck::index_t NDimSpatial,
          typename In0DataType,
          typename Wei0DataType,
          typename Acc0DataType,
          typename Wei1DataType,
          typename Out1DataType,
          typename In0ElementOp,
          typename Wei0ElementOp,
          typename Out0ElementOp,
          typename Wei1ElementOp,
          typename Out1ElementOp,
          typename DeviceOpInstance>
int run_grouped_conv_conv_fwd(bool do_verification,
                              int init_method,
                              bool time_kernel,
                              const ck::utils::conv::ConvParam& conv0_param,
                              const ck::utils::conv::ConvParam& conv1_param,
                              const HostTensorDescriptor& in0_g_n_c_wis_desc,
                              const HostTensorDescriptor& wei0_g_k_c_xs_desc,
                              const HostTensorDescriptor& out0_g_n_k_wos_desc,
                              const HostTensorDescriptor& wei1_g_k_c_xs_desc,
                              const HostTensorDescriptor& out1_g_n_k_wos_desc,
                              const In0ElementOp& in0_element_op,
                              const Wei0ElementOp& wei0_element_op,
                              const Wei1ElementOp& wei1_element_op,
                              const Out0ElementOp& out0_element_op,
                              const Out1ElementOp& out1_element_op)
{
    Tensor<In0DataType> in0(in0_g_n_c_wis_desc);
    Tensor<Wei0DataType> wei0(wei0_g_k_c_xs_desc);
    Tensor<Wei1DataType> wei1(wei1_g_k_c_xs_desc);
    Tensor<Out1DataType> out1_host(out1_g_n_k_wos_desc);
    Tensor<Out1DataType> out1_device(out1_g_n_k_wos_desc);

    std::cout << "in0: " << in0.mDesc << std::endl;
    std::cout << "wei0: " << wei0.mDesc << std::endl;
    std::cout << "wei1: " << wei1.mDesc << std::endl;
    std::cout << "out1: " << out1_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in0.GenerateTensorValue(GeneratorTensor_2<In0DataType>{-5, 5});
        wei0.GenerateTensorValue(GeneratorTensor_2<Wei0DataType>{-5, 5});
        wei1.GenerateTensorValue(GeneratorTensor_2<Wei1DataType>{-5, 5});
        break;
    default:
        in0.GenerateTensorValue(GeneratorTensor_3<In0DataType>{0.0, 1.0});
        wei0.GenerateTensorValue(GeneratorTensor_3<Wei0DataType>{-0.5, 0.5});
        wei1.GenerateTensorValue(GeneratorTensor_3<Wei1DataType>{-0.5, 0.5});
    }

    DeviceMem in0_device_buf(sizeof(In0DataType) * in0.mDesc.GetElementSpaceSize());
    DeviceMem wei0_device_buf(sizeof(Wei0DataType) * wei0.mDesc.GetElementSpaceSize());
    DeviceMem wei1_device_buf(sizeof(Wei1DataType) * wei1.mDesc.GetElementSpaceSize());
    DeviceMem out1_device_buf(sizeof(Out1DataType) * out1_device.mDesc.GetElementSpaceSize());

    in0_device_buf.ToDevice(in0.mData.data());
    wei0_device_buf.ToDevice(wei0.mData.data());
    wei1_device_buf.ToDevice(wei1.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a0_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a0_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b0_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b0_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b1_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b1_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e1_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e1_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv0_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv0_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input0_left_pads{};
    std::array<ck::index_t, NDimSpatial> input0_right_pads{};
    std::array<ck::index_t, NDimSpatial> conv1_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv1_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input1_left_pads{};
    std::array<ck::index_t, NDimSpatial> input1_right_pads{};

    auto copy = [](auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };

    copy(in0_g_n_c_wis_desc.GetLengths(), a0_g_n_c_wis_lengths);
    copy(in0_g_n_c_wis_desc.GetStrides(), a0_g_n_c_wis_strides);
    copy(wei0_g_k_c_xs_desc.GetLengths(), b0_g_k_c_xs_lengths);
    copy(wei0_g_k_c_xs_desc.GetStrides(), b0_g_k_c_xs_strides);
    copy(wei1_g_k_c_xs_desc.GetLengths(), b1_g_k_c_xs_lengths);
    copy(wei1_g_k_c_xs_desc.GetStrides(), b1_g_k_c_xs_strides);
    copy(out1_g_n_k_wos_desc.GetLengths(), e1_g_n_k_wos_lengths);
    copy(out1_g_n_k_wos_desc.GetStrides(), e1_g_n_k_wos_strides);
    copy(conv0_param.conv_filter_strides_, conv0_filter_strides);
    copy(conv0_param.conv_filter_dilations_, conv0_filter_dilations);
    copy(conv0_param.input_left_pads_, input0_left_pads);
    copy(conv0_param.input_right_pads_, input0_right_pads);
    copy(conv1_param.conv_filter_strides_, conv1_filter_strides);
    copy(conv1_param.conv_filter_dilations_, conv1_filter_dilations);
    copy(conv1_param.input_left_pads_, input1_left_pads);
    copy(conv1_param.input_right_pads_, input1_right_pads);

#if 1
    // do Conv using GEMM, only works for 1x1 conv for now
    const ck::index_t gemm_batch = a0_g_n_c_wis_lengths[0];

    const ck::index_t gemm0_m_length =
        e1_g_n_k_wos_lengths[1] * std::accumulate(e1_g_n_k_wos_lengths.begin() + 3,
                                                  e1_g_n_k_wos_lengths.begin() + 3 + NDimSpatial,
                                                  ck::index_t{1},
                                                  std::multiplies<ck::index_t>{});

    const ck::index_t gemm0_n_length = b0_g_k_c_xs_lengths[1];

    const ck::index_t gemm0_k_length =
        std::accumulate(b0_g_k_c_xs_lengths.begin() + 2,
                        b0_g_k_c_xs_lengths.begin() + 2 + NDimSpatial + 1,
                        ck::index_t{1},
                        std::multiplies<ck::index_t>{});

    const ck::index_t gemm1_n_length = b1_g_k_c_xs_lengths[1];

    //
    const ck::index_t a0_stride = a0_g_n_c_wis_strides[2 + NDimSpatial];
    const ck::index_t b0_stride = b0_g_k_c_xs_strides[2 + NDimSpatial];
    const ck::index_t b1_stride = b1_g_k_c_xs_strides[2 + NDimSpatial];
    const ck::index_t e1_stride = e1_g_n_k_wos_strides[2 + NDimSpatial];

    //
    const ck::index_t a0_batch_stride = a0_g_n_c_wis_strides[0];
    const ck::index_t b0_batch_stride = b0_g_k_c_xs_strides[0];
    const ck::index_t b1_batch_stride = b1_g_k_c_xs_strides[0];
    const ck::index_t e1_batch_stride = e1_g_n_k_wos_strides[0];

    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(static_cast<In0DataType*>(in0_device_buf.GetDeviceBuffer()),
                               static_cast<Wei0DataType*>(wei0_device_buf.GetDeviceBuffer()),
                               static_cast<Wei1DataType*>(wei1_device_buf.GetDeviceBuffer()),
                               static_cast<Out1DataType*>(out1_device_buf.GetDeviceBuffer()),
                               gemm0_m_length,
                               gemm0_n_length,
                               gemm0_k_length,
                               gemm1_n_length,
                               gemm_batch,
                               a0_stride,
                               b0_stride,
                               b1_stride,
                               e1_stride,
                               a0_batch_stride,
                               b0_batch_stride,
                               b1_batch_stride,
                               e1_batch_stride,
                               in0_element_op,
                               wei0_element_op,
                               out0_element_op,
                               wei1_element_op,
                               out1_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv0_param.GetFlops() + conv1_param.GetFlops();
    std::size_t num_btype = conv0_param.template GetInputByte<In0DataType>() +
                            conv0_param.template GetWeightByte<Wei0DataType>() +
                            conv1_param.template GetWeightByte<Wei1DataType>() +
                            conv1_param.template GetOutputByte<Out1DataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << device_op.GetTypeString() << std::endl;
#endif

    if(do_verification)
    {
        using PassThrough = ck::tensor_operation::element_wise::PassThrough;

        Tensor<Acc0DataType> out0_host(out0_g_n_k_wos_desc);

        auto ref_conv0 = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                      In0DataType,
                                                                      Wei0DataType,
                                                                      Acc0DataType,
                                                                      In0ElementOp,
                                                                      Wei0ElementOp,
                                                                      Out0ElementOp>();

        auto ref_conv1 = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                      Acc0DataType,
                                                                      Wei1DataType,
                                                                      Out1DataType,
                                                                      PassThrough,
                                                                      Wei1ElementOp,
                                                                      Out1ElementOp>();

        auto ref_conv0_invoker = ref_conv0.MakeInvoker();
        auto ref_conv1_invoker = ref_conv1.MakeInvoker();

        auto ref_conv0_argument = ref_conv0.MakeArgument(in0,
                                                         wei0,
                                                         out0_host,
                                                         conv0_param.conv_filter_strides_,
                                                         conv0_param.conv_filter_dilations_,
                                                         conv0_param.input_left_pads_,
                                                         conv0_param.input_right_pads_,
                                                         in0_element_op,
                                                         wei0_element_op,
                                                         out0_element_op);

        auto ref_conv1_argument = ref_conv1.MakeArgument(out0_host,
                                                         wei1,
                                                         out1_host,
                                                         conv1_param.conv_filter_strides_,
                                                         conv1_param.conv_filter_dilations_,
                                                         conv1_param.input_left_pads_,
                                                         conv1_param.input_right_pads_,
                                                         out0_element_op,
                                                         wei1_element_op,
                                                         out1_element_op);

        ref_conv0_invoker.Run(ref_conv0_argument);
        ref_conv1_invoker.Run(ref_conv1_argument);

        out1_device_buf.FromDevice(out1_device.mData.data());

        return ck::utils::check_err(
                   out1_device.mData, out1_host.mData, "Error: incorrect results!", 1e-5f, 1e-4f)
                   ? 0
                   : 1;
    }

    return 0;
}
