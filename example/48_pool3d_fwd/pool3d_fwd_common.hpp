// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool3d_fwd_ndhwc_ndhwc.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_pool_fwd.hpp"

template <typename TensorLayout>
std::vector<ck::index_t> f_tensor_strides_ncdhw(ck::index_t N_,
                                                ck::index_t C_,
                                                ck::index_t D,
                                                ck::index_t H,
                                                ck::index_t W,
                                                TensorLayout layout)
{
    using namespace ck::literals;
    (void)N_;
    if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NCDHW>::value)
        return {C_ * D * H * W, D * H * W, H * W, W, 1_uz};
    else if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NDHWC>::value)
        return {D * C_ * H * W, 1_uz, C_ * H * W, W * C_, C_};
};

template <typename TensorLayout>
HostTensorDescriptor f_host_tensor_descriptor(std::size_t N_,
                                              std::size_t C_,
                                              std::size_t D,
                                              std::size_t H,
                                              std::size_t W,
                                              TensorLayout layout)
{
    using namespace ck::literals;

    if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NCDHW>::value)
    {
        return HostTensorDescriptor({N_, C_, D, H, W}, {C_ * D * H * W, D * H * W, H * W, W, 1_uz});
    }
    else if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NDHWC>::value)
    {
        return HostTensorDescriptor({N_, C_, D, H, W},
                                    {D * C_ * H * W, 1_uz, C_ * H * W, W * C_, C_});
    }
};

template <typename DevicePoolFwdInstance,
          typename InDataType,
          typename OutDataType,
          typename ComputeDataType,
          typename IndexDataType,
          typename InLayout,
          typename OutLayout,
          ck::ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool OutputIndex>
bool pool3d_test(bool do_verification,
                 bool time_kernel,
                 ck::index_t N,
                 ck::index_t C,
                 ck::index_t Z,
                 ck::index_t Y,
                 ck::index_t X,
                 ck::index_t Di,
                 ck::index_t Hi,
                 ck::index_t Wi,
                 ck::index_t window_stride_d,
                 ck::index_t window_stride_h,
                 ck::index_t window_stride_w,
                 ck::index_t window_dilation_d,
                 ck::index_t window_dilation_h,
                 ck::index_t window_dilation_w,
                 ck::index_t in_left_pad_d,
                 ck::index_t in_left_pad_h,
                 ck::index_t in_left_pad_w,
                 ck::index_t in_right_pad_d,
                 ck::index_t in_right_pad_h,
                 ck::index_t in_right_pad_w)
{
    const ck::index_t Zs = (Z - 1) * window_dilation_d + 1;
    const ck::index_t Ys = (Y - 1) * window_dilation_h + 1;
    const ck::index_t Xs = (X - 1) * window_dilation_w + 1;
    const ck::index_t Do = (Di + in_left_pad_d + in_right_pad_d - Zs) / window_stride_d + 1;
    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - Ys) / window_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - Xs) / window_stride_w + 1;

    const std::vector<ck::index_t> window_spatial_lengths{Z, Y, X};
    const std::vector<ck::index_t> window_strides{
        window_stride_d, window_stride_h, window_stride_w};
    const std::vector<ck::index_t> window_dilations{
        window_dilation_d, window_dilation_h, window_dilation_w};
    const std::vector<ck::index_t> input_left_pads{in_left_pad_d, in_left_pad_h, in_left_pad_w};
    const std::vector<ck::index_t> input_right_pads{in_right_pad_d, in_right_pad_h, in_right_pad_w};

    Tensor<InDataType> in_n_c_di_hi_wi(f_host_tensor_descriptor(N, C, Di, Hi, Wi, InLayout{}));
    Tensor<OutDataType> out_n_c_do_ho_wo_host(
        f_host_tensor_descriptor(N, C, Do, Ho, Wo, OutLayout{}));
    Tensor<IndexDataType> out_indices_n_c_do_ho_wo_host(
        f_host_tensor_descriptor(N, C, Do, Ho, Wo, OutLayout{}));
    Tensor<OutDataType> out_n_c_do_ho_wo_device(
        f_host_tensor_descriptor(N, C, Do, Ho, Wo, OutLayout{}));
    Tensor<IndexDataType> out_indices_n_c_do_ho_wo_device(
        f_host_tensor_descriptor(N, C, Do, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_di_hi_wi: " << in_n_c_di_hi_wi.mDesc << std::endl;
    std::cout << "out_n_c_do_ho_wo: " << out_n_c_do_ho_wo_host.mDesc << std::endl;

    in_n_c_di_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_di_hi_wi.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_c_do_ho_wo_device.mDesc.GetElementSpaceSize());
    DeviceMem out_indices_device_buf(sizeof(IndexDataType) *
                                     out_indices_n_c_do_ho_wo_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in_n_c_di_hi_wi.mData.data());

    auto pool         = DevicePoolFwdInstance{};
    auto invoker_ptr  = pool.MakeInvokerPointer();
    auto argument_ptr = pool.MakeArgumentPointer(
        static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
        static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
        static_cast<IndexDataType*>(out_indices_device_buf.GetDeviceBuffer()),
        {N, C, Di, Hi, Wi},
        {Z, Y, X},
        {N, C, Do, Ho, Wo},
        f_tensor_strides_ncdhw(N, C, Di, Hi, Wi, InLayout{}),
        f_tensor_strides_ncdhw(N, C, Do, Ho, Wo, OutLayout{}),
        f_tensor_strides_ncdhw(N, C, Do, Ho, Wo, OutLayout{}),
        window_strides,
        window_dilations,
        input_left_pads,
        input_right_pads,
        {2, 3, 4});

    if(!pool.IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error("wrong! device_op with the specified compilation parameters does "
                                 "not support this problem");
    }

    float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});
    std::cout << "Perf: " << ave_time << std::endl;

    bool pass = true;

    if(do_verification)
    {
        using ReferencePoolingFwdInstance =
            ck::tensor_operation::host::ReferencePoolingFwd<5,
                                                            3,
                                                            InDataType,
                                                            OutDataType,
                                                            ComputeDataType,
                                                            IndexDataType,
                                                            ReduceOpId,
                                                            PropagateNan,
                                                            OutputIndex>;

        auto ref_pooling          = ReferencePoolingFwdInstance{};
        auto ref_pooling_invoker  = ref_pooling.MakeInvoker();
        auto ref_pooling_argument = ref_pooling.MakeArgument(in_n_c_di_hi_wi,
                                                             out_n_c_do_ho_wo_host,
                                                             out_indices_n_c_do_ho_wo_host,
                                                             window_spatial_lengths,
                                                             window_strides,
                                                             window_dilations,
                                                             input_left_pads,
                                                             input_right_pads);

        ref_pooling_invoker.Run(ref_pooling_argument);

        out_device_buf.FromDevice(out_n_c_do_ho_wo_device.mData.data());

        pass = pass && ck::utils::check_err(out_n_c_do_ho_wo_device, out_n_c_do_ho_wo_host);

        if constexpr(OutputIndex)
        {
            out_indices_device_buf.FromDevice(out_indices_n_c_do_ho_wo_device.mData.data());

            pass = pass && ck::utils::check_err(out_indices_n_c_do_ho_wo_device,
                                                out_indices_n_c_do_ho_wo_host);
        };
    }

    return (pass);
};
