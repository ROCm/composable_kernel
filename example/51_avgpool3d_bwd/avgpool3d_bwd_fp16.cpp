// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_avgpool3d_bwd_impl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_avgpool_bwd.hpp"

using DOutDataType    = float;
using DInDataType     = float;
using ComputeDataType = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DOutLayout = ck::tensor_layout::convolution::NDHWC;
using DInLayout  = ck::tensor_layout::convolution::NDHWC;

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

bool pool3d_bwd_test(bool do_verification,
                     bool time_kernel,
                     ck::index_t N,
                     ck::index_t C,
                     ck::index_t Di,
                     ck::index_t Hi,
                     ck::index_t Wi,
                     std::vector<ck::index_t> window_lengths,
                     std::vector<ck::index_t> window_strides,
                     std::vector<ck::index_t> window_dilations,
                     std::vector<ck::index_t> dinput_left_pads,
                     std::vector<ck::index_t> dinput_right_pads)
{
    using DevicePoolBwdInstance =
        ck::tensor_operation::device::DeviceAvgPool3dBwdImpl<DOutDataType,
                                                             DInDataType,
                                                             ComputeDataType, // ComputeDataType
                                                             64,              // BlockSize
                                                             64, // ReduceMThreadClusterSize
                                                             1,  // ReduceKThreadClusterSize
                                                             1,  // ReduceMThreadSliceSize
                                                             1,  // ReduceKThreadSliceSize
                                                             1,  // InSrcOutDstVectorSize
                                                             true>;

    auto OutSpatialLength = [&](auto InSpatialLength, int index) {
        ck::index_t left_pad   = dinput_left_pads[index];
        ck::index_t right_pad  = dinput_right_pads[index];
        ck::index_t window_len = window_lengths[index];
        ck::index_t stride     = window_strides[index];
        return (InSpatialLength + left_pad + right_pad - window_len) / stride + 1;
    };

    ck::index_t Do = OutSpatialLength(Di, 0);
    ck::index_t Ho = OutSpatialLength(Hi, 1);
    ck::index_t Wo = OutSpatialLength(Wi, 1);

    Tensor<DOutDataType> dout(f_host_tensor_descriptor(N, C, Do, Ho, Wo, DOutLayout{}));
    Tensor<DInDataType> din_dev(f_host_tensor_descriptor(N, C, Di, Hi, Wi, DInLayout{}));
    Tensor<DInDataType> din_host(f_host_tensor_descriptor(N, C, Di, Hi, Wi, DInLayout{}));

    std::cout << "dout: " << dout.mDesc << std::endl;
    std::cout << "din_host: " << din_host.mDesc << std::endl;

    dout.GenerateTensorValue(GeneratorTensor_3<DOutDataType>{0.0, 1.0});

    DeviceMem dout_device_buf(sizeof(DOutDataType) * dout.mDesc.GetElementSpaceSize());
    DeviceMem din_device_buf(sizeof(DInDataType) * din_dev.mDesc.GetElementSpaceSize());

    dout_device_buf.ToDevice(dout.mData.data());
    din_device_buf.SetZero();

    auto pool        = DevicePoolBwdInstance{};
    auto invoker_ptr = pool.MakeInvokerPointer();
    auto argument_ptr =
        pool.MakeArgumentPointer(static_cast<DOutDataType*>(dout_device_buf.GetDeviceBuffer()),
                                 static_cast<DInDataType*>(din_device_buf.GetDeviceBuffer()),
                                 {N, C, Do, Ho, Wo},
                                 {N, C, Di, Hi, Wi},
                                 f_tensor_strides_ncdhw(N, C, Do, Ho, Wo, DOutLayout{}),
                                 f_tensor_strides_ncdhw(N, C, Di, Hi, Wi, DInLayout{}),
                                 window_lengths,
                                 window_strides,
                                 window_dilations,
                                 dinput_left_pads,
                                 dinput_right_pads);

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
        auto ref_pool =
            ck::tensor_operation::host::ReferenceAvgPoolBwd<3, DInDataType, DOutDataType>();

        auto ref_invoker = ref_pool.MakeInvoker();

        auto ref_argument = ref_pool.MakeArgument(din_host,
                                                  dout,
                                                  window_lengths,
                                                  window_strides,
                                                  window_dilations,
                                                  dinput_left_pads,
                                                  dinput_right_pads);

        ref_invoker.Run(ref_argument);

        din_device_buf.FromDevice(din_dev.mData.data());
        pass = ck::utils::check_err(din_dev, din_host);
    }

    return pass;
}

int main()
{
    std::vector<ck::index_t> window_lengths    = {5, 5, 5};
    std::vector<ck::index_t> window_strides    = {2, 2, 2};
    std::vector<ck::index_t> window_dilations  = {2, 2, 2};
    std::vector<ck::index_t> dinput_left_pads  = {0, 0, 0};
    std::vector<ck::index_t> dinput_right_pads = {0, 0, 0};

    ck::index_t N  = 1;
    ck::index_t C  = 16;
    ck::index_t Di = 40;
    ck::index_t Hi = 40;
    ck::index_t Wi = 40;

    pool3d_bwd_test(true,
                    false,
                    N,
                    C,
                    Di,
                    Hi,
                    Wi,
                    window_lengths,
                    window_strides,
                    window_dilations,
                    dinput_left_pads,
                    dinput_right_pads);
}
