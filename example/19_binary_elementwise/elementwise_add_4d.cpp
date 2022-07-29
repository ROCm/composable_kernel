// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_binary_elementwise.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ABDataType             = F16;
using CDataType              = F16;
using EltwiseComputeDataType = F32;

using Add = ck::tensor_operation::element_wise::Add;

using DeviceElementwiseAddInstance =
    ck::tensor_operation::device::DeviceBinaryElementwise<ABDataType,
                                                          ABDataType,
                                                          CDataType,
                                                          EltwiseComputeDataType,
                                                          Add,
                                                          4,
                                                          8,
                                                          8,
                                                          8,
                                                          8>;

template <typename HostTensorA,
          typename HostTensorB,
          typename HostTensorC,
          typename ComputeDataType,
          typename Functor>
void host_elementwise4D(HostTensorC& C,
                        const HostTensorA& A,
                        const HostTensorB& B,
                        const std::vector<std::size_t>& shape,
                        Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0, 0, 0, 0))>;

    for(std::size_t n = 0; n < shape[0]; ++n)
        for(std::size_t c = 0; c < shape[1]; ++c)
            for(std::size_t h = 0; h < shape[2]; ++h)
                for(std::size_t w = 0; w < shape[3]; ++w)
                {
                    ComputeDataType a_val = ck::type_convert<ComputeDataType>(A(n, c, h, w));
                    ComputeDataType b_val = ck::type_convert<ComputeDataType>(B(n, c, h, w));
                    ComputeDataType c_val = 0;
                    functor(c_val, a_val, b_val);
                    C(n, c, h, w) = ck::type_convert<ctype>(c_val);
                }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    std::vector<std::size_t> nchw = {4, 16, 32, 32};

    Tensor<ABDataType> a(nchw);
    Tensor<ABDataType> b(nchw);
    Tensor<CDataType> c(nchw);

    a.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});
    b.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});

    DeviceMem a_device_buf(sizeof(ABDataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(ABDataType) * b.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());
    b_device_buf.ToDevice(b.mData.data());

    std::array<const void*, 2> input = {a_device_buf.GetDeviceBuffer(),
                                        b_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {c_device_buf.GetDeviceBuffer()};

    std::vector<ck::index_t> a_strides{a.mDesc.GetStrides().begin(), a.mDesc.GetStrides().end()};
    std::vector<ck::index_t> b_strides{b.mDesc.GetStrides().begin(), b.mDesc.GetStrides().end()};
    std::vector<ck::index_t> c_strides{c.mDesc.GetStrides().begin(), c.mDesc.GetStrides().end()};

    auto broadcastAdd = DeviceElementwiseAddInstance{};
    auto argument =
        broadcastAdd.MakeArgumentPointer(input,
                                         output,
                                         std::vector<ck::index_t>{nchw.begin(), nchw.end()},
                                         {{a_strides}, b_strides},
                                         {c_strides},
                                         Add{});

    if(!broadcastAdd.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error("The runtime parameters seems not supported by the "
                                 "DeviceBinaryElementwise instance, exiting!");
    };

    auto broadcastAdd_invoker_ptr = broadcastAdd.MakeInvokerPointer();
    float ave_time =
        broadcastAdd_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_device_buf.FromDevice(c.mData.data());
        Tensor<CDataType> host_c(nchw);

        host_elementwise4D<Tensor<ABDataType>,
                           Tensor<ABDataType>,
                           Tensor<CDataType>,
                           EltwiseComputeDataType,
                           Add>(host_c, a, b, nchw, Add{});

        pass &=
            ck::utils::check_err(c.mData, host_c.mData, "Error: Incorrect results c", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
