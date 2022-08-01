// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_binary_elementwise.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
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
                                                          1,
                                                          8,
                                                          8,
                                                          8,
                                                          8>;

template <typename HostTensorA,
          typename HostTensorB,
          typename HostTensorC,
          typename ComputeDataType,
          typename Functor>
void host_elementwise1D(
    HostTensorC& C, const HostTensorA& A, const HostTensorB& B, int M, Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0))>;

    for(int m = 0; m < M; ++m)
    {
        ComputeDataType Am = ck::type_convert<ComputeDataType>(A(m));
        ComputeDataType Bm = ck::type_convert<ComputeDataType>(B(m));
        ComputeDataType Cm = 0;
        functor(Cm, Am, Bm);
        C(m) = ck::type_convert<ctype>(Cm);
    }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    ck::index_t M = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor(std::vector<std::size_t>({len}),
                                    std::vector<std::size_t>({stride}));
    };

    Tensor<ABDataType> a_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<ABDataType> b_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<CDataType> c_m(f_host_tensor_descriptor1d(M, 1));

    a_m.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});
    b_m.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});

    DeviceMem a_m_device_buf(sizeof(ABDataType) * a_m.mDesc.GetElementSpaceSize());
    DeviceMem b_m_device_buf(sizeof(ABDataType) * b_m.mDesc.GetElementSpaceSize());
    DeviceMem c_m_device_buf(sizeof(CDataType) * c_m.mDesc.GetElementSpaceSize());

    a_m_device_buf.ToDevice(a_m.mData.data());
    b_m_device_buf.ToDevice(b_m.mData.data());

    std::array<const void*, 2> input = {a_m_device_buf.GetDeviceBuffer(),
                                        b_m_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {c_m_device_buf.GetDeviceBuffer()};

    std::vector<ck::index_t> a_strides = {1};
    std::vector<ck::index_t> b_strides = {1};
    std::vector<ck::index_t> c_strides = {1};

    auto broadcastAdd = DeviceElementwiseAddInstance{};
    auto argument     = broadcastAdd.MakeArgumentPointer(
        input, output, {M}, {{a_strides}, b_strides}, {c_strides}, Add{});

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
        c_m_device_buf.FromDevice(c_m.mData.data());
        Tensor<CDataType> host_c_m(f_host_tensor_descriptor1d(M, 1));

        host_elementwise1D<Tensor<ABDataType>,
                           Tensor<ABDataType>,
                           Tensor<CDataType>,
                           EltwiseComputeDataType,
                           Add>(host_c_m, a_m, b_m, M, Add{});

        pass &= ck::utils::check_err(
            c_m.mData, host_c_m.mData, "Error: Incorrect results c", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
