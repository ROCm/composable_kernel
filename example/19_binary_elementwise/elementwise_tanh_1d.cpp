// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using F32  = float;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;

using ADataType = F16;
using CDataType = F16;

using Tanh = ck::tensor_operation::element_wise::TanH;

using DeviceElementwiseTanhInstance =
    ck::tensor_operation::device::DeviceElementwiseImpl<ck::Tuple<ADataType>,
                                                        ck::Tuple<CDataType>,
                                                        Tanh,
                                                        1,
                                                        64,
                                                        16,
                                                        16,
                                                        2,
                                                        2,
                                                        ck::Sequence<1, 0>,
                                                        ck::Sequence<1>,
                                                        ck::Sequence<1>>;

template <typename HostTensorA, typename HostTensorC, typename Functor>
void host_elementwise1D(HostTensorC& C, const HostTensorA& A, int M, Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0))>;

    for(int m = 0; m < M; ++m)
    {
        auto Am  = A(m);
        ctype Cm = 0;
        functor(Cm, Am);
        C(m) = Cm;
    }
}

int main(int argc, char* argv[])
{
    bool do_verification;
    bool time_kernel;

    if(argc == 1)
    {
        do_verification = true;
        time_kernel     = false;
    }
    else if(argc == 3)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = static_cast<bool>(std::stoi(argv[2]));
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    ck::index_t M = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    Tensor<ADataType> a_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<CDataType> c_m(f_host_tensor_descriptor1d(M, 1));

    a_m.GenerateTensorValue(GeneratorTensor_3<ADataType>{-5, 5});

    DeviceMem a_m_device_buf(sizeof(ADataType) * a_m.mDesc.GetElementSpaceSize());
    DeviceMem c_m_device_buf(sizeof(CDataType) * c_m.mDesc.GetElementSpaceSize());

    a_m_device_buf.ToDevice(a_m.mData.data());

    std::array<const void*, 1> input = {a_m_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {c_m_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 1> abc_lengths = {M};
    std::array<ck::index_t, 1> a_strides   = {1};
    std::array<ck::index_t, 1> c_strides   = {1};

    auto broadcastTanh = DeviceElementwiseTanhInstance{};
    auto argument      = broadcastTanh.MakeArgumentPointer(
        abc_lengths, {a_strides}, {c_strides}, input, output, Tanh{});

    if(!broadcastTanh.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    auto broadcastTanh_invoker_ptr = broadcastTanh.MakeInvokerPointer();
    float ave_time =
        broadcastTanh_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_m_device_buf.FromDevice(c_m.mData.data());
        Tensor<CDataType> host_c_m(f_host_tensor_descriptor1d(M, 1));

        host_elementwise1D<Tensor<ADataType>, Tensor<CDataType>, Tanh>(host_c_m, a_m, M, Tanh{});

        pass &= ck::utils::check_err(c_m, host_c_m, "Error: Incorrect results c", 4e-3, 4e-3);
    }

    return pass ? 0 : 1;
}
