/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <iostream>
#include <cstdlib>
#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"

#include "device_tensor.hpp"
#include "binary_element_wise_operation.hpp"
#include "device_binary_elementwise.hpp"

using F16 = ck::half_t;
using F32 = float;

using ABDataType             = F16;
using CDataType              = F16;
using EltwiseComputeDataType = F32;

using Add = ck::tensor_operation::binary_element_wise::
    Add<EltwiseComputeDataType, EltwiseComputeDataType, EltwiseComputeDataType>;

using DeviceElementwiseAddInstance =
    ck::tensor_operation::device::DeviceBinaryElementwise<ABDataType,
                                                          ABDataType,
                                                          CDataType,
                                                          EltwiseComputeDataType,
                                                          Add,
                                                          2,
                                                          8,
                                                          8,
                                                          8,
                                                          8>;

template <typename HostTensorA,
          typename HostTensorB,
          typename HostTensorC,
          typename ComputeDataType,
          typename Functor,
          int broadcastDim>
void host_broadcast2D(
    HostTensorC& C, const HostTensorA& A, const HostTensorB& B, int M, int N, Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0, 0))>;

    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            ComputeDataType Amn = ck::type_convert<ComputeDataType>(A(m, n));
            ComputeDataType Cmn = 0;
            if constexpr(broadcastDim == 0)
            {
                ComputeDataType Bn = ck::type_convert<ComputeDataType>(B(n));
                functor(Cmn, Amn, Bn);
            }
            else
            {
                ComputeDataType Bm = ck::type_convert<ComputeDataType>(B(m));
                functor(Cmn, Amn, Bm);
            }
            C(m, n) = ck::type_convert<ctype>(Cmn);
        }
    }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    ck::index_t M      = 1024;
    ck::index_t N      = 1024;
    ck::index_t Stride = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor(std::vector<std::size_t>({len}),
                                    std::vector<std::size_t>({stride}));
    };

    auto f_host_tensor_descriptor2d = [](std::size_t row, std::size_t col, std::size_t stride) {
        return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                    std::vector<std::size_t>({stride, 1}));
    };

    Tensor<ABDataType> a_m_n(f_host_tensor_descriptor2d(M, N, Stride));
    Tensor<ABDataType> b_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<CDataType> c_m_n(f_host_tensor_descriptor2d(M, N, Stride));

    a_m_n.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});
    b_n.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});

    DeviceMem a_m_n_device_buf(sizeof(ABDataType) * a_m_n.mDesc.GetElementSpace());
    DeviceMem b_n_device_buf(sizeof(ABDataType) * b_n.mDesc.GetElementSpace());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n.mDesc.GetElementSpace());

    a_m_n_device_buf.ToDevice(a_m_n.mData.data());
    b_n_device_buf.ToDevice(b_n.mData.data());

    auto broadcastAdd = DeviceElementwiseAddInstance{};
    auto argument     = broadcastAdd.MakeArgumentPointer(a_m_n_device_buf.GetDeviceBuffer(),
                                                     b_n_device_buf.GetDeviceBuffer(),
                                                     c_m_n_device_buf.GetDeviceBuffer(),
                                                     {M, N},
                                                     {Stride, 1},
                                                     {0, 1}, // broadcast in first dimension
                                                     {Stride, 1},
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
        c_m_n_device_buf.FromDevice(c_m_n.mData.data());
        Tensor<CDataType> host_c_m_n(f_host_tensor_descriptor2d(M, N, Stride));

        host_broadcast2D<Tensor<ABDataType>,
                         Tensor<ABDataType>,
                         Tensor<CDataType>,
                         EltwiseComputeDataType,
                         Add,
                         0>(host_c_m_n, a_m_n, b_n, M, N, Add{});

        pass &= ck::utils::check_err(
            c_m_n.mData, host_c_m_n.mData, "Error: Incorrect results c", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
