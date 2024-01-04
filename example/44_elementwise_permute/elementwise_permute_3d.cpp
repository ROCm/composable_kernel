// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_3d_impl.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ADataType = F32;
using BDataType = F32;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using DeviceElementwisePermuteInstance =
    ck::tensor_operation::device::DeviceElementwise3dImpl<ck::Tuple<ADataType>, // InDataTypeTuple
                                                          ck::Tuple<BDataType>, // OutDataTypeTuple
                                                          PassThrough,          // ElementwiseOp
                                                          2,                    // NumDim_m, {N, C}
                                                          2,                    // NumDim_n, {H, W}
                                                          1,                    // NumDim_k, {D}
                                                          4,                    // MPerThread
                                                          4,                    // NPerThread
                                                          4,                    // KPerThread
                                                          ck::Sequence<4>,  // InScalarPerVectorSeq
                                                          ck::Sequence<4>>; // OutScalarPerVectorSeq

template <typename HostTensorA, typename HostTensorB, typename Functor>
void host_elementwise4D(HostTensorB& B_ndhwc, const HostTensorA& A_ncdhw, Functor functor)
{
    for(std::size_t n = 0; n < A_ncdhw.mDesc.GetLengths()[0]; ++n)
        for(std::size_t c = 0; c < A_ncdhw.mDesc.GetLengths()[1]; ++c)
            for(std::size_t d = 0; d < A_ncdhw.mDesc.GetLengths()[2]; ++d)
                for(std::size_t h = 0; h < A_ncdhw.mDesc.GetLengths()[3]; ++h)
                    for(std::size_t w = 0; w < A_ncdhw.mDesc.GetLengths()[4]; ++w)
                    {
                        auto a_val = A_ncdhw(n, c, d, h, w);
                        functor(B_ndhwc(n, d, h, w, c), a_val);
                    }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = true;

    const int N = 4;
    const int C = 16;
    const int H = 32;
    const int W = 5;
    const int D = 16;

    std::vector<std::size_t> ncdhw = {N, C, D, H, W};
    std::vector<std::size_t> ndhwc = {N, D, H, W, C};
    Tensor<ADataType> a(ncdhw);
    Tensor<BDataType> b(ndhwc);

    a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 5> ab_lengths{N, C, H, W, D};
    std::array<ck::index_t, 5> a_strides = {C * D * H * W, H * W, W, 1, D * H * W}; // N, C, D, H, W
    std::array<ck::index_t, 5> b_strides = {C * H * W * D, H * W * D, W * D, D, 1}; // N, D, H, W, C

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument         = broadcastPermute.MakeArgumentPointer(
        ab_lengths, {a_strides}, {b_strides}, input, output, PassThrough{});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    std::cout << "A (ncdhw): " << a.mDesc << std::endl;
    std::cout << "B (ndhwc): " << b.mDesc << std::endl;

    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    std::size_t flop = std::size_t(2) * ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4];

    std::size_t num_btype =
        sizeof(ADataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]) +
        sizeof(BDataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        b_device_buf.FromDevice(b.mData.data());
        Tensor<BDataType> host_b(ndhwc);
        host_elementwise4D(host_b, a, PassThrough{});

        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
