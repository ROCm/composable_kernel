// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>
#include <random>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_scale_impl.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ADataType = F16;
using BDataType = F16;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using UnaryOp     = ck::tensor_operation::element_wise::UnarySquare;
using Scale       = ck::tensor_operation::element_wise::Scale;
using DeviceElementwisePermuteInstance =
    ck::tensor_operation::device::DeviceElementwiseImpl<ck::Tuple<ADataType>, // InDataTypeTuple
                                                        ck::Tuple<BDataType>, // OutDataTypeTuple
                                                        PassThrough,          // ElementwiseOp
                                                        UnaryOp,              // UnaryOp
                                                        Scale,                // Scalar
                                                        4,                    // NumDim
                                                        8,                    // MPerThread
                                                        ck::Sequence<1>,  // InScalarPerVectorSeq
                                                        ck::Sequence<1>>; // OutScalarPerVectorSeq

template <typename HostTensorA, typename HostTensorB, typename FunctorA, typename FunctorB>
void host_elementwise4D(HostTensorB& B_nhwc,
                        const HostTensorA& A_nchw,
                        FunctorA functor_a,
                        FunctorB functor_b,
                        float scale)
{
    std::size_t N = A_nchw.mDesc.GetLengths()[0];
    std::size_t C = A_nchw.mDesc.GetLengths()[1];
    std::size_t H = A_nchw.mDesc.GetLengths()[2];
    std::size_t W = A_nchw.mDesc.GetLengths()[3];
    for(std::size_t w = 0; w < W; ++w)
        for(std::size_t h = 0; h < H; ++h)
            for(std::size_t c = 0; c < C; ++c)
                for(std::size_t n = 0; n < N; ++n)
                {
                    ADataType tmp_val;
                    auto a_val = A_nchw.mData[(n) + (c * N) + (h * C * N) + (w * H * C * N)];
                    functor_b(tmp_val, a_val);
                    functor_a(B_nhwc.mData[(n) + (c * W * H * N) + (h * N) + (w * H * N)],
                              scale * tmp_val);
                }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = true;

    std::vector<std::size_t> nchw = {16, 8, 32, 64};
    std::vector<std::size_t> nhwc = {16, 32, 64, 8};
    Tensor<ADataType> a(nchw);
    Tensor<BDataType> b(nhwc);
    float scale = 1.f;
    auto i      = 0;
    std::mt19937 gen(11939);
    std::uniform_int_distribution<int> dis(0, 1);
    for(std::size_t w = 0; w < a.mDesc.GetLengths()[3]; ++w)
        for(std::size_t h = 0; h < a.mDesc.GetLengths()[2]; ++h)
            for(std::size_t c = 0; c < a.mDesc.GetLengths()[1]; ++c)
                for(std::size_t n = 0; n < a.mDesc.GetLengths()[0]; ++n)
                {
                    a.mData[(n * nchw[1] * nchw[2] * nchw[3]) + (c * nchw[2] * nchw[3]) +
                            (h * nchw[3]) + w] = i;
                    i                          = dis(gen);
                }

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 4> ab_lengths;

    std::array<ck::index_t, 4> a_strides = {1,
                                            static_cast<int>(nchw[0]),
                                            static_cast<int>(nchw[0] * nchw[1]),
                                            static_cast<int>(nchw[0] * nchw[1] * nchw[2])};

    std::array<ck::index_t, 4> b_strides = {1,
                                            static_cast<int>(nhwc[0] * nhwc[1] * nhwc[2]),
                                            static_cast<int>(nhwc[0]),
                                            static_cast<int>(nhwc[0] * nhwc[1])};
    ck::ranges::copy(nchw, ab_lengths.begin());

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument         = broadcastPermute.MakeArgumentPointer(ab_lengths,
                                                         {a_strides},
                                                         {b_strides},
                                                         input,
                                                         output,
                                                         PassThrough{},
                                                         UnaryOp{},
                                                         Scale{scale});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    std::cout << "A (nchw): " << a.mDesc << std::endl;
    std::cout << "B (nhwc): " << b.mDesc << std::endl;

    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    std::size_t flop = std::size_t(2) * nchw[0] * nchw[1] * nchw[2] * nchw[3];

    std::size_t num_btype = sizeof(ADataType) * (nchw[0] * nchw[1] * nchw[2] * nchw[3]) +
                            sizeof(BDataType) * (nchw[0] * nchw[1] * nchw[2] * nchw[3]);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        b_device_buf.FromDevice(b.mData.data());
        Tensor<BDataType> host_b(nhwc);
        host_elementwise4D(host_b, a, PassThrough{}, UnaryOp{}, scale);

        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
