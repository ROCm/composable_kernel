// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ADataType = F16;
using BDataType = F16;

using UnaryScale  = ck::tensor_operation::element_wise::Scale;
using UnarySquare = ck::tensor_operation::element_wise::UnarySquare;
using UnaryScaleSquare =
    ck::tensor_operation::element_wise::UnaryCombinedOp<UnarySquare, UnaryScale>;
using BinaryAdd = ck::tensor_operation::element_wise::Add;
// B = alpha * A0 * A0 + beta * A1 * A1 + gamma * A2 * A2
using TrinaryAddUnaryScaleSquare =
    ck::tensor_operation::element_wise::TrinaryWithUnaryCombinedOp<BinaryAdd,
                                                                   BinaryAdd,
                                                                   UnaryScaleSquare,
                                                                   UnaryScaleSquare,
                                                                   UnaryScaleSquare>;
using DeviceElementwisePermuteInstance = ck::tensor_operation::device::DeviceElementwiseImpl<
    ck::Tuple<ADataType, ADataType, ADataType>, // InDataTypeTuple
    ck::Tuple<BDataType>,                       // OutDataTypeTuple
    TrinaryAddUnaryScaleSquare,                 // ElementwiseOp
    4,                                          // NumDim
    256,                                        // BlockSize
    128,                                        // M0PerBlock
    128,                                        // M1PerBlock
    8,                                          // M0PerThread
    8,                                          // M1PerThread
    ck::Sequence<1, 0>,                         // ThreadClusterArrangeOrder
    ck::Sequence<8, 8, 8>,                      // InScalarPerVectorSeq
    ck::Sequence<8>>;                           // OutScalarPerVectorSeq

template <typename HostTensorA0, typename HostTensorA1, typename HostTensorA2, typename HostTensorB>
void host_elementwise4D(HostTensorB& B,
                        const HostTensorA0& A0,
                        const HostTensorA1& A1,
                        const HostTensorA2& A2,
                        float alpha,
                        float beta,
                        float gamma)
{
    auto elementwise_op_function = [&](auto& b, auto& a0, auto& a1, auto& a2) {
        const float a0_tmp = alpha * ck::type_convert<float>(a0) * ck::type_convert<float>(a0);
        const float a1_tmp = beta * ck::type_convert<float>(a1) * ck::type_convert<float>(a1);
        const float a2_tmp = gamma * ck::type_convert<float>(a2) * ck::type_convert<float>(a2);
        b                  = ck::type_convert<BDataType>(a0_tmp + a1_tmp + a2_tmp);
    };

    for(std::size_t n = 0; n < A0.mDesc.GetLengths()[0]; ++n)
        for(std::size_t c = 0; c < A0.mDesc.GetLengths()[1]; ++c)
            for(std::size_t h = 0; h < A0.mDesc.GetLengths()[2]; ++h)
                for(std::size_t w = 0; w < A0.mDesc.GetLengths()[3]; ++w)
                {
                    auto a0_val = A0(n, c, h, w);
                    auto a1_val = A1(n, c, h, w);
                    auto a2_val = A2(n, c, h, w);
                    elementwise_op_function(B(n, c, h, w), a0_val, a1_val, a2_val);
                }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = true;

    std::vector<std::size_t> nchw = {16, 128, 32, 64};
    Tensor<ADataType> a0(nchw);
    Tensor<ADataType> a1(nchw);
    Tensor<ADataType> a2(nchw);
    Tensor<BDataType> b(nchw);
    float alpha = 3.f;
    float beta  = 2.f;
    float gamma = 4.f;
    a0.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
    a1.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
    a2.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});

    DeviceMem a0_device_buf(sizeof(ADataType) * a0.mDesc.GetElementSpaceSize());
    DeviceMem a1_device_buf(sizeof(ADataType) * a1.mDesc.GetElementSpaceSize());
    DeviceMem a2_device_buf(sizeof(ADataType) * a2.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0.mData.data());
    a1_device_buf.ToDevice(a1.mData.data());
    a2_device_buf.ToDevice(a2.mData.data());

    std::array<const void*, 3> inputs = {a0_device_buf.GetDeviceBuffer(),
                                         a1_device_buf.GetDeviceBuffer(),
                                         a2_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output       = {b_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 4> ab_lengths;
    std::array<ck::index_t, 4> ab_strides = {static_cast<int>(nchw[1] * nchw[2] * nchw[3]),
                                             static_cast<int>(nchw[2] * nchw[3]),
                                             static_cast<int>(nchw[3]),
                                             1};

    ck::ranges::copy(nchw, ab_lengths.begin());

    auto broadcastPermute  = DeviceElementwisePermuteInstance{};
    auto unary_scale_op_a0 = UnaryScaleSquare{UnarySquare{}, UnaryScale{alpha}};
    auto unary_scale_op_a1 = UnaryScaleSquare{UnarySquare{}, UnaryScale{beta}};
    auto unary_scale_op_a2 = UnaryScaleSquare{UnarySquare{}, UnaryScale{gamma}};
    auto argument          = broadcastPermute.MakeArgumentPointer(
        ab_lengths,
        {ab_strides, ab_strides, ab_strides},
        {ab_strides},
        inputs,
        output,
        TrinaryAddUnaryScaleSquare{
            BinaryAdd{}, BinaryAdd{}, unary_scale_op_a0, unary_scale_op_a1, unary_scale_op_a2});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    std::cout << "A0 (nchw): " << a0.mDesc << std::endl;
    std::cout << "A1 (nchw): " << a1.mDesc << std::endl;
    std::cout << "A2 (nchw): " << a2.mDesc << std::endl;
    std::cout << "B (nchw): " << b.mDesc << std::endl;

    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    std::size_t flop = std::size_t(5) * nchw[0] * nchw[1] * nchw[2] * nchw[3];

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
        Tensor<BDataType> host_b(nchw);
        host_elementwise4D(host_b, a0, a1, a2, alpha, beta, gamma);

        const double threshold = std::pow(2, -10) * 2;

        pass &= ck::utils::check_err(
            b.mData, host_b.mData, "Error: Incorrect results b", threshold, threshold);
    }

    return pass ? 0 : 1;
}
