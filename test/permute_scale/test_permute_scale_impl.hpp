// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>
#include <random>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_scale_impl.hpp"

#include "ck/library/tensor_operation_instance/gpu/permute_scale.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

namespace ck {
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
                    using tmp_type   = ck::remove_reference_t<decltype(B_nhwc(0, 0))>;
                    tmp_type tmp_val = 0;
                    auto a_val       = A_nchw.mData[(n) + (c * N) + (h * C * N) + (w * H * C * N)];
                    functor_b(tmp_val, a_val);
                    functor_a(B_nhwc.mData[(n) + (c * W * H * N) + (h * N) + (w * H * N)],
                              scale * tmp_val);
                }
}

template <typename ADataType, typename BDataType, index_t NumDim>
bool test_permute_scale_impl(int do_verification,
                             int init_method,
                             bool do_log,
                             bool time_kernel,
                             std::vector<index_t> lengths)
{
    bool pass = true;

    using ElementOp = ck::tensor_operation::element_wise::PassThrough;
    using UnaryOp   = ck::tensor_operation::element_wise::UnarySquare;
    using Scale     = ck::tensor_operation::element_wise::Scale;
    float scale     = 2.f;

    index_t N = lengths[0];
    index_t C = lengths[1];
    index_t H = lengths[2];
    index_t W = lengths[3];

    std::vector<ck::index_t> nchw = {N, C, H, W};
    std::vector<ck::index_t> nhwc = {N, H, W, C};
    Tensor<ADataType> a(nchw);
    Tensor<BDataType> b(nhwc);
    Tensor<BDataType> host_b(nhwc);

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

    std::cout << "A: " << a.mDesc << std::endl;
    std::cout << "B: " << b.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: a.GenerateTensorValue(GeneratorTensor_2<ADataType>{-1, 2}); break;
    default: // a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}
        std::mt19937 gen(11939);
        std::uniform_int_distribution<int> dis(0, 1);
        auto i = 0;
        for(std::size_t w = 0; w < a.mDesc.GetLengths()[3]; ++w)
            for(std::size_t h = 0; h < a.mDesc.GetLengths()[2]; ++h)
                for(std::size_t c = 0; c < a.mDesc.GetLengths()[1]; ++c)
                    for(std::size_t n = 0; n < a.mDesc.GetLengths()[0]; ++n)
                    {
                        a.mData[(n * nchw[1] * nchw[2] * nchw[3]) + (c * nchw[2] * nchw[3]) +
                                (h * nchw[3]) + w] = i;
                        i                          = dis(gen);
                    }
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};
    using DeviceOp = ck::tensor_operation::device::DeviceElementwise<ck::Tuple<ADataType>,
                                                                     ck::Tuple<BDataType>,
                                                                     ElementOp,
                                                                     UnaryOp,
                                                                     Scale,
                                                                     NumDim>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_instance_name;
    float best_ave_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    float best_tflops     = 0;

    if(do_verification)
    {
        host_elementwise4D(host_b, a, ElementOp{}, UnaryOp{}, scale);
    }

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(ab_lengths,
                                                        {a_strides},
                                                        {b_strides},
                                                        input,
                                                        output,
                                                        ElementOp{},
                                                        UnaryOp{},
                                                        Scale{scale});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            b_device_buf.SetZero();

            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

            if(do_verification)
            {
                b_device_buf.FromDevice(b.mData.data());

                pass &= ck::utils::check_err(
                    b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b.mData, ",") << std::endl;
                }
            }

            std::string op_name = op_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = std::size_t(2) * nchw[0] * nchw[1] * nchw[2] * nchw[3];

            std::size_t num_btype = sizeof(ADataType) * (nchw[0] * nchw[1] * nchw[2] * nchw[3]) +
                                    sizeof(BDataType) * (nchw[0] * nchw[1] * nchw[2] * nchw[3]);

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_instance_name = op_name;
                best_tflops        = tflops;
                best_ave_time      = ave_time;
                best_gb_per_sec    = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }
    if(time_kernel)
    {
        LogRange(std::cout << "length = ", lengths, ",") << ", ";
        std::cout << "best perf = " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
                  << best_instance_name << std::endl;
    }

    return true;
}

} // namespace ck
