// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
template <typename HostTensorA,
          typename HostTensorB,
          typename AElementOp,
          typename BElementOp,
          typename ScaleElementOp>
void reference_permute_scale(HostTensorB& b_tensor,
                             const HostTensorA& a_tensor,
                             AElementOp a_tensor_op,
                             BElementOp b_tensor_op,
                             ScaleElementOp scale_op)
{
    b_tensor.ForEach([&](auto& self, auto idx) {
        auto tmp_val = a_tensor(idx);
        b_tensor_op(tmp_val, tmp_val);
        scale_op(tmp_val, tmp_val);
        a_tensor_op(self(idx), tmp_val);
    });
}

namespace profiler {

template <typename ADataType, typename BDataType, index_t NumDim>
bool profile_permute_scale_impl(int do_verification,
                                int init_method,
                                bool do_log,
                                bool time_kernel,
                                std::vector<index_t> lengths_vector,
                                std::vector<index_t> input_strides_vector,
                                std::vector<index_t> output_strides_vector)
{
    bool pass           = true;
    bool instance_found = false;

    using ElementOp = ck::tensor_operation::element_wise::PassThrough;
    using UnaryOp   = ck::tensor_operation::element_wise::UnarySquare;
    using Scale     = ck::tensor_operation::element_wise::Scale;
    float scale     = 2.f;

    Tensor<ADataType> a(lengths_vector, input_strides_vector);
    Tensor<BDataType> b(lengths_vector, output_strides_vector);
    Tensor<BDataType> host_b(lengths_vector, output_strides_vector);

    std::cout << "A: " << a.mDesc << std::endl;
    std::cout << "B: " << b.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: a.GenerateTensorValue(GeneratorTensor_2<ADataType>{-1, 2}); break;
    default: a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}); break;
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
        reference_permute_scale(host_b, a, ElementOp{}, UnaryOp{}, Scale{scale});
    }

    auto copy = [](const auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };
    std::array<ck::index_t, NumDim> lengths{};
    std::array<ck::index_t, NumDim> input_strides{};
    std::array<ck::index_t, NumDim> output_strides{};
    copy(lengths_vector, lengths);
    copy(input_strides_vector, input_strides);
    copy(output_strides_vector, output_strides);

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(lengths,
                                                        {input_strides},
                                                        {output_strides},
                                                        input,
                                                        output,
                                                        ElementOp{},
                                                        UnaryOp{},
                                                        Scale{scale});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            instance_found = true;

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

            std::size_t flop = std::size_t(2) * a.mDesc.GetElementSpaceSize() / sizeof(ADataType);

            std::size_t num_btype = sizeof(ADataType) * a.mDesc.GetElementSpaceSize() +
                                    sizeof(BDataType) * b.mDesc.GetElementSpaceSize();

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
        std::cout << "Best perf = " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
                  << best_instance_name << std::endl;
    }

    return pass && instance_found;
}

} // namespace profiler
} // namespace ck
