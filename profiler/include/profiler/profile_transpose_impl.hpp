// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType, typename BDataType>
bool profile_gemm_splitk_impl(int do_verification,
                              int init_method,
                              bool do_log,
                              bool time_kernel,
                              int N,
                              int C,
                              int D,
                              int H,
                              int W)
{
    bool pass = true;

    std::vector<std::size_t> ncdhw = {N, C, D, H, W};
    std::vector<std::size_t> nchwd = {N, C, H, W, D};
    Tensor<ADataType> a(ncdhw);
    Tensor<BDataType> b(nchwd);

    // a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 5> ab_lengths{N, C, H, W, D};
    std::array<ck::index_t, 5> a_strides = {C * D * H * W, D * H * W, 1, D * H, D};
    std::array<ck::index_t, 5> b_strides = {C * H * W * D, H * W * D, W * D, D, 1};

    std::cout << "A: " << a.mDesc << std::endl;
    std::cout << "B: " << b.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: a.GenerateTensorValue(GeneratorTensor_2<ADataType>{-1, 2}); break;
    default: a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
    }

    using ElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto element_op = ElementOp{};

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    using DeviceOp =
        ck::tensor_operation::device::DeviceElementwise3dImpl<ck::Tuple<ADataType>,
                                                              ck::Tuple<BDataType>,
                                                              ElementOp,
                                                              NumDim_m,
                                                              NumDim_n,
                                                              NumDim_k,
                                                              MPerThread,
                                                              NPerThread,
                                                              KPerThread,
                                                              ck::Sequence<InScalarPerVector>,
                                                              ck::Sequence<OutScalarPerVector>>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    if(do_verification)
    {
        using ReferenceTransposeInstance = ck::tensor_operation::host::ReferenceTranspose
                                           << ck::Tuple<ADataType>,
              ck::Tuple<BDataType>, ElementOp, NumDim_m, NumDim_n, NumDim_k, MPerThread, NPerThread,
              KPerThread, ck::Sequence<InScalarPerVector>, ck::Sequence<OutScalarPerVector> > ;

        auto ref_transpose = ReferenceTransposeInstance{};
        auto ref_invoker   = ref_transpose.MakeInvoker();

        auto ref_argument =
            ref_transpose
                .MakeArgument(ab_lengths, {a_strides}, {b_strides}, input, output, element_op{})

                    ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            ab_lengths, {a_strides}, {b_strides}, input, output, element_op{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {

            // re-init C to zero before profiling next kernel
            b_device_buf.SetZero();

            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

            if(do_verification)
            {
                b_device_buf.FromDevice(b_device_result.mData.data());

                pass = pass & ck::utils::check_err(b_device_result, b_host_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b.mData, ",") << std::endl;
                }
            }

            std::string op_name = op_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop =
                std::size_t(2) * ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4];

            std::size_t num_btype =
                sizeof(ADataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]) +
                sizeof(BDataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]);

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            pass = pass & ck::utils::check_err(b_device_result, b_host_result);

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }
}

if constexpr(is_same<BDataType, float>::value)
{
    std::cout << "Best Perf for datatype = f32";
}
else if constexpr(is_same<BDataType, half_t>::value)
{
    std::cout << "Best Perf for datatype = f16";
}

std::cout << " N = " << N << " C = " << C << " D = " << D << " H = " << H << " W = " << W << " : "
          << best_ave_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec << " GB/s, "
          << best_op_name << std::endl;

return pass;
}

} // namespace profiler
} // namespace ck