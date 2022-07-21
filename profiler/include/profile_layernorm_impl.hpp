// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "profiler/include/data_type_enum.hpp"
#include "ck/tensor_operation/gpu/device/device_layernorm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_layernorm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16         = ck::half_t;
using F32         = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

void add_device_layernorm_f16_rank2_instances(
    std::vector<DeviceNormalization2Ptr<F16, F16, F16, F32, F16, PassThrough, 2, 1>>&);

void add_device_layernorm_f32_rank2_instances(
    std::vector<DeviceNormalization2Ptr<F32, F32, F32, F32, F32, PassThrough, 2, 1>>&);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          index_t Rank>
void profile_layernorm_impl(int do_verification,
                            int init_method,
                            bool do_log,
                            bool time_kernel,
                            std::vector<index_t> length,
                            std::vector<index_t> strideXY,
                            std::vector<index_t> strideGamma,
                            std::vector<index_t> strideBeta)
{
    using F16         = ck::half_t;
    using F32         = float;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    if(length.size() < 2)
        return;

    // Assume normalize dimension except for first dimension
    std::vector<index_t> reduce_length{length.begin() + 1, length.end()};
    std::vector<index_t> reduce_dim;
    for(int i = 1; i < Rank; ++i)
        reduce_dim.push_back(i);

    Tensor<XDataType> x(length);
    Tensor<GammaDataType> gamma(reduce_length, strideGamma);
    Tensor<BetaDataType> beta(reduce_length, strideBeta);
    Tensor<YDataType> y(length, strideXY);
    Tensor<YDataType> host_y(length, strideXY);

    switch(init_method)
    {
    // case 0: break;
    case 0:
        x.GenerateTensorValue(GeneratorTensor_1<XDataType>{});
        gamma.GenerateTensorValue(GeneratorTensor_1<GammaDataType>{});
        beta.GenerateTensorValue(GeneratorTensor_1<BetaDataType>{});
        y.GenerateTensorValue(GeneratorTensor_1<YDataType>{});
        break;
    case 1:
        x.GenerateTensorValue(GeneratorTensor_2<XDataType>{-5, 5});
        gamma.GenerateTensorValue(GeneratorTensor_2<GammaDataType>{-5, 5});
        beta.GenerateTensorValue(GeneratorTensor_2<BetaDataType>{-5, 5});
        y.GenerateTensorValue(GeneratorTensor_2<YDataType>{-5, 5});
        break;
    default:
        x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1});
        gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{-0.5, 0.5});
        beta.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{-0.5, 0.5});
        y.GenerateTensorValue(GeneratorTensor_3<YDataType>{-0.5, 0.5});
    }

    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpace());
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpace());
    DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpace());
    DeviceMem y_dev(sizeof(YDataType) * y.mDesc.GetElementSpace());

    x_dev.ToDevice(x.mData.data());
    gamma_dev.ToDevice(gamma.mData.data());
    beta_dev.ToDevice(beta.mData.data());

    // add device normalization instances
    constexpr int NumReduceDim = Rank - 1;
    std::vector<tensor_operation::device::DeviceNormalization2Ptr<XDataType,
                                                                  GammaDataType,
                                                                  BetaDataType,
                                                                  AccDataType,
                                                                  YDataType,
                                                                  PassThrough,
                                                                  Rank,
                                                                  NumReduceDim>>
        instances;

    if constexpr(is_same<XDataType, F16>::value && is_same<GammaDataType, F16>::value &&
                 is_same<BetaDataType, F16>::value && is_same<YDataType, F16>::value &&
                 is_same<AccDataType, F32>::value)
    {
        if(length.size() == 2)
            tensor_operation::device::instance::add_device_layernorm_f16_rank2_instances(instances);
    }
    else if constexpr(is_same<XDataType, F32>::value && is_same<GammaDataType, F32>::value &&
                      is_same<BetaDataType, F32>::value && is_same<YDataType, F32>::value &&
                      is_same<AccDataType, F32>::value)
    {
        if(length.size() == 2)
            tensor_operation::device::instance::add_device_layernorm_f32_rank2_instances(instances);
    }

    if(instances.size() <= 0)
    {
        throw std::runtime_error("wrong! no device normalization instance found");
    }

    std::string best_instance_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    if(do_verification)
    {
        using ReferenceInstance = ck::tensor_operation::host::ReferenceLayernorm<XDataType,
                                                                                 GammaDataType,
                                                                                 BetaDataType,
                                                                                 YDataType,
                                                                                 AccDataType,
                                                                                 PassThrough,
                                                                                 Rank,
                                                                                 NumReduceDim>;

        ReferenceInstance ref;
        auto ref_argument =
            ref.MakeArgument(x, gamma, beta, host_y, PassThrough{}, length, reduce_dim, 1e-4);
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);
    }

    for(auto& inst_ptr : instances)
    {
        auto argument_ptr = inst_ptr->MakeArgumentPointer(length,
                                                          strideXY,
                                                          strideGamma,
                                                          strideBeta,
                                                          reduce_dim,
                                                          1e-4,
                                                          x_dev.GetDeviceBuffer(),
                                                          gamma_dev.GetDeviceBuffer(),
                                                          beta_dev.GetDeviceBuffer(),
                                                          y_dev.GetDeviceBuffer(),
                                                          PassThrough{});

        if(!inst_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::cout << inst_ptr->GetTypeString() << " skipped due to unsupported argument: ";
            LogRange(std::cout << "input lengths = [", length, "], ") << std::endl;

            return;
        }

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t num_bytes = x.mDesc.GetElementSize() * sizeof(XDataType) +
                                gamma.mDesc.GetElementSize() * sizeof(GammaDataType) +
                                beta.mDesc.GetElementSize() * sizeof(BetaDataType) +
                                y.mDesc.GetElementSize() * sizeof(YDataType);

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
                  << inst_ptr->GetTypeString() << std::endl;

        if(avg_time < best_avg_time)
        {
            best_instance_name = inst_ptr->GetTypeString();
            best_avg_time      = avg_time;
            best_gb_per_sec    = gb_per_sec;
        }

        if(do_verification)
        {
            y_dev.FromDevice(y.mData.data());

            bool pass = ck::utils::check_err(
                y.mData, host_y.mData, "Error: Incorrect results d1", 1e-3, 1e-3);

            if(do_log)
            {
                LogRangeAsType<float>(std::cout << "x  : ", x.mData, ",") << std::endl;
                LogRangeAsType<float>(std::cout << "host_y  : ", host_y.mData, ",") << std::endl;
                LogRangeAsType<float>(std::cout << "y  : ", y.mData, ",") << std::endl;
            }

            if(!pass)
            {
                std::cout << inst_ptr->GetTypeString() << " failed verification: ";
                LogRange(std::cout << "lengths = [", length, ", ") << "]." << std::endl;
                return;
            }
            else
            {
                std::cout << "pass" << std::endl;
            }
        }
    }

    LogRange(std::cout << "length = ", length, ",") << ", ";
    LogRange(std::cout << "stride = ", strideXY, ",") << ", ";
    LogRange(std::cout << "reduce dims ", reduce_dim, ",") << std::endl;
    std::cout << "best perf = " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s, "
              << best_instance_name << std::endl;
}

} // namespace profiler
} // namespace ck
