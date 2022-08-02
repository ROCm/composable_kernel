// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_f16_f16_rank3_instances(std::vector<DeviceNormalizationPtr>&);
void add_device_softmax_f16_f16_rank4_instances(std::vector<DeviceNormalizationPtr>&);

void add_device_softmax_f32_f32_rank3_instances(std::vector<DeviceNormalizationPtr>&);
void add_device_softmax_f32_f32_rank4_instances(std::vector<DeviceNormalizationPtr>&);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

enum struct NormType
{
    LAYERNORM,
    BATCHNORM,
    SOFTMAX,
};

enum struct NormDataType
{
    F32_F32, // in, out
    F16_F16,
    BF16_BF16,
    INT8_INT8,
};

// clang-format off
template <typename NormDataType> std::string type_to_string();
template <> std::string type_to_string<float>()   { return "f32"; }
template <> std::string type_to_string<half_t>()  { return "f16"; }
template <> std::string type_to_string<bhalf_t>() { return "bf16"; }
template <> std::string type_to_string<int8_t>()  { return "int8"; }
template <> std::string type_to_string<int32_t>() { return "int32"; }
// clang-format on

template <typename InDataType, typename AccDataType, typename OutDataType>
void profile_normalization_impl(int do_verification,
                                int init_method,
                                bool do_log,
                                bool time_kernel,
                                std::vector<index_t> in_length,
                                std::vector<index_t> in_strides,
                                std::vector<index_t> reduce_dims,
                                AccDataType alpha,
                                AccDataType beta,
                                NormType norm_type)
{
    Tensor<InDataType> in = in_strides.empty() ? Tensor<InDataType>(in_length)
                                               : Tensor<InDataType>(in_length, in_strides);
    Tensor<OutDataType> out(in.mDesc);

    switch(init_method)
    {
    // case 0: break;
    case 0:
        in.GenerateTensorValue(GeneratorTensor_1<InDataType>{});
        out.GenerateTensorValue(GeneratorTensor_1<OutDataType>{});
        break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.5, 0.5});
    }

    Tensor<OutDataType> out_ref(out);

    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
    in_dev.ToDevice(in.mData.data());
    out_dev.ToDevice(out.mData.data());

    std::vector<index_t> i_in_lengths(in.mDesc.GetLengths().begin(), in.mDesc.GetLengths().end());
    std::vector<index_t> i_in_strides(in.mDesc.GetStrides().begin(), in.mDesc.GetStrides().end());

    // add device normalization instances
    std::vector<tensor_operation::device::DeviceNormalizationPtr> instances;

    if(norm_type == NormType::SOFTMAX)
    {
        if constexpr(is_same<InDataType, half_t>::value && is_same<OutDataType, half_t>::value &&
                     is_same<AccDataType, float>::value)
        {
            if(in_length.size() == 3)
                tensor_operation::device::instance::add_device_softmax_f16_f16_rank3_instances(
                    instances);

            if(in_length.size() == 4)
                tensor_operation::device::instance::add_device_softmax_f16_f16_rank4_instances(
                    instances);
        }
        else if constexpr(is_same<InDataType, float>::value && is_same<OutDataType, float>::value &&
                          is_same<AccDataType, float>::value)
        {
            if(in_length.size() == 3)
                tensor_operation::device::instance::add_device_softmax_f32_f32_rank3_instances(
                    instances);

            if(in_length.size() == 4)
                tensor_operation::device::instance::add_device_softmax_f32_f32_rank4_instances(
                    instances);
        }
    }

    if(instances.size() <= 0)
    {
        throw std::runtime_error("wrong! no device normalization instance found");
    }

    std::string best_instance_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    for(auto& inst_ptr : instances)
    {
        // Is this user's responsibility to check if problem mismatches kernel instance (ie. rank 3
        // problem to rank 4 kernel) other than invoking IsSupportedArgument()?
        if(!(inst_ptr->GetRank() == static_cast<index_t>(i_in_lengths.size()) &&
             inst_ptr->GetNumReduceDim() == static_cast<index_t>(reduce_dims.size())))
        {
            continue;
        }

        auto argument_ptr = inst_ptr->MakeArgumentPointer(i_in_lengths,
                                                          i_in_strides,
                                                          reduce_dims,
                                                          &alpha,
                                                          &beta,
                                                          in_dev.GetDeviceBuffer(),
                                                          out_dev.GetDeviceBuffer());

        if(!inst_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::cout << inst_ptr->GetTypeString() << " skipped due to unsupported argument: ";
            LogRange(std::cout << "input lengths = [", in_length, ", ")
                << "], "
                << "scaler = [" << alpha << ", " << beta << "]." << std::endl;
            return;
        }

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t num_bytes =
            in.mDesc.GetElementSize() * sizeof(InDataType) +
            (beta == 0.0f ? 1 : 2) * out.mDesc.GetElementSize() * sizeof(OutDataType);

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
            // TODO: factory method to dynamically switch between different reference normalizations
            using ReferenceFactory =
                tensor_operation::host::ReferenceSoftmax<InDataType, OutDataType, AccDataType>;

            ReferenceFactory{}.MakeInvoker().Run({in, out_ref, alpha, beta, reduce_dims});

            out_dev.FromDevice(out.mData.data());

            bool pass;
            if(std::is_same<InDataType, int8_t>::value)
            {
                pass = ck::utils::check_err(
                    out.mData, out_ref.mData, "Error: Incorrect results!", 0, 1);
                if(do_log)
                {
                    LogRangeAsType<int>(std::cout << "in  : ", in.mData, ",") << std::endl;
                    LogRangeAsType<int>(std::cout << "out_ref  : ", out_ref.mData, ",")
                        << std::endl;
                    LogRangeAsType<int>(std::cout << "out  : ", out.mData, ",") << std::endl;
                }
            }
            else
            {
                pass = ck::utils::check_err(out.mData, out_ref.mData);
                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "in  : ", in.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "out_ref  : ", out_ref.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "out  : ", out.mData, ",") << std::endl;
                }
            }

            if(!pass)
            {
                std::cout << inst_ptr->GetTypeString() << " failed verification: ";
                LogRange(std::cout << "input lengths = [", in_length, ", ")
                    << "], "
                    << "scaler = [" << alpha << ", " << beta << "]." << std::endl;
            }
        }
    }
    std::cout << "Best Perf for datatype = " << type_to_string<InDataType>() << "_"
              << type_to_string<OutDataType>() << ", ";
    LogRange(std::cout << "length = ", i_in_lengths, ",") << ", ";
    LogRange(std::cout << "stride = ", i_in_strides, ",") << ", ";
    LogRange(std::cout << "reduce dims ", reduce_dims, ",") << ", ";
    std::cout << "alpha = " << alpha << ", "
              << "beta = " << beta << ", " << best_avg_time << " ms, " << best_gb_per_sec
              << " GB/s, " << best_instance_name << std::endl;
}

} // namespace profiler
} // namespace ck
