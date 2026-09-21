// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_bias.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/env.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/type.hpp"

#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout>
bool profile_grouped_gemm_fixed_nk_bias_impl(int do_verification,
                                             int init_method,
                                             bool do_log,
                                             bool time_kernel,
                                             const std::vector<int>& Ms,
                                             const std::vector<int>& Ns,
                                             const std::vector<int>& Ks,
                                             const std::vector<int>& StrideAs,
                                             const std::vector<int>& StrideBs,
                                             const std::vector<int>& StrideDs,
                                             const std::vector<int>& StrideEs,
                                             const std::vector<int>& kbatches = {1},
                                             int n_warmup                     = 1,
                                             int n_iter                       = 10)
{
    bool pass             = true;
    using ComputeDataType = ADataType;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz}, layout);
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride}, layout);
            }
        };

    std::size_t group_count = Ms.size();

    if(!(group_count == Ns.size() && group_count == Ks.size() && group_count == StrideAs.size() &&
         group_count == StrideBs.size() && group_count == StrideDs.size() &&
         group_count == StrideEs.size()))
    {
        throw std::runtime_error("wrong! inconsistent M/N/Ks, StrideA/B/Cs size\n");
    }

    using D0DataType = remove_cvref_t<ck::tuple_element_t<Number<0>{}, DsDataType>>;

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<D0DataType>> d0_tensors;
    std::vector<Tensor<EDataType>> e_host_tensors;
    std::vector<Tensor<EDataType>> e_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    d0_tensors.reserve(group_count);
    e_host_tensors.reserve(group_count);
    e_device_tensors.reserve(group_count);

    double max_abs_in_val = 0.f;
    int sum_of_m          = 0;

    using D0Layout = remove_cvref_t<ck::tuple_element_t<Number<0>{}, DsLayout>>;

    for(std::size_t i = 0; i < group_count; ++i)
    {
        sum_of_m += Ms[i];
        a_tensors.push_back(
            Tensor<ADataType>(f_host_tensor_descriptor(Ms[i], Ks[i], StrideAs[i], ALayout{})));
        b_tensors.push_back(
            Tensor<BDataType>(f_host_tensor_descriptor(Ks[i], Ns[i], StrideBs[i], BLayout{})));
        d0_tensors.push_back(
            Tensor<D0DataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideDs[i], D0Layout{})));
        e_host_tensors.push_back(
            Tensor<EDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideEs[i], ELayout{})));
        e_device_tensors.push_back(
            Tensor<EDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideEs[i], ELayout{})));

        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            std::cout << "group: " << i << " a_m_k[" << i << "]:" << a_tensors[i].mDesc
                      << ", b_k_n[" << i << "]:" << b_tensors[i].mDesc << ", d_m_n[" << i
                      << "]:" << d0_tensors[i].mDesc << ", e_m_n_device_results[" << i
                      << "]:" << e_device_tensors[i].mDesc << std::endl;
        }
        switch(init_method)
        {
        case 0: break;
        case 1:
            ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_tensors[i]);
            ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_tensors[i]);
            max_abs_in_val = 10.f;
            break;
        default:
            ck::utils::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_tensors[i]);
            ck::utils::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_tensors[i]);
            max_abs_in_val = 1.0f;
        }
        ck::utils::FillUniformDistribution<D0DataType>{-0.5, 0.5}(d0_tensors[i]);
    }

    using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementOp = ck::tensor_operation::element_wise::SplitKAdd;

    constexpr auto a_element_op   = AElementOp{};
    constexpr auto b_element_op   = BElementOp{};
    constexpr auto cde_element_op = CDEElementOp{};

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, d0_tensors_device,
        e_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    d0_tensors_device.reserve(group_count);
    e_tensors_device.reserve(group_count);

    std::vector<const void*> p_a, p_b;
    std::vector<std::array<const void*, 1>> p_ds;
    std::vector<void*> p_e;

    p_a.reserve(group_count);
    p_b.reserve(group_count);
    p_ds.reserve(group_count);
    p_e.reserve(group_count);

    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    gemm_descs.reserve(group_count);

    std::vector<ck::tensor_operation::device::GroupedGemmKernelArgument<1>>
        grouped_gemm_kernel_args_;
    grouped_gemm_kernel_args_.reserve(group_count);

    for(std::size_t i = 0; i < group_count; ++i)
    {
        a_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(ADataType) * a_tensors[i].mDesc.GetElementSpaceSize()));
        b_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(BDataType) * b_tensors[i].mDesc.GetElementSpaceSize()));
        d0_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(D0DataType) * d0_tensors[i].mDesc.GetElementSpaceSize()));
        e_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(EDataType) * e_device_tensors[i].mDesc.GetElementSpaceSize()));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());
        d0_tensors_device[i]->ToDevice(d0_tensors[i].mData.data());

        gemm_descs.push_back(
            {sum_of_m, Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideEs[i], {StrideDs[i]}});

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_ds.push_back(std::array<const void*, 1>{d0_tensors_device[i]->GetDeviceBuffer()});
        p_e.push_back(e_tensors_device[i]->GetDeviceBuffer());

        grouped_gemm_kernel_args_.push_back(
            {a_tensors_device[i]->GetDeviceBuffer(),
             b_tensors_device[i]->GetDeviceBuffer(),
             std::array<const void*, 1>{d0_tensors_device[i]->GetDeviceBuffer()},
             e_tensors_device[i]->GetDeviceBuffer(),
             Ms[i],
             Ns[i],
             Ks[i],
             StrideAs[i],
             StrideBs[i],
             std::array<ck::index_t, 1>{StrideDs[i]},
             StrideEs[i]});
    }

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemmFixedNK<ALayout,
                                                                            BLayout,
                                                                            DsLayout,
                                                                            ELayout,
                                                                            ADataType,
                                                                            BDataType,
                                                                            DsDataType,
                                                                            EDataType,
                                                                            AElementOp,
                                                                            BElementOp,
                                                                            CDEElementOp>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    if(op_ptrs.size() <= 0)
    {
        std::cerr << "Skip! no device GEMM instance found" << std::endl;
        return true;
    }

    std::string best_gemm_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;

    if(do_verification)
    {
        for(std::size_t i = 0; i < gemm_descs.size(); ++i)
        {
            using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<
                ADataType,
                BDataType,
                EDataType,
                AccDataType,
                AElementOp,
                BElementOp,
                ck::tensor_operation::element_wise::PassThrough>;

            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument =
                ref_gemm.MakeArgument(a_tensors[i],
                                      b_tensors[i],
                                      e_host_tensors[i],
                                      a_element_op,
                                      b_element_op,
                                      ck::tensor_operation::element_wise::PassThrough{});

            ref_invoker.Run(ref_argument);

            for(int m = 0; m < Ms[i]; ++m)
            {
                for(int n = 0; n < Ns[i]; ++n)
                {
                    cde_element_op(
                        e_host_tensors[i](m, n), e_host_tensors[i](m, n), d0_tensors[i](m, n));
                }
            }
        }
    }

    // profile device GEMM instances
    for(auto& gemm_ptr : op_ptrs)
    {
        auto argument_ptr = gemm_ptr->MakeArgumentPointer(
            p_a, p_b, p_ds, p_e, gemm_descs, a_element_op, b_element_op, cde_element_op);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        DeviceMem gemm_desc_workspace(gemm_ptr->GetWorkSpaceSize(argument_ptr.get()));

        DeviceMem grouped_gemm_kernel_args_dev(
            gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()));

        hipGetErrorString(hipMemcpy(grouped_gemm_kernel_args_dev.GetDeviceBuffer(),
                                    grouped_gemm_kernel_args_.data(),
                                    gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()),
                                    hipMemcpyHostToDevice));

        gemm_ptr->SetWorkSpacePointer(argument_ptr.get(), gemm_desc_workspace.GetDeviceBuffer());

        gemm_ptr->SetDeviceKernelArgs(argument_ptr.get(),
                                      grouped_gemm_kernel_args_dev.GetDeviceBuffer());

        std::string gemm_name = gemm_ptr->GetTypeString();

        for(std::size_t j = 0; j < kbatches.size(); ++j)
        {

            auto kbatch_curr = kbatches[j];

            gemm_ptr->SetKBatch(argument_ptr.get(), kbatch_curr);

            if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                for(std::size_t i = 0; i < gemm_descs.size(); ++i)
                {
                    e_tensors_device[i]->SetZero();
                }

                invoker_ptr->Run(argument_ptr.get(),
                                 StreamConfig{nullptr, false, 0, n_warmup, n_iter});

                if(do_verification)
                {
                    bool instance_pass = true;
                    for(std::size_t i = 0; i < gemm_descs.size(); ++i)
                    {
                        e_tensors_device[i]->FromDevice(e_device_tensors[i].mData.data());
                        auto atol = ck::utils::get_absolute_threshold<ComputeDataType, EDataType>(
                            max_abs_in_val, gemm_descs[i].K_);
                        auto rtol = ck::utils::get_relative_threshold<ComputeDataType, EDataType>(
                            gemm_descs[i].K_);

                        instance_pass =
                            instance_pass && ck::utils::check_err(e_device_tensors[i],
                                                                  e_host_tensors[i],
                                                                  "Error: Incorrect results!",
                                                                  rtol,
                                                                  atol);

                        if(do_log)
                        {
                            LogRangeAsType<float>(std::cout << "a : ", a_tensors[i].mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(std::cout << "b: ", b_tensors[i].mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(std::cout << "d0: ", d0_tensors[i].mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(
                                std::cout << "e_device: ", e_device_tensors[i].mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(
                                std::cout << "e_host  : ", e_host_tensors[i].mData, ",")
                                << std::endl;
                        }
                    }

                    std::cout << "Instance: " << gemm_name << "; KBatch: " << kbatch_curr << " "
                              << (instance_pass ? "SUCCEED" : "FAILED") << std::endl;

                    pass = pass && instance_pass;
                }

                float ave_time = invoker_ptr->Run(
                    argument_ptr.get(), StreamConfig{nullptr, time_kernel, 0, n_warmup, n_iter});

                if(time_kernel)
                {
                    std::size_t flop = 0, num_btype = 0;
                    for(std::size_t i = 0; i < gemm_descs.size(); i++)
                    {
                        flop += std::size_t(2) * Ms[i] * Ns[i] * Ks[i];

                        num_btype +=
                            sizeof(ADataType) * Ms[i] * Ks[i] + sizeof(BDataType) * Ks[i] * Ns[i] +
                            sizeof(D0DataType) * Ms[i] * Ns[i] + sizeof(EDataType) * Ms[i] * Ns[i];
                    }

                    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                    float gb_per_sec = num_btype / 1.E6 / ave_time;
                    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops
                              << " TFlops, " << gb_per_sec << " GB/s, " << gemm_name << ", KBatch "
                              << kbatch_curr << std::endl;

                    if(tflops > best_tflops)
                    {
                        best_gemm_name  = gemm_name;
                        best_tflops     = tflops;
                        best_ave_time   = ave_time;
                        best_gb_per_sec = gb_per_sec;
                        best_kbatch     = kbatch_curr;
                    }
                }
            }
            else
            {
                std::cout << "Instance: " << gemm_name
                          << ", does not support this GEMM problem (KBatch: " << kbatch_curr << ")"
                          << std::endl;
            }
        }
    }

    if(time_kernel)
    {
        std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
                  << best_gb_per_sec << " GB/s, " << best_gemm_name << ", KBatch = " << best_kbatch
                  << std::endl;
    }
    return pass;
}

} // namespace profiler
} // namespace ck
