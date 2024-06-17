// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop_multiply.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename DLayout,
          typename ELayout>
bool profile_grouped_gemm_multiply_tile_loop_impl(int do_verification,
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
                                                  int n_warmup = 10,
                                                  int n_iter   = 50)
{
    using CDataType = EDataType;
    bool pass       = true;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    std::size_t group_count = Ms.size();

    if(!(group_count == Ns.size() && group_count == Ks.size() && group_count == StrideAs.size() &&
         group_count == StrideBs.size() && group_count == StrideEs.size()))
    {
        throw std::runtime_error("wrong! inconsistent M/N/Ks, StrideA/B/Cs size\n");
    }

    std::vector<Tensor<ADataType>> a_m_k;
    std::vector<Tensor<BDataType>> b_k_n;
    std::vector<Tensor<DDataType>> d_m_n;
    std::vector<Tensor<CDataType>> e_m_n_host_results;
    std::vector<Tensor<CDataType>> e_m_n_device_results;

    for(std::size_t i = 0; i < group_count; i++)
    {
        a_m_k.push_back(
            Tensor<ADataType>(f_host_tensor_descriptor(Ms[i], Ks[i], StrideAs[i], ALayout{})));
        b_k_n.push_back(
            Tensor<BDataType>(f_host_tensor_descriptor(Ks[i], Ns[i], StrideBs[i], BLayout{})));
        d_m_n.push_back(
            Tensor<DDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideDs[i], DLayout{})));
        e_m_n_device_results.push_back(
            Tensor<CDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideEs[i], ELayout{})));
        e_m_n_host_results.push_back(
            Tensor<CDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideEs[i], ELayout{})));
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            std::cout << "group: " << i << " a_m_k[" << i << "]:" << a_m_k[i].mDesc << ", b_k_n["
                      << i << "]:" << b_k_n[i].mDesc << ", e_m_n_device_results[" << i
                      << "]:" << e_m_n_device_results[i].mDesc << std::endl;
        }
        switch(init_method)
        {
        case 0: break;
        case 1:
            ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5, 5}(a_m_k[i]);
            ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5, 5}(b_k_n[i]);
            ck::utils::FillUniformDistributionIntegerValue<DDataType>{-5, 5}(d_m_n[i]);
            break;
        case 2:
            ck::utils::FillUniformDistribution<ADataType>{.0, 1.}(a_m_k[i]);
            ck::utils::FillUniformDistribution<BDataType>{-0.5, 0.5}(b_k_n[i]);
            ck::utils::FillUniformDistribution<DDataType>{-0.5, 0.5}(d_m_n[i]);
            break;
        default:
            ck::utils::FillConstant<ADataType>{1}(a_m_k[i]);
            ck::utils::FillConstant<BDataType>{1}(b_k_n[i]);
            ck::utils::FillConstant<DDataType>{1}(d_m_n[i]);
        }
    }

    using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp   = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementOp = ck::tensor_operation::element_wise::Multiply;

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto c_element_op   = CElementOp{};
    const auto cde_element_op = CDEElementOp{};

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<DeviceMemPtr> a_device_buf, b_device_buf, d_device_buf, e_device_buf;

    a_device_buf.reserve(group_count);
    b_device_buf.reserve(group_count);
    d_device_buf.reserve(group_count);
    e_device_buf.reserve(group_count);

    std::vector<const void*> p_a, p_b, p_d;
    constexpr ck::index_t NumDTensor = 1;
    auto p_ds                        = std::vector<std::array<const void*, NumDTensor>>{};
    std::vector<void*> p_e;

    p_a.reserve(group_count);
    p_b.reserve(group_count);
    p_ds.reserve(group_count);
    p_e.reserve(group_count);

    using KernelArguments =
        ck::tensor_operation::device::GroupedGemmTileLoopKernelArguments<NumDTensor>;

    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    std::vector<KernelArguments> gemm_kargs;

    gemm_descs.reserve(group_count);
    gemm_kargs.reserve(group_count);

    for(std::size_t i = 0; i < group_count; i++)
    {
        a_device_buf.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ADataType) * a_m_k[i].mDesc.GetElementSpaceSize()));
        b_device_buf.emplace_back(
            std::make_unique<DeviceMem>(sizeof(BDataType) * b_k_n[i].mDesc.GetElementSpaceSize()));
        d_device_buf.emplace_back(
            std::make_unique<DeviceMem>(sizeof(DDataType) * d_m_n[i].mDesc.GetElementSpaceSize()));
        e_device_buf.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * e_m_n_device_results[i].mDesc.GetElementSpaceSize()));

        a_device_buf[i]->ToDevice(a_m_k[i].mData.data());
        b_device_buf[i]->ToDevice(b_k_n[i].mData.data());
        d_device_buf[i]->ToDevice(d_m_n[i].mData.data());
        e_device_buf[i]->SetZero();

        p_a.push_back(a_device_buf[i]->GetDeviceBuffer());
        p_b.push_back(b_device_buf[i]->GetDeviceBuffer());
        p_ds.push_back({d_device_buf[i]->GetDeviceBuffer()});
        p_e.push_back(e_device_buf[i]->GetDeviceBuffer());

        gemm_descs.push_back(
            {0, Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideEs[i], {StrideDs[i]}});
        gemm_kargs.push_back({a_device_buf[i]->GetDeviceBuffer(),
                              b_device_buf[i]->GetDeviceBuffer(),
                              {d_device_buf[i]->GetDeviceBuffer()},
                              e_device_buf[i]->GetDeviceBuffer(),
                              Ms[i],
                              Ns[i],
                              Ks[i],
                              StrideAs[i],
                              StrideBs[i],
                              {StrideDs[i]},
                              StrideEs[i]});
    }

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemmTileLoop<ALayout,
                                                                             BLayout,
                                                                             ck::Tuple<DLayout>,
                                                                             ELayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             ck::Tuple<DDataType>,
                                                                             EDataType,
                                                                             AElementOp,
                                                                             BElementOp,
                                                                             CDEElementOp>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    if(op_ptrs.size() <= 0)
    {
        throw std::runtime_error("wrong! no device GEMM instance found");
    }

    std::string best_gemm_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    if(do_verification)
    {
        for(std::size_t i = 0; i < gemm_descs.size(); i++)
        {
            Tensor<CDataType> c_m_n({Ms[i], Ns[i]});

            using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                    BDataType,
                                                                                    CDataType,
                                                                                    AccDataType,
                                                                                    AElementOp,
                                                                                    BElementOp,
                                                                                    CElementOp>;

            auto ref_gemm     = ReferenceGemmInstance{};
            auto ref_invoker  = ref_gemm.MakeInvoker();
            auto ref_argument = ref_gemm.MakeArgument(
                a_m_k[i], b_k_n[i], c_m_n, a_element_op, b_element_op, c_element_op);
            ref_invoker.Run(ref_argument);

            for(int m = 0; m < Ms[i]; ++m)
            {
                for(int n = 0; n < Ns[i]; ++n)
                {
                    cde_element_op(e_m_n_host_results[i](m, n), c_m_n(m, n), d_m_n[i](m, n));
                }
            }
        }
    }

    // profile device GEMM instances
    for(auto& gemm_ptr : op_ptrs)
    {
        auto argument_ptr =
            gemm_ptr->MakeArgumentPointer(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          gemm_descs,
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          cde_element_op);
        auto invoker_ptr      = gemm_ptr->MakeInvokerPointer();
        std::string gemm_name = gemm_ptr->GetTypeString();

        DeviceMem gemm_arg_dev_mem(gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()));
        hip_check_error(hipMemcpy(gemm_arg_dev_mem.GetDeviceBuffer(),
                                  gemm_kargs.data(),
                                  gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()),
                                  hipMemcpyHostToDevice));
        gemm_ptr->SetDeviceKernelArgs(argument_ptr.get(), gemm_arg_dev_mem.GetDeviceBuffer());

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false, 0, n_warmup, n_iter});
            if(do_verification)
            {
                bool instance_pass = true;
                for(std::size_t i = 0; i < gemm_descs.size(); i++)
                {
                    e_device_buf[i]->FromDevice(e_m_n_device_results[i].mData.data());
                    instance_pass = instance_pass && ck::utils::check_err(e_m_n_device_results[i],
                                                                          e_m_n_host_results[i]);

                    if(do_log)
                    {
                        LogRangeAsType<float>(std::cout << "a : ", a_m_k[i].mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(std::cout << "b: ", b_k_n[i].mData, ",") << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "e_device: ", e_m_n_device_results[i].mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "e_host  : ", e_m_n_host_results[i].mData, ",")
                            << std::endl;
                    }
                }

                std::cout << "Instance: " << gemm_name << " verification "
                          << (instance_pass ? "SUCCEED" : "FAILED") << std::endl;

                pass = pass && instance_pass;
            }

            if(time_kernel)
            {
                float ave_time = invoker_ptr->Run(
                    argument_ptr.get(), StreamConfig{nullptr, time_kernel, 0, n_warmup, n_iter});

                std::size_t flop = 0, num_btype = 0;
                for(std::size_t i = 0; i < gemm_descs.size(); i++)
                {
                    flop += std::size_t(2) * Ms[i] * Ns[i] * Ks[i];

                    num_btype += sizeof(ADataType) * Ms[i] * Ks[i] +
                                 sizeof(BDataType) * Ks[i] * Ns[i] +
                                 sizeof(EDataType) * Ms[i] * Ns[i] + // D matrix
                                 sizeof(EDataType) * Ms[i] * Ns[i];
                }

                float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
                float gb_per_sec = num_btype / 1.E6 / ave_time;
                std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << gemm_name << std::endl;

                if(tflops > best_tflops)
                {
                    best_gemm_name  = gemm_name;
                    best_tflops     = tflops;
                    best_ave_time   = ave_time;
                    best_gb_per_sec = gb_per_sec;
                }
            }
        }
        else
        {
            std::cout << "Instance: " << gemm_name << ", does not support this GEMM problem"
                      << std::endl;
        }
    }

    if(time_kernel)
    {
        std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
                  << best_gb_per_sec << " GB/s, " << best_gemm_name << std::endl;
    }

    return pass;
}

} // namespace profiler
} // namespace ck
