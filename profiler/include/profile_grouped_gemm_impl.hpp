// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/conv_util.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DeviceGroupedGemmNoOpPtr = ck::tensor_operation::device::DeviceGroupedGemmPtr<
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(
    std::vector<DeviceGroupedGemmNoOpPtr>&);
void add_device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(
    std::vector<DeviceGroupedGemmNoOpPtr>&);
void add_device_grouped_gemm_xdl_f16_f16_f16_km_kn_mn_instances(
    std::vector<DeviceGroupedGemmNoOpPtr>&);
void add_device_grouped_gemm_xdl_f16_f16_f16_km_nk_mn_instances(
    std::vector<DeviceGroupedGemmNoOpPtr>&);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
void profile_grouped_gemm_impl(int do_verification,
                               int init_method,
                               bool do_log,
                               bool time_kernel,
                               const std::vector<int>& Ms,
                               const std::vector<int>& Ns,
                               const std::vector<int>& Ks,
                               const std::vector<int>& StrideAs,
                               const std::vector<int>& StrideBs,
                               const std::vector<int>& StrideCs)
{
    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    std::size_t group_count = Ms.size();

    if(!(group_count == Ns.size() && group_count == Ks.size() && group_count == StrideAs.size() &&
         group_count == StrideBs.size() && group_count == StrideCs.size()))
    {
        throw std::runtime_error("wrong! inconsistent M/N/Ks, StrideA/B/Cs size\n");
    }

    std::vector<Tensor<ADataType>> a_m_k;
    std::vector<Tensor<BDataType>> b_k_n;
    std::vector<Tensor<CDataType>> c_m_n_device_results;

    for(std::size_t i = 0; i < group_count; i++)
    {
        a_m_k.push_back(
            Tensor<ADataType>(f_host_tensor_descriptor(Ms[i], Ks[i], StrideAs[i], ALayout{})));
        b_k_n.push_back(
            Tensor<BDataType>(f_host_tensor_descriptor(Ks[i], Ns[i], StrideBs[i], BLayout{})));

        c_m_n_device_results.push_back(
            Tensor<CDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideCs[i], CLayout{})));

        std::cout << "group: " << i << " a_m_k[" << i << "]:" << a_m_k[i].mDesc << ", b_k_n[" << i
                  << "]:" << b_k_n[i].mDesc << ", c_m_n_device_results[" << i
                  << "]:" << c_m_n_device_results[i].mDesc << std::endl;

        std::size_t num_thread = 1;
        switch(init_method)
        {
        case 0: break;
        case 1:
            a_m_k[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
            b_k_n[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
            break;
        default:
            a_m_k[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
            b_k_n[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
        }

        c_m_n_device_results[i].GenerateTensorValue(GeneratorTensor_0<CDataType>{}, num_thread);
    }

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    // if(do_verification)
    // {

    // }

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<DeviceMemPtr> a_device_buf, b_device_buf, c_device_buf;

    a_device_buf.reserve(group_count);
    b_device_buf.reserve(group_count);
    c_device_buf.reserve(group_count);

    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_c;

    p_a.reserve(group_count);
    p_b.reserve(group_count);
    p_c.reserve(group_count);

    std::vector<ck::tensor_operation::device::GemmShape> gemm_shapes;

    gemm_shapes.reserve(group_count);

    for(std::size_t i = 0; i < group_count; i++)
    {
        a_device_buf.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ADataType) * a_m_k[i].mDesc.GetElementSpace()));
        b_device_buf.emplace_back(
            std::make_unique<DeviceMem>(sizeof(BDataType) * b_k_n[i].mDesc.GetElementSpace()));

        c_device_buf.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * c_m_n_device_results[i].mDesc.GetElementSpace()));

        a_device_buf[i]->ToDevice(a_m_k[i].mData.data());
        b_device_buf[i]->ToDevice(b_k_n[i].mData.data());
        c_device_buf[i]->ToDevice(c_m_n_device_results[i].mData.data());

        gemm_shapes.push_back({Ms[i], Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideCs[i]});

        p_a.push_back(a_device_buf[i]->GetDeviceBuffer());
        p_b.push_back(b_device_buf[i]->GetDeviceBuffer());
        p_c.push_back(c_device_buf[i]->GetDeviceBuffer());
    }

    // add device GEMM instances
    std::vector<ck::tensor_operation::device::instance::DeviceGroupedGemmNoOpPtr> gemm_ptrs;

    if constexpr(is_same<ADataType, half_t>::value && is_same<BDataType, half_t>::value &&
                 is_same<CDataType, half_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_grouped_gemm_xdl_f16_f16_f16_km_kn_mn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_grouped_gemm_xdl_f16_f16_f16_km_nk_mn_instances(gemm_ptrs);
        }
    }

    if(gemm_ptrs.size() <= 0)
    {
        throw std::runtime_error("wrong! no device GEMM instance found");
    }

    std::string best_gemm_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device GEMM instances
    for(auto& gemm_ptr : gemm_ptrs)
    {
        auto argument_ptr =
            gemm_ptr->MakeArgumentPointer(p_a,
                                          p_b,
                                          p_c,
                                          gemm_shapes,
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{});

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        DeviceMem gemm_desc_workspace(gemm_ptr->GetWorkSpaceSize(argument_ptr.get()));

        gemm_ptr->SetWorkSpacePointer(argument_ptr.get(), gemm_desc_workspace.GetDeviceBuffer());

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string gemm_name = gemm_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = 0, num_btype = 0;
            for(std::size_t i = 0; i < gemm_shapes.size(); i++)
            {
                flop += std::size_t(2) * Ms[i] * Ns[i] * Ks[i];

                num_btype += sizeof(ADataType) * Ms[i] * Ks[i] + sizeof(BDataType) * Ks[i] * Ns[i] +
                             sizeof(CDataType) * Ms[i] * Ns[i];
            }

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;
            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << gemm_name << std::endl;

            if(tflops > best_tflops)
            {
                best_gemm_name  = gemm_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                for(std::size_t i = 0; i < gemm_shapes.size(); i++)
                {

                    c_device_buf[i]->FromDevice(c_m_n_device_results[i].mData.data());

                    Tensor<CDataType> c_m_n_host_result(
                        f_host_tensor_descriptor(Ms[i], Ns[i], StrideCs[i], CLayout{}));

                    using ReferenceGemmInstance =
                        ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                  BDataType,
                                                                  CDataType,
                                                                  AccDataType,
                                                                  AElementOp,
                                                                  BElementOp,
                                                                  CElementOp>;

                    auto ref_gemm    = ReferenceGemmInstance{};
                    auto ref_invoker = ref_gemm.MakeInvoker();

                    auto ref_argument = ref_gemm.MakeArgument(a_m_k[i],
                                                              b_k_n[i],
                                                              c_m_n_host_result,
                                                              a_element_op,
                                                              b_element_op,
                                                              c_element_op);

                    ref_invoker.Run(ref_argument);
                    ck::utils::check_err(c_m_n_device_results[i].mData, c_m_n_host_result.mData);

                    if(do_log)
                    {
                        LogRangeAsType<float>(std::cout << "a : ", a_m_k[i].mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(std::cout << "b: ", b_k_n[i].mData, ",") << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "c_device: ", c_m_n_device_results[i].mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                            << std::endl;
                    }
                }
            }
        }
        else
        {
            std::cout << "does not support this GEMM problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_gemm_name << std::endl;
} // namespace profiler

} // namespace profiler
} // namespace ck
