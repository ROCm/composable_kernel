// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_tile_loop.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop_multiply.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm_multiple_d.hpp"
#include "ck/utility/integral_constant.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/tuple_helper.hpp"

namespace ck {
namespace profiler {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <class F, std::size_t... I>
constexpr auto make_array_from_fn_impl(F&& f, std::index_sequence<I...>)
{
    using T = std::decay_t<decltype(f(std::integral_constant<std::size_t, 0>{}))>;
    return std::array<T, sizeof...(I)>{f(std::integral_constant<std::size_t, I>{})...};
}

template <std::size_t N, class F>
constexpr auto make_array_from_fn(F&& f)
{
    return make_array_from_fn_impl(std::forward<F>(f), std::make_index_sequence<N>{});
}

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementOp   = PassThrough,
          typename BElementOp   = PassThrough,
          typename CDEElementOp = PassThrough>
bool profile_grouped_gemm_tile_loop_generic_impl(
    int do_verification,
    int init_method,
    bool do_log,
    bool time_kernel,
    const std::vector<int>& Ms,
    const std::vector<int>& Ns,
    const std::vector<int>& Ks,
    const std::vector<int>& StrideAs,
    const std::vector<int>& StrideBs,
    const std::vector<std::array<int, DsDataType::Size()>>& StrideDs,
    const std::vector<int>& StrideEs,
    int n_warmup = 10,
    int n_iter   = 50)
{
    using AccDataType                = float;
    constexpr ck::index_t NumDTensor = DsDataType::Size();

    static_assert(DsLayout::Size() == DsDataType::Size(), "wrong! inconsistent NumDTensor");

    bool pass = true;

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
         group_count == StrideBs.size() &&
         ((StrideDs.size() == 0 && NumDTensor == 0) || group_count == StrideDs.size()) &&
         group_count == StrideEs.size()))
    {
        throw std::runtime_error("wrong! inconsistent M/N/Ks, StrideA/B/D/Es size\n");
    }

    std::vector<Tensor<ADataType>> a_m_k;
    std::vector<Tensor<BDataType>> b_k_n;
    std::vector<tuple_map_t<Tensor, DsDataType>> d_m_n;
    std::vector<Tensor<EDataType>> e_m_n_host_results;
    std::vector<Tensor<EDataType>> e_m_n_device_results;

    for(std::size_t i = 0; i < group_count; i++)
    {
        a_m_k.push_back(
            Tensor<ADataType>(f_host_tensor_descriptor(Ms[i], Ks[i], StrideAs[i], ALayout{})));
        b_k_n.push_back(
            Tensor<BDataType>(f_host_tensor_descriptor(Ks[i], Ns[i], StrideBs[i], BLayout{})));

        auto d_tensors = ck::generate_tuple(
            [&](auto j) {
                using DDataType = tuple_element_t<j, DsDataType>;

                return Tensor<DDataType>(f_host_tensor_descriptor(
                    Ms[i], Ns[i], StrideDs[i][j], tuple_element_t<j, DsLayout>{}));
            },
            Number<NumDTensor>{});
        d_m_n.emplace_back(d_tensors);

        e_m_n_device_results.push_back(
            Tensor<EDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideEs[i], ELayout{})));
        e_m_n_host_results.push_back(
            Tensor<EDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideEs[i], ELayout{})));
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
            a_m_k[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
            b_k_n[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
            static_for<0, NumDTensor, 1>{}([&](auto j) -> void {
                d_m_n[i](j).GenerateTensorValue(
                    GeneratorTensor_2<tuple_element_t<j, DsDataType>>{-5, 5});
            });
            break;
        case 2:
            a_m_k[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b_k_n[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
            static_for<0, NumDTensor, 1>{}([&](auto j) -> void {
                d_m_n[i](j).GenerateTensorValue(
                    GeneratorTensor_3<tuple_element_t<j, DsDataType>>{-0.5, 0.5});
            });
            break;
        default:
            ck::utils::FillConstant<ADataType>{1}(a_m_k[i]);
            ck::utils::FillConstant<BDataType>{1}(b_k_n[i]);
            static_for<0, NumDTensor, 1>{}([&](auto j) -> void {
                ck::utils::FillConstant<tuple_element_t<j, DsDataType>>{1}(d_m_n[i](j));
            });
        }
    }
    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{};

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<DeviceMemPtr> a_device_buf, b_device_buf, e_device_buf;
    std::vector<std::array<DeviceMemPtr, NumDTensor>> d_device_bufs;

    a_device_buf.reserve(group_count);
    b_device_buf.reserve(group_count);
    d_device_bufs.reserve(group_count);
    e_device_buf.reserve(group_count);

    std::vector<const void*> p_a, p_b;
    std::vector<std::array<const void*, NumDTensor>> p_ds;
    std::vector<void*> p_e;

    p_a.reserve(group_count);
    p_b.reserve(group_count);
    p_ds.reserve(group_count);
    p_e.reserve(group_count);

    using KernelArguments = ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>;

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

        if constexpr(NumDTensor > 0)
        {
            d_device_bufs.emplace_back(make_array_from_fn<NumDTensor>([&](auto j) {
                return std::make_unique<DeviceMem>(
                    sizeof(tuple_element_t<j, DsDataType>) *
                    d_m_n[i][ck::integral_constant<index_t, j>{}].mDesc.GetElementSpaceSize());
            }));
        }

        e_device_buf.emplace_back(std::make_unique<DeviceMem>(
            sizeof(EDataType) * e_m_n_device_results[i].mDesc.GetElementSpaceSize()));

        a_device_buf[i]->ToDevice(a_m_k[i].mData.data());
        b_device_buf[i]->ToDevice(b_k_n[i].mData.data());

        static_for<0, NumDTensor, 1>{}(
            [&](auto j) -> void { d_device_bufs[i][j]->ToDevice(d_m_n[i][j].mData.data()); });

        e_device_buf[i]->SetZero();

        p_a.push_back(a_device_buf[i]->GetDeviceBuffer());
        p_b.push_back(b_device_buf[i]->GetDeviceBuffer());

        std::array<const void*, NumDTensor> p_d;
        static_for<0, NumDTensor, 1>{}(
            [&](auto j) -> void { p_d[j] = d_device_bufs[i][j]->GetDeviceBuffer(); });

        p_ds.push_back(p_d);

        p_e.push_back(e_device_buf[i]->GetDeviceBuffer());

        gemm_descs.push_back({Ms[i],
                              Ns[i],
                              Ks[i],
                              StrideAs[i],
                              StrideBs[i],
                              StrideEs[i],
                              std::vector<int>(StrideDs[i].begin(), StrideDs[i].end())});
        gemm_kargs.push_back({a_device_buf[i]->GetDeviceBuffer(),
                              b_device_buf[i]->GetDeviceBuffer(),
                              p_d,
                              e_device_buf[i]->GetDeviceBuffer(),
                              Ms[i],
                              Ns[i],
                              Ks[i],
                              StrideAs[i],
                              StrideBs[i],
                              StrideDs[i],
                              StrideEs[i]});
    }

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemmTileLoop<ALayout,
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
            if constexpr(NumDTensor > 0)
            {
                using ReferenceGemmInstance =
                    ck::tensor_operation::host::ReferenceGemmMultipleD<ADataType,
                                                                       BDataType,
                                                                       DsDataType,
                                                                       EDataType,
                                                                       AccDataType,
                                                                       AElementOp,
                                                                       BElementOp,
                                                                       CDEElementOp>;

                // HACK: reference GEMM expects D tensors as std::array
                // This limits D tensors to all have the same data type
                using DDataType = tuple_element_t<0, DsDataType>;
                std::array<Tensor<DDataType>, NumDTensor> d_tensors =
                    make_array_from_fn<NumDTensor>(
                        [&](auto j) { return d_m_n[i][ck::integral_constant<index_t, j>{}]; });

                auto ref_gemm     = ReferenceGemmInstance{};
                auto ref_invoker  = ref_gemm.MakeInvoker();
                auto ref_argument = ref_gemm.MakeArgument(a_m_k[i],
                                                          b_k_n[i],
                                                          d_tensors,
                                                          e_m_n_host_results[i],
                                                          a_element_op,
                                                          b_element_op,
                                                          cde_element_op);
                ref_invoker.Run(ref_argument);
            }
            else
            {

                using ReferenceGemmInstance =
                    ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                              BDataType,
                                                              EDataType,
                                                              AccDataType,
                                                              AElementOp,
                                                              BElementOp,
                                                              CDEElementOp>;

                auto ref_gemm     = ReferenceGemmInstance{};
                auto ref_invoker  = ref_gemm.MakeInvoker();
                auto ref_argument = ref_gemm.MakeArgument(a_m_k[i],
                                                          b_k_n[i],
                                                          e_m_n_host_results[i],
                                                          a_element_op,
                                                          b_element_op,
                                                          cde_element_op);
                ref_invoker.Run(ref_argument);
            }
        }
    }

    // profile device GEMM instances
    for(auto& gemm_ptr : op_ptrs)
    {
        auto argument_ptr = gemm_ptr->MakeArgumentPointer(
            p_a, p_b, p_ds, p_e, gemm_descs, a_element_op, b_element_op, cde_element_op);
        auto invoker_ptr      = gemm_ptr->MakeInvokerPointer();
        std::string gemm_name = gemm_ptr->GetTypeString();

        DeviceMem gemm_arg_dev_mem(gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()));
        ck::hip_check_error(hipMemcpy(gemm_arg_dev_mem.GetDeviceBuffer(),
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
                                 sizeof(EDataType) * Ms[i] * Ns[i];

                    static_for<0, NumDTensor, 1>{}([&](auto j) -> void {
                        num_btype +=
                            sizeof(tuple_element_t<j, DsDataType>) * Ms[i] * Ns[i]; // D matrix
                    });
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
