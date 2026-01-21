// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <array>
#include <vector>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm_multi_abd.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_multi_abd_fixed_nk.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/fill.hpp"

namespace ck {
namespace profiler {

template <typename T>
auto reserveVector(std::size_t size)
{
    std::vector<T> vec;
    vec.reserve(size);
    return vec;
}

template <typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          typename AccDataType,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementOp   = ck::tensor_operation::element_wise::PassThrough,
          typename BElementOp   = ck::tensor_operation::element_wise::Multiply,
          typename CDEElementOp = ck::tensor_operation::element_wise::PassThrough>
bool profile_grouped_gemm_multi_abd_fixed_nk_impl(int do_verification,
                                                  int init_method,
                                                  bool do_log,
                                                  bool time_kernel,
                                                  const std::vector<int>& Ms,
                                                  const std::vector<int>& Ns,
                                                  const std::vector<int>& Ks,
                                                  const std::vector<int>& StrideAs,
                                                  const std::vector<int>& StrideBs,
                                                  const std::vector<int>& StrideDs,
                                                  const std::vector<int>& StrideE,
                                                  const std::vector<int>& kbatch_list = {1},
                                                  int n_warmup                        = 1,
                                                  int n_iter                          = 10)
{
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

    const std::size_t group_count = Ms.size();
    const int sum_of_m            = std::accumulate(Ms.begin(), Ms.end(), 0);

    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    if(group_count != Ns.size() || group_count != Ks.size() || group_count != StrideAs.size() ||
       group_count != StrideBs.size() || (NumDTensor > 0 && group_count != StrideDs.size()))
    {
        throw std::runtime_error("wrong! inconsistent M/N/Ks, StrideAs/Bs/Ds/E size\n");
    }

    auto generateInputTupleA = [&](std::size_t g) {
        if constexpr(NumATensor == 0)
        {
            static_assert("Gemm problem should have at least 1 A tensor.");
        }
        else
        {
            using ALayout = remove_cvref_t<tuple_element_t<Number<0>{}, AsLayout>>;
            return generate_tuple(
                [&](auto i) {
                    using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                    return Tensor<ADataType>(
                        f_host_tensor_descriptor(Ms[g], Ks[g], StrideAs[g], ALayout{}));
                },
                Number<NumATensor>{});
        }
    };
    auto generateInputTupleB = [&](std::size_t g) {
        if constexpr(NumBTensor == 0)
        {
            static_assert("Gemm problem should have at least 1 B tensor.");
        }
        else
        {
            using BLayout = remove_cvref_t<tuple_element_t<Number<0>{}, BsLayout>>;
            return generate_tuple(
                [&](auto i) {
                    using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                    return Tensor<BDataType>(
                        f_host_tensor_descriptor(Ks[g], Ns[g], StrideBs[g], BLayout{}));
                },
                Number<NumBTensor>{});
        }
    };
    auto generateInputTupleD = [&](std::size_t g) {
        if constexpr(NumDTensor == 0)
        {
            return ck::Tuple<>();
        }
        else
        {
            using DLayout = remove_cvref_t<tuple_element_t<Number<0>{}, DsLayout>>;
            return generate_tuple(
                [&](auto i) {
                    using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                    return Tensor<DDataType>(
                        f_host_tensor_descriptor(Ms[g], Ns[g], StrideDs[g], DLayout{}));
                },
                Number<NumDTensor>{});
        }
    };

    using AsTensorTuple = decltype(generateInputTupleA(0));
    using BsTensorTuple = decltype(generateInputTupleB(0));
    using DsTensorTuple = decltype(generateInputTupleD(0));

    auto g_as_m_k               = reserveVector<AsTensorTuple>(group_count);
    auto g_bs_k_n               = reserveVector<BsTensorTuple>(group_count);
    auto g_ds_m_n               = reserveVector<DsTensorTuple>(group_count);
    auto g_e_m_n_host_results   = reserveVector<Tensor<EDataType>>(group_count);
    auto g_e_m_n_device_results = reserveVector<Tensor<EDataType>>(group_count);

    for(std::size_t g = 0; g < group_count; g++)
    {
        auto& as_m_k = g_as_m_k.emplace_back(generateInputTupleA(g));
        auto& bs_k_n = g_bs_k_n.emplace_back(generateInputTupleB(g));
        auto& ds_m_n = g_ds_m_n.emplace_back(generateInputTupleD(g));

        g_e_m_n_host_results.push_back(
            Tensor<EDataType>(f_host_tensor_descriptor(Ms[g], Ns[g], StrideE[g], ELayout{})));
        g_e_m_n_device_results.push_back(
            Tensor<EDataType>(f_host_tensor_descriptor(Ms[g], Ns[g], StrideE[g], ELayout{})));

        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            std::cout << "group: " << g << std::endl;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                std::cout << "a" << i.value << "_m_k: " << as_m_k(i).mDesc << std::endl;
            });
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                std::cout << "b" << i.value << "_k_n: " << bs_k_n(i).mDesc << std::endl;
            });
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                std::cout << "d" << i.value << "_m_n: " << ds_m_n(i).mDesc << std::endl;
            });
            std::cout << "e_m_n: " << g_e_m_n_device_results[g].mDesc << std::endl;
        }

        std::size_t num_thread = 1;
        switch(init_method)
        {
        case 0: break;
        case 1:
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                as_m_k(i).GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
            });

            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                bs_k_n(i).GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
            });

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                ds_m_n(i).GenerateTensorValue(GeneratorTensor_2<DDataType>{-5, 5}, num_thread);
            });

            break;
        default:
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                as_m_k(i).GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
            });

            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                bs_k_n(i).GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
            });

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                ds_m_n(i).GenerateTensorValue(GeneratorTensor_3<DDataType>{0.0, 1.0}, num_thread);
            });
        }
    }

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{};

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<std::array<DeviceMemPtr, NumATensor>> g_as_device_buf(group_count);
    std::vector<std::array<DeviceMemPtr, NumBTensor>> g_bs_device_buf(group_count);
    std::vector<std::array<DeviceMemPtr, NumDTensor>> g_ds_device_buf(group_count);
    std::vector<DeviceMemPtr> g_e_device_buf(group_count);

    std::vector<std::array<const void*, NumATensor>> g_as_device_view(group_count);
    std::vector<std::array<const void*, NumBTensor>> g_bs_device_view(group_count);
    std::vector<std::array<const void*, NumDTensor>> g_ds_device_view(group_count);
    std::vector<void*> g_e_device_view(group_count);

    auto g_gemm_descs = reserveVector<tensor_operation::device::GemmMultiABDDesc>(group_count);

    auto grouped_gemm_kernel_args_host =
        reserveVector<tensor_operation::device::
                          GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>>(
            group_count);

    for(std::size_t g = 0; g < group_count; g++)
    {
        std::array<ck::index_t, NumATensor> as_stride;
        std::array<ck::index_t, NumBTensor> bs_stride;
        std::array<ck::index_t, NumDTensor> ds_stride;

        auto& as_m_k         = g_as_m_k[g];
        auto& as_device_buf  = g_as_device_buf[g];
        auto& as_device_view = g_as_device_view[g];

        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType  = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
            as_device_buf[i] = std::make_unique<DeviceMem>(sizeof(ADataType) * Ms[g] * Ks[g]);
            as_device_buf[i]->ToDevice(as_m_k[i].mData.data());
            as_device_view[i] = as_device_buf[i]->GetDeviceBuffer();
            as_stride[i]      = StrideAs[g];
        });

        auto& bs_k_n         = g_bs_k_n[g];
        auto& bs_device_buf  = g_bs_device_buf[g];
        auto& bs_device_view = g_bs_device_view[g];

        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType  = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
            bs_device_buf[i] = std::make_unique<DeviceMem>(sizeof(BDataType) * Ks[g] * Ns[g]);
            bs_device_buf[i]->ToDevice(bs_k_n[i].mData.data());
            bs_device_view[i] = bs_device_buf[i]->GetDeviceBuffer();
            bs_stride[i]      = StrideBs[g];
        });

        auto& ds_m_n         = g_ds_m_n[g];
        auto& ds_device_buf  = g_ds_device_buf[g];
        auto& ds_device_view = g_ds_device_view[g];

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType  = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
            ds_device_buf[i] = std::make_unique<DeviceMem>(sizeof(DDataType) * Ms[g] * Ns[g]);
            ds_device_buf[i]->ToDevice(ds_m_n[i].mData.data());
            ds_device_view[i] = ds_device_buf[i]->GetDeviceBuffer();
            ds_stride[i]      = StrideDs[g];
        });

        g_e_device_buf[g]  = std::make_unique<DeviceMem>(sizeof(EDataType) * Ms[g] * Ns[g]);
        g_e_device_view[g] = g_e_device_buf[g]->GetDeviceBuffer();

        g_gemm_descs.push_back(tensor_operation::device::GemmMultiABDDesc{
            sum_of_m,
            Ns[g],
            Ks[g],
            std::vector<ck::index_t>(as_stride.begin(), as_stride.end()),
            std::vector<ck::index_t>(bs_stride.begin(), bs_stride.end()),
            std::vector<ck::index_t>(ds_stride.begin(), ds_stride.end()),
            StrideE[g]});

        tensor_operation::device::
            GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>
                kernelArg{as_device_view,
                          bs_device_view,
                          ds_device_view,
                          g_e_device_view[g],
                          Ms[g],
                          Ns[g],
                          Ks[g],
                          as_stride,
                          bs_stride,
                          ds_stride,
                          StrideE[g]};

        grouped_gemm_kernel_args_host.push_back(std::move(kernelArg));
    }

    using DeviceOp = tensor_operation::device::DeviceGroupedGemmMultiABDFixedNK<AsLayout,
                                                                                BsLayout,
                                                                                DsLayout,
                                                                                ELayout,
                                                                                AsDataType,
                                                                                BsDataType,
                                                                                DsDataType,
                                                                                EDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                CDEElementOp>;

    const auto op_ptrs = tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    if(op_ptrs.size() <= 0)
    {
        throw std::runtime_error("wrong! no device GEMM instance found");
    }

    std::string best_gemm_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;

    if(do_verification)
    {
        using AComputeType =
            typename std::conditional<(NumATensor > 1),
                                      EDataType,
                                      remove_cvref_t<tuple_element_t<0, AsDataType>>>::type;

        using BComputeType =
            typename std::conditional<(NumBTensor > 1),
                                      EDataType,
                                      remove_cvref_t<tuple_element_t<0, BsDataType>>>::type;

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemmMultiABD<AsTensorTuple,
                                                              BsTensorTuple,
                                                              DsTensorTuple,
                                                              EDataType,
                                                              AccDataType,
                                                              AElementOp,
                                                              BElementOp,
                                                              CDEElementOp,
                                                              AComputeType,
                                                              BComputeType>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        for(std::size_t i = 0; i < group_count; i++)
        {
            auto ref_argument = ref_gemm.MakeArgument(g_as_m_k[i],
                                                      g_bs_k_n[i],
                                                      g_ds_m_n[i],
                                                      g_e_m_n_host_results[i],
                                                      a_element_op,
                                                      b_element_op,
                                                      cde_element_op);

            ref_invoker.Run(ref_argument);
        }
    }

    // profile device GEMM instances
    for(auto& gemm_ptr : op_ptrs)
    {
        auto argument_ptr = gemm_ptr->MakeArgumentPointer(
            g_as_device_view, g_bs_device_view, g_ds_device_view, g_e_device_view, g_gemm_descs);

        if(!gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Gemm incompatible with runtime set parameters. Skipping..."
                          << std::endl;
            }

            continue;
        }

        DeviceMem gemm_workspace_dev(gemm_ptr->GetWorkSpaceSize(argument_ptr.get()));
        gemm_ptr->SetWorkSpacePointer(argument_ptr.get(), gemm_workspace_dev.GetDeviceBuffer());

        DeviceMem grouped_gemm_kernel_args_dev(
            gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()));
        hipGetErrorString(hipMemcpy(grouped_gemm_kernel_args_dev.GetDeviceBuffer(),
                                    grouped_gemm_kernel_args_host.data(),
                                    gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()),
                                    hipMemcpyHostToDevice));

        gemm_ptr->SetDeviceKernelArgs(argument_ptr.get(),
                                      grouped_gemm_kernel_args_dev.GetDeviceBuffer());
        gemm_ptr->SetElementwiseOps(argument_ptr.get(), a_element_op, b_element_op, cde_element_op);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        std::string gemm_name = gemm_ptr->GetTypeString();

        for(const auto kbatch_curr : kbatch_list)
        {
            gemm_ptr->SetKBatch(argument_ptr.get(), kbatch_curr);

            if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                for(std::size_t g = 0; g < group_count; g++)
                {
                    g_e_device_buf[g]->SetZero();
                }

                float ave_time = invoker_ptr->Run(
                    argument_ptr.get(), StreamConfig{nullptr, time_kernel, 0, n_warmup, n_iter});

                if(do_verification)
                {
                    bool instance_pass = true;
                    for(std::size_t g = 0; g < group_count; g++)
                    {
                        g_e_device_buf[g]->FromDevice(
                            g_e_m_n_device_results[g].mData.data(),
                            g_e_m_n_device_results[g].mDesc.GetElementSize() * sizeof(EDataType));

                        instance_pass =
                            instance_pass && ck::utils::check_err(g_e_m_n_device_results[g],
                                                                  g_e_m_n_host_results[g]);

                        if(do_log)
                        {
                            static_for<0, NumATensor, 1>{}([&](auto i) {
                                LogRangeAsType<float>(
                                    std::cout << "a[" << g << "]: ", g_as_m_k[g](i).mData, ",")
                                    << std::endl;
                            });
                            static_for<0, NumBTensor, 1>{}([&](auto i) {
                                LogRangeAsType<float>(
                                    std::cout << "b[" << g << "]: ", g_bs_k_n[g](i).mData, ",")
                                    << std::endl;
                            });
                            static_for<0, NumDTensor, 1>{}([&](auto i) {
                                LogRangeAsType<float>(
                                    std::cout << "d[" << g << "]: ", g_ds_m_n[g](i).mData, ",")
                                    << std::endl;
                            });
                            LogRangeAsType<float>(
                                std::cout << "e_device: ", g_e_m_n_device_results[g].mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(
                                std::cout << "e_host  : ", g_e_m_n_host_results[g].mData, ",")
                                << std::endl;
                        }
                    }

                    std::cout << "Instance: " << gemm_name << " verification "
                              << (instance_pass ? "SUCCEED" : "FAILED") << std::endl;

                    pass = pass && instance_pass;
                }

                if(time_kernel)
                {
                    std::size_t flop = 0, num_btype = 0;
                    for(std::size_t g = 0; g < group_count; g++)
                    {
                        flop += std::size_t(2) * Ms[g] * Ns[g] * Ks[g];

                        static_for<0, NumATensor, 1>{}([&](auto i) {
                            using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                            num_btype += sizeof(ADataType) * Ms[g] * Ks[g];
                        });
                        static_for<0, NumBTensor, 1>{}([&](auto i) {
                            using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                            num_btype += sizeof(BDataType) * Ks[g] * Ns[g];
                        });
                        static_for<0, NumDTensor, 1>{}([&](auto i) {
                            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                            num_btype += sizeof(DDataType) * Ms[g] * Ns[g];
                        });
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
                std::cout << "Instance: " << gemm_name << ", does not support this GEMM problem"
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
