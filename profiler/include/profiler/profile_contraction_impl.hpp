// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <limits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/contraction_bilinear.hpp"
#include "ck/library/tensor_operation_instance/gpu/contraction_scale.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_contraction.hpp"

#include "ck/host_utility/io.hpp"

namespace ck {
namespace profiler {

using Bilinear = ck::tensor_operation::element_wise::Bilinear;
using Scale    = ck::tensor_operation::element_wise::Scale;

template <typename ALayout,
          typename BLayout,
          typename CDELayout,
          typename DataType,
          typename DTupleDataType,
          typename CDElementOp>
int profile_contraction_impl(ck::index_t do_verification,
                             ck::index_t init_method,
                             bool do_log,
                             bool time_kernel,
                             CDElementOp cde_element_op,
                             const std::vector<ck::index_t>& M,
                             const std::vector<ck::index_t>& N,
                             const std::vector<ck::index_t>& K,
                             const std::vector<ck::index_t>& StridesA,
                             const std::vector<ck::index_t>& StridesB,
                             const std::vector<ck::index_t>& StridesE,
                             const std::vector<ck::index_t>& StridesD)
{
    bool pass = true;

    auto f_host_tensor_descriptor = [](const std::vector<ck::index_t>& dims01,
                                       const std::vector<ck::index_t>& dims23,
                                       const std::vector<ck::index_t>& strides) {
        std::vector<std::size_t> dims_szt(dims01.begin(), dims01.end());
        dims_szt.insert(dims_szt.end(), dims23.begin(), dims23.end());
        std::vector<std::size_t> strides_szt(strides.begin(), strides.end());

        return HostTensorDescriptor(dims_szt, strides);
    };

    Tensor<DataType> a_m_k(f_host_tensor_descriptor(M, K, StridesA));
    Tensor<DataType> b_k_n(f_host_tensor_descriptor(K, N, StridesB));
    Tensor<DataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StridesE));
    Tensor<DataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StridesE));
    Tensor<DataType> d_m_n(f_host_tensor_descriptor(M, N, StridesD));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "d_m_n: " << d_m_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_device_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        d_m_n.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<DataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<DataType>{-0.5, 0.5});
        d_m_n.GenerateTensorValue(GeneratorTensor_3<DataType>{-0.5, 0.5});
    }

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;

    DeviceMem a_device_buf(sizeof(DataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(DataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(DataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DataType) * d_m_n.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    e_device_buf.SetZero();
    d_device_buf.ToDevice(d_m_n.mData.data());

    const std::vector<index_t> a_ms_ks_lengths = {M[0], M[1], K[0], K[1]};
    const std::vector<index_t> b_ns_ks_lengths = {N[0], N[1], K[0], K[1]};
    const std::vector<index_t> e_ms_ns_lengths = {M[0], M[1], N[0], N[1]};
    const std::vector<index_t> d_m_n_lengths   = {M[0], M[1], N[0], N[1]};

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};

    constexpr ck::index_t NumDim = 2;
    using DeviceOp               = ck::tensor_operation::device::DeviceContractionMultipleD<NumDim,
                                                                              NumDim,
                                                                              NumDim,
                                                                              DataType,
                                                                              DataType,
                                                                              DTupleDataType,
                                                                              DataType,
                                                                              AElementOp,
                                                                              BElementOp,
                                                                              CDElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // Run reference op
    if(do_verification)
    {
        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceContraction_M2_N2_K2<NumDim,
                                                                      NumDim,
                                                                      NumDim,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      AElementOp,
                                                                      BElementOp>;

        auto ref_op      = ReferenceGemmInstance{};
        auto ref_invoker = ref_op.MakeInvoker();

        Tensor<DataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StridesE));

        auto ref_argument =
            ref_op.MakeArgument(a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op);

        ref_invoker.Run(ref_argument);

        for(size_t m0 = 0; m0 < e_m_n_host_result.mDesc.GetLengths()[0]; ++m0)
        {
            for(size_t m1 = 0; m1 < e_m_n_host_result.mDesc.GetLengths()[1]; ++m1)
            {
                for(size_t n0 = 0; n0 < e_m_n_host_result.mDesc.GetLengths()[2]; ++n0)
                {
                    for(size_t n1 = 0; n1 < e_m_n_host_result.mDesc.GetLengths()[3]; ++n1)
                    {
                        if constexpr(is_same<CDElementOp, Bilinear>::value)
                        {
                            cde_element_op(e_m_n_host_result(m0, m1, n0, n1),
                                           c_m_n_host_result(m0, m1, n0, n1),
                                           d_m_n(m0, m1, n0, n1));
                        }
                        else if constexpr(is_same<CDElementOp, Scale>::value)
                        {
                            cde_element_op(e_m_n_host_result(m0, m1, n0, n1),
                                           c_m_n_host_result(m0, m1, n0, n1));
                        }
                        else
                        {
                            static_assert("Unsupported CDElementOp in contraction profiler.");
                        }
                    }
                }
            }
        }
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    for(auto& op_ptr : op_ptrs)
    {
        std::unique_ptr<tensor_operation::device::BaseArgument> argument_ptr;
        if constexpr(is_same<CDElementOp, Bilinear>::value)
        {
            argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<DataType*>(a_device_buf.GetDeviceBuffer()),
                static_cast<DataType*>(b_device_buf.GetDeviceBuffer()),
                std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                static_cast<DataType*>(e_device_buf.GetDeviceBuffer()),
                a_ms_ks_lengths,
                StridesA,
                b_ns_ks_lengths,
                StridesB,
                std::array<std::vector<ck::index_t>, 1>{d_m_n_lengths},
                std::array<std::vector<ck::index_t>, 1>{StridesD},
                e_ms_ns_lengths,
                StridesE,
                a_element_op,
                b_element_op,
                cde_element_op);
        }
        else if constexpr(is_same<CDElementOp, Scale>::value)
        {
            argument_ptr =
                op_ptr->MakeArgumentPointer(static_cast<DataType*>(a_device_buf.GetDeviceBuffer()),
                                            static_cast<DataType*>(b_device_buf.GetDeviceBuffer()),
                                            std::array<const void*, 0>{},
                                            static_cast<DataType*>(e_device_buf.GetDeviceBuffer()),
                                            a_ms_ks_lengths,
                                            StridesA,
                                            b_ns_ks_lengths,
                                            StridesB,
                                            std::array<std::vector<ck::index_t>, 0>{},
                                            std::array<std::vector<ck::index_t>, 0>{},
                                            e_ms_ns_lengths,
                                            StridesE,
                                            a_element_op,
                                            b_element_op,
                                            cde_element_op);
        }
        else
        {
            static_assert("Unsupported CDElementOp in contraction profiler.");
        }

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        auto nelems_m = M[0] * M[1];
        auto nelems_n = N[0] * N[1];
        auto nelems_k = K[0] * K[1];

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // re-init C to zero before profiling next kernel
            e_device_buf.SetZero();

            std::string op_name = op_ptr->GetTypeString();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = std::size_t(2) * nelems_m * nelems_n * nelems_k;

            std::size_t num_btype = sizeof(DataType) * nelems_m * nelems_k +
                                    sizeof(DataType) * nelems_k * nelems_n +
                                    sizeof(DataType) * nelems_m * nelems_n;

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                e_device_buf.FromDevice(e_m_n_device_result.mData.data());

                float threshold =
                    static_cast<DataType>(nelems_k) * std::numeric_limits<DataType>::epsilon();
                pass = pass & ck::utils::check_err(e_m_n_device_result,
                                                   e_m_n_host_result,
                                                   "Error: incorrect results!",
                                                   threshold,
                                                   threshold);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host  : ", e_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_device: ", e_m_n_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    if constexpr(is_same<DataType, float>::value)
    {
        std::cout << "Best Perf for datatype = f32";
    }
    else if constexpr(is_same<DataType, double>::value)
    {
        std::cout << "Best Perf for datatype = f64";
    }

    if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " ALayout =  RowMajor";
    }
    else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " ALayout =  ColumnMajor";
    }

    if constexpr(is_same<BLayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " BLayout =  RowMajor";
    }
    else if constexpr(is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " BLayout =  ColumnMajor";
    }

    if constexpr(is_same<CDELayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " CDELayout =  RowMajor";
    }
    else if constexpr(is_same<CDELayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " CDELayout =  ColumnMajor";
    }

    std::cout << " M = " << M << " N = " << N << " K = " << K << " StridesA = " << StridesA
              << " StridesB = " << StridesB << " StridesE = " << StridesE << " : " << best_avg_time
              << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
