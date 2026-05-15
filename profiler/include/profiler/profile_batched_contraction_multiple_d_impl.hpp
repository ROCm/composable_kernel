// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <tuple>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_contraction.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_bias_permute.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/numeric.hpp"

namespace ck {
namespace profiler {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Row         = ck::tensor_layout::gemm::RowMajor;
using Bypass      = ck::tensor_layout::BypassLayoutVerification;

template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp>
bool profile_batched_contraction_multiple_d_impl(int do_verification,
                                                 int init_method,
                                                 bool do_log,
                                                 bool time_kernel,
                                                 std::array<ck::index_t, NumDimG> Gs,
                                                 std::array<ck::index_t, NumDimM> Ms,
                                                 std::array<ck::index_t, NumDimN> Ns,
                                                 std::array<ck::index_t, NumDimK> Ks,
                                                 int instance_index                  = -1,
                                                 bool fail_if_no_supported_instances = false)
{
    static_assert(NumDimG == 1 && NumDimM == 2 && NumDimN == 3 && NumDimK == 1,
                  "Tensor ranks not supported. Supported: G=1, M=2, N=3, K=1");
    static_assert(DsDataType::Size() == 1, "Only single D tensor is supported at the moment.");

    using AccDataType = float;
    using DDataType   = ck::tuple_element_t<0, DsDataType>;

    bool pass = true;

    ignore = do_log;

    ck::index_t G0 = Gs[0];

    ck::index_t M0 = Ms[0];
    ck::index_t M1 = Ms[1];

    ck::index_t N0 = Ns[0];
    ck::index_t N1 = Ns[1];
    ck::index_t N2 = Ns[2];

    ck::index_t K0 = Ks[0];

    // A[M0, M1, M2, K0]
    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, M0, M1, K0};
    std::vector<ck::index_t> a_gs_ms_ks_strides{M0 * M1 * K0, M1 * K0, K0, 1};
    // B[N0, N1, K0]
    std::vector<ck::index_t> b_gs_ns_ks_lengths{G0, N0, N1, N2, K0};
    std::vector<ck::index_t> b_gs_ns_ks_strides{N0 * N1 * N2 * K0, N1 * N2 * K0, N2 * K0, K0, 1};

    // D[N0, M0, N1, M1, N2]
    std::vector<ck::index_t> d_gs_ms_ns_lengths{G0, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> d_gs_ms_ns_strides{N0 * N1 * N2, 0, 0, N1 * N2, N2, 1};
    // E[N0, M0, N1, M1, N2]
    std::vector<ck::index_t> e_gs_ms_ns_lengths{G0, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> e_gs_ms_ns_strides{
        M0 * M1 * N0 * N1 * N2, N1 * M1 * N2, N2, M0 * N1 * M1 * N2, M1 * N2, 1};

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides, Row{});
    Tensor<BDataType> b_gs_ns_ks(b_gs_ns_ks_lengths, b_gs_ns_ks_strides, Row{});
    Tensor<DDataType> d_gs_ms_ns(d_gs_ms_ns_lengths, d_gs_ms_ns_strides, Bypass{});
    Tensor<EDataType> e_gs_ms_ns_host_result(e_gs_ms_ns_lengths, e_gs_ms_ns_strides, Bypass{});
    Tensor<EDataType> e_gs_ms_ns_device_result(e_gs_ms_ns_lengths, e_gs_ms_ns_strides, Bypass{});

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b_gs_ns_ks: " << b_gs_ns_ks.mDesc << std::endl;
    std::cout << "d_gs_ms_ns: " << d_gs_ms_ns.mDesc << std::endl;
    std::cout << "e_gs_ms_ns: " << e_gs_ms_ns_host_result.mDesc << std::endl;

    // get device op instances
    using DeviceOp     = ck::tensor_operation::device::DeviceBatchedContractionMultipleD<NumDimG,
                                                                                         NumDimM,
                                                                                         NumDimN,
                                                                                         NumDimK,
                                                                                         ADataType,
                                                                                         BDataType,
                                                                                         DsDataType,
                                                                                         EDataType,
                                                                                         AElementOp,
                                                                                         BElementOp,
                                                                                         CDEElementOp>;
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_2<DDataType>{-5, 5});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_3<DDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) *
                           e_gs_ms_ns_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b_device_buf.ToDevice(b_gs_ns_ks.mData.data());
    d_device_buf.ToDevice(d_gs_ms_ns.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    if(do_verification)
    {
        Tensor<EDataType> c_gs_ms_ns_host_result(e_gs_ms_ns_lengths, e_gs_ms_ns_strides, Bypass{});

        using ReferenceOpInstance =
            ck::tensor_operation::host::ReferenceBatchedContraction_G1_M2_N3_K1<NumDimG,
                                                                                NumDimM,
                                                                                NumDimN,
                                                                                NumDimK,
                                                                                ADataType,
                                                                                BDataType,
                                                                                EDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                PassThrough>;

        auto ref_gemm    = ReferenceOpInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_gs_ms_ks,
                                                  b_gs_ns_ks,
                                                  c_gs_ms_ns_host_result,
                                                  a_element_op,
                                                  b_element_op,
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        for(size_t g0 = 0; g0 < e_gs_ms_ns_host_result.mDesc.GetLengths()[0]; ++g0)
        {
            for(size_t m0 = 0; m0 < e_gs_ms_ns_host_result.mDesc.GetLengths()[1]; ++m0)
            {
                for(size_t m1 = 0; m1 < e_gs_ms_ns_host_result.mDesc.GetLengths()[2]; ++m1)
                {
                    for(size_t n0 = 0; n0 < e_gs_ms_ns_host_result.mDesc.GetLengths()[3]; ++n0)
                    {
                        for(size_t n1 = 0; n1 < e_gs_ms_ns_host_result.mDesc.GetLengths()[4]; ++n1)
                        {
                            for(size_t n2 = 0; n2 < e_gs_ms_ns_host_result.mDesc.GetLengths()[5];
                                ++n2)
                            {
                                cde_element_op(e_gs_ms_ns_host_result(g0, m0, m1, n0, n1, n2),
                                               c_gs_ms_ns_host_result(g0, m0, m1, n0, n1, n2),
                                               d_gs_ms_ns(g0, m0, m1, n0, n1, n2));
                            }
                        }
                    }
                }
            }
        }
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    int num_kernel        = 0;

    // profile device op instances
    for(size_t i = 0; i < op_ptrs.size(); i++)
    {
        if((instance_index != -1) && (instance_index != static_cast<int>(i)))
        {
            // skip test if instance_index is specified
            continue;
        }
        auto& op_ptr     = op_ptrs[i];
        auto invoker_ptr = op_ptr->MakeInvokerPointer();
        auto argument_ptr =
            op_ptr->MakeArgumentPointer(a_device_buf.GetDeviceBuffer(),
                                        b_device_buf.GetDeviceBuffer(),
                                        std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                                        e_device_buf.GetDeviceBuffer(),
                                        a_gs_ms_ks_lengths,
                                        a_gs_ms_ks_strides,
                                        b_gs_ns_ks_lengths,
                                        b_gs_ns_ks_strides,
                                        std::array<std::vector<ck::index_t>, 1>{d_gs_ms_ns_lengths},
                                        std::array<std::vector<ck::index_t>, 1>{d_gs_ms_ns_strides},
                                        e_gs_ms_ns_lengths,
                                        e_gs_ms_ns_strides,
                                        a_element_op,
                                        b_element_op,
                                        cde_element_op);

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            num_kernel++;
            // re-init E to zero before profiling next kernel
            e_device_buf.SetZero();

            std::string op_name = op_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            ck::index_t G = ck::accumulate_n<ck::index_t>(
                e_gs_ms_ns_lengths.begin(), NumDimG, 1, std::multiplies<>{});

            ck::index_t M = ck::accumulate_n<ck::index_t>(
                e_gs_ms_ns_lengths.begin() + NumDimG, NumDimM, 1, std::multiplies<>{});

            ck::index_t N = ck::accumulate_n<ck::index_t>(
                e_gs_ms_ns_lengths.begin() + NumDimG + NumDimM, NumDimN, 1, std::multiplies<>{});

            ck::index_t K = ck::accumulate_n<ck::index_t>(
                a_gs_ms_ks_lengths.begin() + NumDimG + NumDimM, NumDimK, 1, std::multiplies<>{});

            std::size_t flop      = std::size_t(2) * G * M * N * K;
            std::size_t num_btype = sizeof(ADataType) * G * M * K + sizeof(BDataType) * G * K * N +
                                    sizeof(DDataType) * G * M * N + sizeof(EDataType) * G * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                e_device_buf.FromDevice(e_gs_ms_ns_device_result.mData.data());

                pass =
                    pass & ck::utils::check_err(e_gs_ms_ns_device_result, e_gs_ms_ns_host_result);
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    if(fail_if_no_supported_instances && num_kernel == 0 && instance_index == -1)
    {
        std::cout << "Error: No kernel is applicable" << std::endl;
        return false;
    }

    return pass;
}

} // namespace profiler
} // namespace ck
