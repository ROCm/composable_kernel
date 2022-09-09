// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/device_batched_gemm_multiple_d_gemm_multiple_d.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

namespace ck {
namespace profiler {

template <typename A0Layout,
          typename B0Layout,
          typename D0sLayout,
          typename B1Layout,
          typename C1Layout,
          typename D1sLayout,
          typename A0DataType,
          typename B0DataType,
          typename D0sDataType,
          typename B1DataType,
          typename C1DataType,
          typename D1sDataType>
bool profile_batched_gemm_bias_gelu_gemm_bias_impl(bool do_verification,
                                                   int init_method,
                                                   bool do_log,
                                                   bool time_kernel,
                                                   int M,
                                                   int N,
                                                   int K,
                                                   int O,
                                                   int BatchCount    = 1,
                                                   int StrideA0      = -1,
                                                   int StrideB0      = -1,
                                                   int StrideB1      = -1,
                                                   int StrideC1      = -1,
                                                   int BatchStrideA0 = -1,
                                                   int BatchStrideB0 = -1,
                                                   int BatchStrideB1 = -1,
                                                   int BatchStrideC1 = -1)

{

    using Row           = tensor_layout::gemm::RowMajor;
    using Col           = tensor_layout::gemm::ColumnMajor;
    using PassThrough   = tensor_operation::element_wise::PassThrough;
    using A0ElementOp   = PassThrough;
    using B0ElementOp   = PassThrough;
    using C0ElementOp   = PassThrough;
    using CDE0ElementOp = ck::tensor_operation::element_wise::AddRelu;
    using A1ElementOp   = PassThrough;
    using B1ElementOp   = PassThrough;
    using C1ElementOp   = PassThrough;
    using CDE1ElementOp = ck::tensor_operation::element_wise::Add;
    using AccDataType   = float;
    using D0DataType    = remove_cvref_t<tuple_element_t<0, D0sDataType>>;
    using D0Layout      = remove_cvref_t<tuple_element_t<0, D0sLayout>>;
    using D1DataType    = remove_cvref_t<tuple_element_t<0, D1sDataType>>;
    using D1Layout      = remove_cvref_t<tuple_element_t<0, D1sLayout>>;

    // Ref Gemm0
    using ReferenceGemm0Instance = tensor_operation::host::ReferenceBatchedGemm<A0DataType,
                                                                                B0DataType,
                                                                                A0DataType,
                                                                                AccDataType,
                                                                                A0ElementOp,
                                                                                B0ElementOp,
                                                                                C0ElementOp>;

    // Ref Gemm
    using ReferenceGemm1Instance = tensor_operation::host::ReferenceBatchedGemm<A0DataType,
                                                                                B1DataType,
                                                                                C1DataType,
                                                                                AccDataType,
                                                                                A1ElementOp,
                                                                                B1ElementOp,
                                                                                C1ElementOp>;

    bool pass = true;

    const int DefaultStrideA0 = ck::is_same_v<A0Layout, Row> ? K : M;
    const int DefaultStrideB0 = ck::is_same_v<B0Layout, Row> ? N : K;
    const int DefaultStrideB1 = ck::is_same_v<B1Layout, Row> ? O : N;
    const int DefaultStrideC1 = ck::is_same_v<C1Layout, Row> ? O : M;

    StrideA0 = (StrideA0 < 0) ? DefaultStrideA0 : StrideA0;
    StrideB0 = (StrideB0 < 0) ? DefaultStrideB0 : StrideB0;
    StrideB1 = (StrideB1 < 0) ? DefaultStrideB1 : StrideB1;
    StrideC1 = (StrideC1 < 0) ? DefaultStrideC1 : StrideC1;

    const int DefaultBatchStrideA0 = (ck::is_same_v<A0Layout, Col> ? K : M) * StrideA0;
    const int DefaultBatchStrideB0 = (ck::is_same_v<B0Layout, Col> ? N : K) * StrideB0;
    const int DefaultBatchStrideB1 = (ck::is_same_v<B1Layout, Col> ? O : N) * StrideB1;
    const int DefaultBatchStrideC1 = (ck::is_same_v<C1Layout, Col> ? O : M) * StrideC1;

    BatchStrideA0 = BatchStrideA0 < 0 ? DefaultBatchStrideA0 : BatchStrideA0;
    BatchStrideB0 = BatchStrideB0 < 0 ? DefaultBatchStrideB0 : BatchStrideB0;
    BatchStrideB1 = BatchStrideB1 < 0 ? DefaultBatchStrideB1 : BatchStrideB1;
    BatchStrideC1 = BatchStrideC1 < 0 ? DefaultBatchStrideC1 : BatchStrideC1;

    const int StrideD0      = 0;
    const int BatchStrideD0 = N;

    const int StrideD1      = 0;
    const int BatchStrideD1 = O;

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        if(std::is_same<decltype(layout), Row>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({batch_stride, stride, 1}));
        }
        else
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({batch_stride, 1, stride}));
        }
    };

    // C_m_o = A_m_k * B0_k_n * B1_n_o
    Tensor<A0DataType> a0_g_m_k(
        f_host_tensor_descriptor(BatchCount, M, K, StrideA0, BatchStrideA0, A0Layout{}));
    Tensor<B0DataType> b0_g_k_n(
        f_host_tensor_descriptor(BatchCount, K, N, StrideB0, BatchStrideB0, B0Layout{}));
    Tensor<D0DataType> d0_g_m_n(
        f_host_tensor_descriptor(BatchCount, M, N, StrideD0, BatchStrideD0, D0Layout{}));
    Tensor<B1DataType> b1_g_n_o(
        f_host_tensor_descriptor(BatchCount, N, O, StrideB1, BatchStrideB1, B1Layout{}));
    Tensor<C1DataType> c1_g_m_o_host_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideC1, BatchStrideC1, C1Layout{}));
    Tensor<C1DataType> c1_g_m_o_device_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideC1, BatchStrideC1, C1Layout{}));
    Tensor<D1DataType> d1_g_m_o(
        f_host_tensor_descriptor(BatchCount, M, O, StrideD1, BatchStrideD1, D1Layout{}));
    // Host verification: Output of Gemm0 is input A of Gemm1
    Tensor<A0DataType> acc0_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));

    std::cout << "a0_g_m_k: " << a0_g_m_k.mDesc << std::endl;
    std::cout << "b0_g_k_n: " << b0_g_k_n.mDesc << std::endl;
    std::cout << "d0_g_m_n: " << d0_g_m_n.mDesc << " size: " << d0_g_m_n.mDesc.GetElementSpaceSize()
              << std::endl;
    std::cout << "b1_g_n_o: " << b1_g_n_o.mDesc << std::endl;
    std::cout << "c1_g_m_o: " << c1_g_m_o_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_g_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 3});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 3});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-2, 3});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 3});
        d1_g_m_o.GenerateTensorValue(GeneratorTensor_2<D1DataType>{-2, 3});
        break;
    case 2:
        a0_g_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        d1_g_m_o.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
        break;
    default:
        a0_g_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{1});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        d1_g_m_o.GenerateTensorValue(GeneratorTensor_1<D1DataType>{1});
    }

    DeviceMem a0_g_m_k_device_buf(sizeof(A0DataType) * a0_g_m_k.mDesc.GetElementSize());
    DeviceMem b0_g_k_n_device_buf(sizeof(B0DataType) * b0_g_k_n.mDesc.GetElementSize());
    DeviceMem d0_g_m_n_device_buf(sizeof(D0DataType) * d0_g_m_n.mDesc.GetElementSpaceSize());
    DeviceMem b1_g_n_o_device_buf(sizeof(B1DataType) * b1_g_n_o.mDesc.GetElementSize());
    DeviceMem c1_g_m_o_device_buf(sizeof(C1DataType) *
                                  c1_g_m_o_device_result.mDesc.GetElementSize());
    DeviceMem d1_g_m_o_device_buf(sizeof(D1DataType) * d1_g_m_o.mDesc.GetElementSpaceSize());

    a0_g_m_k_device_buf.ToDevice(a0_g_m_k.mData.data());
    b0_g_k_n_device_buf.ToDevice(b0_g_k_n.mData.data());
    d0_g_m_n_device_buf.ToDevice(d0_g_m_n.mData.data());
    b1_g_n_o_device_buf.ToDevice(b1_g_n_o.mData.data());
    d1_g_m_o_device_buf.ToDevice(d1_g_m_o.mData.data());

    auto a0_element_op   = A0ElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto c0_element_op   = C0ElementOp{};
    auto cde0_element_op = CDE0ElementOp{};
    auto a1_element_op   = A1ElementOp{};
    auto b1_element_op   = B1ElementOp{};
    auto c1_element_op   = C1ElementOp{};
    auto cde1_element_op = CDE1ElementOp{};

    using DeviceOp =
        tensor_operation::device::DeviceBatchedGemmMultipleDGemmMultipleD<A0Layout,
                                                                          B0Layout,
                                                                          D0sLayout,
                                                                          B1Layout,
                                                                          C1Layout,
                                                                          D1sLayout,
                                                                          A0DataType,
                                                                          B0DataType,
                                                                          D0sDataType,
                                                                          B1DataType,
                                                                          C1DataType,
                                                                          D1sDataType,
                                                                          A0ElementOp,
                                                                          B0ElementOp,
                                                                          CDE0ElementOp,
                                                                          A1ElementOp,
                                                                          B1ElementOp,
                                                                          CDE1ElementOp>;

    // get device op instances
    const auto op_ptrs = tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    if(do_verification)
    {
        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a0_g_m_k, b0_g_k_n, acc0_g_m_n, a0_element_op, b0_element_op, c0_element_op);

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        // bias+gelu
        for(int b = 0; b < BatchCount; ++b)
        {
            for(int m = 0; m < M; ++m)
            {
                for(int n = 0; n < N; ++n)
                {
                    cde0_element_op(acc0_g_m_n(b, m, n), acc0_g_m_n(b, m, n), d0_g_m_n(b, m, n));
                }
            }
        }

        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(acc0_g_m_n,
                                                         b1_g_n_o,
                                                         c1_g_m_o_host_result,
                                                         a1_element_op,
                                                         b1_element_op,
                                                         c1_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // bias
        for(int b = 0; b < BatchCount; ++b)
        {
            for(int m = 0; m < M; ++m)
            {
                for(int o = 0; o < O; ++o)
                {
                    cde1_element_op(c1_g_m_o_host_result(b, m, o),
                                    c1_g_m_o_host_result(b, m, o),
                                    d1_g_m_o(b, m, o));
                }
            }
        }
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<A0DataType*>(a0_g_m_k_device_buf.GetDeviceBuffer()),
            static_cast<B0DataType*>(b0_g_k_n_device_buf.GetDeviceBuffer()),
            std::array<const void*, 1>{d0_g_m_n_device_buf.GetDeviceBuffer()},
            static_cast<B1DataType*>(b1_g_n_o_device_buf.GetDeviceBuffer()),
            static_cast<C1DataType*>(c1_g_m_o_device_buf.GetDeviceBuffer()),
            std::array<const void*, 1>{d1_g_m_o_device_buf.GetDeviceBuffer()},
            M,
            N,
            K,
            O,
            BatchCount,
            StrideA0,
            StrideB0,
            std::array<ck::index_t, 1>{StrideD0},
            StrideB1,
            StrideC1,
            std::array<ck::index_t, 1>{StrideD1},
            BatchStrideA0,
            BatchStrideB0,
            std::array<ck::index_t, 1>{BatchStrideD0},
            BatchStrideB1,
            BatchStrideC1,
            std::array<ck::index_t, 1>{BatchStrideD1},
            a0_element_op,
            b0_element_op,
            cde0_element_op,
            a1_element_op,
            b1_element_op,
            cde1_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string op_name = op_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
            std::size_t num_btype = (sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N +
                                     sizeof(B1DataType) * N * O + sizeof(C1DataType) * M * O) *
                                    BatchCount;

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
                c1_g_m_o_device_buf.FromDevice(c1_g_m_o_device_result.mData.data());

                pass = pass & ck::utils::check_err(c1_g_m_o_device_result.mData,
                                                   c1_g_m_o_host_result.mData);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a0_g_m_k: ", a0_g_m_k.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "b0_g_k_n : ", b0_g_k_n.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "b1_g_n_o : ", b1_g_n_o.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c1_g_m_o_host_result : ", c1_g_m_o_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c1_g_m_o_device_result : ", c1_g_m_o_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
