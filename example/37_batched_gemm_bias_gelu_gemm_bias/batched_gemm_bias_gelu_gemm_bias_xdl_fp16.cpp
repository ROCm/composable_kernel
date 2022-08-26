// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Gemm fused operation. Computes C_m_o = A_m_k * B0_k_n * B1_n_o
                                              |------------|
                                                   Gemm0
                                              |---------------------|
                                                       Gemm1
*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_bias_gelu_gemm_bias_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using A0DataType        = F16;
using B0DataType        = F16;
using Acc0DataType      = F32;
using D0DataType        = F16;
using B1DataType        = F16;
using Acc1DataType      = F32;
using C1ShuffleDataType = F32;
using C1DataType        = F16;

using A0Layout = Row;
using B0Layout = Col;
using B1Layout = Row;
using C1Layout = Row;

using D0Layout = Row;

using A0ElementOp = PassThrough;
using B0ElementOp = PassThrough;
using C0ElementOp = PassThrough;
using D0ElementOp = ck::tensor_operation::element_wise::AddFastGelu;
using B1ElementOp = PassThrough;
using C1ElementOp = PassThrough;

static constexpr bool PadGemm0M = false;
static constexpr bool PadGemm0N = false;
static constexpr bool PadGemm0K = false;
static constexpr bool PadGemm1N = false;
static constexpr bool PadGemm1K = false;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedGemmBiasGeluGemmBias_Xdl_CShuffle<
        A0Layout,
        B0Layout,
        D0Layout,
        B1Layout,
        C1Layout,
        A0DataType,
        B0DataType,
        Acc0DataType,
        D0DataType,
        B1DataType,
        Acc1DataType,
        C1ShuffleDataType,
        C1DataType,
        A0ElementOp,
        B0ElementOp,
        C0ElementOp,
        D0ElementOp,
        B1ElementOp,
        C1ElementOp,
        PadGemm0M,
        PadGemm0N,
        PadGemm0K,
        PadGemm1N,
        PadGemm1K,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        128,         // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        32,          // MPerXDL
        32,          // NPerXDL
        1,           // MXdlPerWave
        4,           // NXdlPerWave
        4,           // Gemm1NXdlPerWave
        S<4, 64, 1>, // ABlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<8, 32, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<A0DataType,
                                                                                B0DataType,
                                                                                Acc0DataType,
                                                                                Acc0DataType,
                                                                                A0ElementOp,
                                                                                B0ElementOp,
                                                                                C0ElementOp>;

using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<Acc0DataType,
                                                                                B1DataType,
                                                                                C1DataType,
                                                                                Acc1DataType,
                                                                                PassThrough,
                                                                                B1ElementOp,
                                                                                C1ElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M             = 1024;
    ck::index_t N             = 1024;
    ck::index_t K             = 64;
    ck::index_t O             = 128;
    ck::index_t BatchCount    = 4;
    ck::index_t StrideA0      = -1;
    ck::index_t StrideB0      = -1;
    ck::index_t StrideB1      = -1;
    ck::index_t StrideC1      = -1;
    ck::index_t BatchStrideA0 = -1;
    ck::index_t BatchStrideB0 = -1;
    ck::index_t BatchStrideB1 = -1;
    ck::index_t BatchStrideC1 = -1;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 9)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
        O = std::stoi(argv[7]);

        BatchCount = std::stoi(argv[8]);
    }
    else if(argc == 17)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
        O = std::stoi(argv[7]);

        BatchCount = std::stoi(argv[8]);

        StrideA0 = std::stoi(argv[9]);
        StrideB0 = std::stoi(argv[10]);
        StrideB1 = std::stoi(argv[11]);
        StrideC1 = std::stoi(argv[12]);

        BatchStrideA0 = std::stoi(argv[13]);
        BatchStrideB0 = std::stoi(argv[14]);
        BatchStrideB1 = std::stoi(argv[15]);
        BatchStrideC1 = std::stoi(argv[16]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf(
            "arg4 to 17: M, N, K, O, Batch, StrideA0, StrideB0, StrideB1, StrideC1, BatchStrideA0, "
            "BatchStrideB0, BatchStrideB1, BatchStrideC1\n");
        exit(0);
    }

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
    const int BatchStrideD0 = ck::is_same_v<D0Layout, Col> ? M : N;

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
    Tensor<A0DataType> a_g_m_k(
        f_host_tensor_descriptor(BatchCount, M, K, StrideA0, BatchStrideA0, A0Layout{}));
    Tensor<B0DataType> b0_g_k_n(
        f_host_tensor_descriptor(BatchCount, K, N, StrideB0, BatchStrideB0, B0Layout{}));
    Tensor<D0DataType> d0_g_m_n(
        f_host_tensor_descriptor(BatchCount, M, N, StrideD0, BatchStrideD0, D0Layout{}));
    Tensor<B1DataType> b1_g_n_o(
        f_host_tensor_descriptor(BatchCount, N, O, StrideB1, BatchStrideB1, B1Layout{}));
    Tensor<C1DataType> c0_g_m_o_host_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideC1, BatchStrideC1, C1Layout{}));
    Tensor<C1DataType> c0_g_m_o_device_result(
        f_host_tensor_descriptor(BatchCount, M, O, StrideC1, BatchStrideC1, C1Layout{}));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b0_g_k_n: " << b0_g_k_n.mDesc << std::endl;
    std::cout << "b1_g_n_o: " << b1_g_n_o.mDesc << std::endl;
    std::cout << "c0_g_m_o: " << c0_g_m_o_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-5, 5});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-5, 5});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-5, 5});
        break;
    case 2:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{1});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        d0_g_m_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a0_g_m_k_device_buf(sizeof(A0DataType) * a_g_m_k.mDesc.GetElementSize());
    DeviceMem b0_g_k_n_device_buf(sizeof(B0DataType) * b0_g_k_n.mDesc.GetElementSize());
    DeviceMem d0_g_m_n_device_buf(sizeof(D0DataType) * d0_g_m_n.mDesc.GetElementSize());
    DeviceMem b1_g_n_o_device_buf(sizeof(B1DataType) * b1_g_n_o.mDesc.GetElementSize());
    DeviceMem c0_g_m_o_device_buf(sizeof(C1DataType) *
                                  c0_g_m_o_device_result.mDesc.GetElementSize());

    a0_g_m_k_device_buf.ToDevice(a_g_m_k.mData.data());
    b0_g_k_n_device_buf.ToDevice(b0_g_k_n.mData.data());
    d0_g_m_n_device_buf.ToDevice(d0_g_m_n.mData.data());
    b1_g_n_o_device_buf.ToDevice(b1_g_n_o.mData.data());

    auto a0_element_op = A0ElementOp{};
    auto b0_element_op = B0ElementOp{};
    auto c0_element_op = C0ElementOp{};
    auto d0_element_op = D0ElementOp{};
    auto b1_element_op = B1ElementOp{};
    auto c1_element_op = C1ElementOp{};

    // do GEMM
    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();
    auto argument =
        gemm.MakeArgument(static_cast<A0DataType*>(a0_g_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<B0DataType*>(b0_g_k_n_device_buf.GetDeviceBuffer()),
                          static_cast<D0DataType*>(d0_g_m_n_device_buf.GetDeviceBuffer()),
                          static_cast<B1DataType*>(b1_g_n_o_device_buf.GetDeviceBuffer()),
                          static_cast<C1DataType*>(c0_g_m_o_device_buf.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          O,
                          BatchCount,
                          StrideA0,
                          StrideB0,
                          StrideD0,
                          StrideB1,
                          StrideC1,
                          BatchStrideA0,
                          BatchStrideB0,
                          BatchStrideD0,
                          BatchStrideB1,
                          BatchStrideC1,
                          a0_element_op,
                          b0_element_op,
                          c0_element_op,
                          d0_element_op,
                          b1_element_op,
                          c1_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
    std::size_t num_btype = (sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N +
                             sizeof(B1DataType) * N * O + sizeof(C1DataType) * M * O) *
                            BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    c0_g_m_o_device_buf.FromDevice(c0_g_m_o_device_result.mData.data());

    if(do_verification)
    {
        // Output of Gemm0 is input A of Gemm1
        Tensor<Acc0DataType> a1_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));

        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_g_m_k, b0_g_k_n, a1_g_m_n, a0_element_op, b0_element_op, c0_element_op);

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            a1_g_m_n, b1_g_n_o, c0_g_m_o_host_result, PassThrough{}, b1_element_op, c1_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        return ck::utils::check_err(c0_g_m_o_device_result.mData, c0_g_m_o_host_result.mData) ? 0
                                                                                              : 1;
    }

    return 0;
}
