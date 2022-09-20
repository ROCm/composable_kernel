// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
                                                                  |-----------------|
                                                                          Gemm0
                                                          |-------------------------------------|
                                                                          Gemm1
*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using B0DataType       = F16;
using B1DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = F16;

using ALayout  = Row;
using B0Layout = Col;
using B1Layout = Row;

using CPermuteNumDims_G_M_O =
    S<1, 1, 1>; // "using CLayout = Row" has been replaced by CPermuteNumDims_M_O

using AElementOp    = PassThrough;
using B0ElementOp   = PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using B1ElementOp   = PassThrough;
using CElementOp    = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGroupedGemmSoftmaxGemmPermute_Xdl_CShuffle<
        ALayout,
        B0Layout,
        B1Layout,
        CPermuteNumDims_G_M_O,
        ADataType,
        B0DataType,
        B1DataType,
        CDataType,
        AccDataType,
        CShuffleDataType,
        AElementOp,
        B0ElementOp,
        Acc0ElementOp,
        B1ElementOp,
        CElementOp,
        GemmSpec,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        64,          // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        32,          // MPerXDL
        32,          // NPerXDL
        1,           // MXdlPerWave
        4,           // NXdlPerWave
        2,           // Gemm1NXdlPerWave
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
        S<16, 16, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
        false>;

// Ref Gemm0: fp16 in, fp32 out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                Acc0ElementOp>;

// Ref Softmax: fp32 in, fp16 out
using ReferenceSoftmaxInstance =
    ck::tensor_operation::host::ReferenceSoftmax<AccDataType, ADataType, AccDataType>;

// Ref Gemm1: fp16 in, fp16 out
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

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
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    float alpha = 1; // scaling after 1st gemm

    std::size_t group_count = 13;

    // Problem descs
    std::vector<DeviceGemmInstance::ProblemDesc> problem_descs;
    std::vector<const void*> p_a;
    std::vector<const void*> p_b0;
    std::vector<const void*> p_b1;
    std::vector<void*> p_c;

    for(std::size_t i = 0; i < group_count; i++)
    {
        int M     = 128 * (rand() % 8 + 1);
        int N     = 128 * (rand() % 8 + 1);
        int K     = 40;
        int O     = 40 * (rand() % 2 + 1);
        int Batch = rand() % 8 + 1;

        const int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
        const int StrideB0 = ck::is_same_v<B0Layout, Row> ? N : K;
        const int StrideB1 = ck::is_same_v<B1Layout, Row> ? O : N;

        const int BatchStrideA  = (ck::is_same_v<ALayout, Col> ? K : M) * StrideA;
        const int BatchStrideB0 = (ck::is_same_v<B0Layout, Col> ? N : K) * StrideB0;
        const int BatchStrideB1 = (ck::is_same_v<B1Layout, Col> ? O : N) * StrideB1;

        std::vector<ck::index_t> c_gs_ms_os_lengths{Batch, M, O};
        std::vector<ck::index_t> c_gs_ms_os_strides{O, Batch * O, 1};

        problem_descs.push_back({M,
                                 N,
                                 K,
                                 O,
                                 Batch,
                                 StrideA,
                                 StrideB0,
                                 StrideB1,
                                 BatchStrideA,
                                 BatchStrideB0,
                                 BatchStrideB1,
                                 c_gs_ms_os_lengths,
                                 c_gs_ms_os_strides});
    }

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

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<B0DataType>> b0_tensors;
    std::vector<Tensor<B1DataType>> b1_tensors;
    std::vector<Tensor<CDataType>> c_tensors;

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device;
    std::vector<DeviceMemPtr> b0_tensors_device;
    std::vector<DeviceMemPtr> b1_tensors_device;
    std::vector<DeviceMemPtr> c_tensors_device;

    std::size_t flop = 0, num_byte = 0;

    std::cout << "group count " << group_count << ". printing first 4 groups\n";
    for(std::size_t i = 0; i < group_count; i++)
    {
        const auto& M                  = problem_descs[i].M;
        const auto& N                  = problem_descs[i].N;
        const auto& K                  = problem_descs[i].K;
        const auto& O                  = problem_descs[i].O;
        const auto& Batch              = problem_descs[i].Batch;
        const auto& StrideA            = problem_descs[i].StrideA;
        const auto& StrideB0           = problem_descs[i].StrideB0;
        const auto& StrideB1           = problem_descs[i].StrideB1;
        const auto& BatchStrideA       = problem_descs[i].BatchStrideA;
        const auto& BatchStrideB0      = problem_descs[i].BatchStrideB0;
        const auto& BatchStrideB1      = problem_descs[i].BatchStrideB1;
        const auto& c_gs_ms_os_lengths = problem_descs[i].c_gs_ms_os_lengths;
        const auto& c_gs_ms_os_strides = problem_descs[i].c_gs_ms_os_strides;

        // C_m_o = A_m_k * B0_k_n * B1_n_o
        Tensor<ADataType> a_g_m_k(
            f_host_tensor_descriptor(Batch, M, K, StrideA, BatchStrideA, ALayout{}));
        Tensor<B0DataType> b0_g_k_n(
            f_host_tensor_descriptor(Batch, K, N, StrideB0, BatchStrideB0, B0Layout{}));
        Tensor<B1DataType> b1_g_n_o(
            f_host_tensor_descriptor(Batch, N, O, StrideB1, BatchStrideB1, B1Layout{}));
        Tensor<CDataType> c_gs_ms_os_device_result(
            std::vector<std::size_t>(c_gs_ms_os_lengths.begin(), c_gs_ms_os_lengths.end()),
            std::vector<std::size_t>(c_gs_ms_os_strides.begin(), c_gs_ms_os_strides.end()));

        flop += (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * Batch;
        num_byte += (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                     sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                    Batch;

        if(i < 4)
        {
            std::cout << "a_g_m_k[" << i << "]: " << a_g_m_k.mDesc << ", "
                      << "b0_g_k_n[" << i << "]: " << b0_g_k_n.mDesc << ", "
                      << "b1_g_n_o[" << i << "]: " << b1_g_n_o.mDesc << ", "
                      << "c_gs_ms_os[" << i << "]: " << c_gs_ms_os_device_result.mDesc << std::endl;
        }

        switch(init_method)
        {
        case 0: break;
        case 1:
            a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
            b0_g_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
            b1_g_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
            break;
        case 2:
            a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b0_g_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
            b1_g_n_o.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
            break;
        case 3:
            a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
            b0_g_k_n.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
            b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
            break;
        default:
            a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
            b0_g_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
            b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        }

        a_tensors.push_back(a_g_m_k);
        b0_tensors.push_back(b0_g_k_n);
        b1_tensors.push_back(b1_g_n_o);
        c_tensors.push_back(c_gs_ms_os_device_result);

        a_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize()));
        b0_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(B0DataType) * b0_g_k_n.mDesc.GetElementSpaceSize()));
        b1_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(B1DataType) * b1_g_n_o.mDesc.GetElementSpaceSize()));
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * c_gs_ms_os_device_result.mDesc.GetElementSpaceSize()));

        a_tensors_device[i]->ToDevice(a_g_m_k.mData.data());
        b0_tensors_device[i]->ToDevice(b0_g_k_n.mData.data());
        b1_tensors_device[i]->ToDevice(b1_g_n_o.mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b0.push_back(b0_tensors_device[i]->GetDeviceBuffer());
        p_b1.push_back(b1_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(p_a,
                                      p_b0,
                                      p_b1,
                                      p_c,
                                      problem_descs,
                                      a_element_op,
                                      b0_element_op,
                                      acc0_element_op,
                                      b1_element_op,
                                      c_element_op);

    // specify workspace for problem_desc
    DeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        for(std::size_t i = 0; i < group_count; i++)
        {
            const auto& M                  = problem_descs[i].M;
            const auto& N                  = problem_descs[i].N;
            const auto& O                  = problem_descs[i].O;
            const auto& Batch              = problem_descs[i].Batch;
            const auto& c_gs_ms_os_lengths = problem_descs[i].c_gs_ms_os_lengths;
            const auto& c_gs_ms_os_strides = problem_descs[i].c_gs_ms_os_strides;

            const auto& a_g_m_k            = a_tensors[i];
            const auto& b0_g_k_n           = b0_tensors[i];
            const auto& b1_g_n_o           = b1_tensors[i];
            auto& c_gs_ms_os_device_result = c_tensors[i];
            auto& c_gs_ms_os_device_buf    = *c_tensors_device[i];

            Tensor<CDataType> c_gs_ms_os_host_result(
                std::vector<std::size_t>(c_gs_ms_os_lengths.begin(), c_gs_ms_os_lengths.end()),
                std::vector<std::size_t>(c_gs_ms_os_strides.begin(), c_gs_ms_os_strides.end()));

            c_gs_ms_os_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

            // Output of Gemm0 is input A of Gemm1
            Tensor<AccDataType> acc0_m_n(f_host_tensor_descriptor(Batch, M, N, N, M * N, Row{}));

            Tensor<ADataType> a1_g_m_n(f_host_tensor_descriptor(Batch, M, N, N, M * N, Row{}));

            Tensor<CDataType> c_g_m_o_host_result(std::vector<int>{Batch, M, O},
                                                  std::vector<int>{M * O, O, 1});

            auto ref_gemm0          = ReferenceGemm0Instance{};
            auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
            auto ref_gemm0_argument = ref_gemm0.MakeArgument(
                a_g_m_k, b0_g_k_n, acc0_m_n, a_element_op, b0_element_op, acc0_element_op);

            ref_gemm0_invoker.Run(ref_gemm0_argument);

            auto ref_softmax          = ReferenceSoftmaxInstance{};
            auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
            auto ref_softmax_argument = ref_softmax.MakeArgument(acc0_m_n, a1_g_m_n, 1, 0, {2});

            ref_softmax_invoker.Run(ref_softmax_argument);

            auto ref_gemm1          = ReferenceGemm1Instance{};
            auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
            auto ref_gemm1_argument = ref_gemm1.MakeArgument(a1_g_m_n,
                                                             b1_g_n_o,
                                                             c_g_m_o_host_result,
                                                             PassThrough{},
                                                             b1_element_op,
                                                             c_element_op);

            ref_gemm1_invoker.Run(ref_gemm1_argument);

            // Note: in this example, we merely permute the dimensions by changing underlying
            // strides so we simply access data as-is
            c_gs_ms_os_host_result.ForEach(
                [&](auto& self, auto idx) { self(idx) = c_g_m_o_host_result(idx); });

            bool pass_ =
                ck::utils::check_err(c_gs_ms_os_device_result.mData, c_gs_ms_os_host_result.mData);
            pass &= pass_;
        }
    }

    return pass ? 0 : 1;
}
