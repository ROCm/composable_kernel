// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
/*
Backprop for Gemm + Softmax + Gemm fused operation, where forward prop is defined as:

  Y_g_m_o = Softmax(alpha * Q_g_m_k * K_g_k_n) * V_g_n_o

Computation graph:

          K^T                   V
          |                     |
          |                     |
    Q --- * ----- Softmax ----- * --> Y
              S             P

Kernel inputs:

    Q, K, V, Y, dY, per-row softmax stats (LSE)

Kernel outputs:

    dQ, dK, dV

*/

#define PRINT_HOST 0
#define USING_MASK 0

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_backward_xdl_cshuffle_v2.hpp"
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

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

using QKVElementOp = PassThrough;
using YElementOp   = PassThrough;

using DataType         = F16;
using AccDataType      = F32;
using ShuffleDataType  = F32;
using LSEDataType      = F32;
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
#if USING_MASK
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskOutUpperTriangle;
#else
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;
#endif

static constexpr auto TensorSpecQ = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecK = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecV = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecY = ck::tensor_operation::device::TensorSpecialization::Default;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Xdl_CShuffle_V2<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        DataType,
        LSEDataType,
        Acc0BiasDataType,
        Acc1BiasDataType,
        AccDataType,
        ShuffleDataType,
        QKVElementOp,
        QKVElementOp,
        Scale,
        QKVElementOp,
        YElementOp,
        GemmSpec,
        TensorSpecQ,
        TensorSpecK,
        TensorSpecV,
        TensorSpecY,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        64,          // KPerBlock
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
        8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
        MaskingSpec>;   // MaskingSpecialization

// Ref Gemm0: S = alpha * Q * K^T
// fp16 in, fp32 out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                Scale>;

// Ref Softmax: P = Softmax(S)
// fp32 in, fp16 out
using ReferenceSoftmaxInstance =
    ck::tensor_operation::host::ReferenceSoftmax<AccDataType, DataType, AccDataType>;

// Ref Gemm1: Y = P * V
// fp16 in, fp16 out
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                DataType,
                                                                                DataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;

// Ref Gemm for backward pass
// fp16 in, fp16 out
using ReferenceGemmGradInstance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                   DataType,
                                                                                   DataType,
                                                                                   AccDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   Scale>;

template <typename TensorQ,
          typename TensorK,
          typename TensorV,
          typename TensorS,
          typename TensorP,
          typename TensorY,
          typename TensorLSE = TensorP>
void run_attention_fwd_host(const TensorQ& q_g_m_k,
                            const TensorK& k_g_n_k,
                            const TensorV& v_g_n_o,
                            const float alpha,
                            TensorS& s_g_m_n,
                            TensorP& p_g_m_n,
                            TensorY& y_g_m_o,
                            TensorLSE& lse_g_m)
{
    // S = alpha * Q * K^T
    auto k_g_k_n            = k_g_n_k.Transpose({0, 2, 1});
    auto ref_gemm0          = ReferenceGemm0Instance{};
    auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
    auto ref_gemm0_argument = ref_gemm0.MakeArgument(
        q_g_m_k, k_g_k_n, s_g_m_n, PassThrough{}, PassThrough{}, Scale{alpha});

    ref_gemm0_invoker.Run(ref_gemm0_argument);

    // masking
#if USING_MASK
    auto N          = s_g_m_n.GetLengths()[2];
    const auto mask = DeviceGemmInstance::C0MatrixMask(N);
    s_g_m_n.ForEach([&](auto& self, auto idx) {
        if(mask.IsMaskedElement(idx[1], idx[2]))
            self(idx) = -ck::NumericLimits<float>::Infinity();
    });
#endif

    // P = Softmax(S)
    auto ref_softmax          = ReferenceSoftmaxInstance{};
    auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
    auto ref_softmax_argument = ref_softmax.MakeArgument(s_g_m_n, p_g_m_n, 1, 0, {2}, &lse_g_m);

    ref_softmax_invoker.Run(ref_softmax_argument);

    // Y = P * V
    auto ref_gemm1          = ReferenceGemm1Instance{};
    auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
    auto ref_gemm1_argument = ref_gemm1.MakeArgument(
        p_g_m_n, v_g_n_o, y_g_m_o, PassThrough{}, PassThrough{}, PassThrough{});

    ref_gemm1_invoker.Run(ref_gemm1_argument);
}

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 2; // method 1 will have slightly higher error; TODO: to investigate
    bool time_kernel     = true;

    // Overall QKV matrices shape
    // y_g_m_o = Softmax(alpha * Q_g_m_k * K_g_k_n) * V_g_n_o
    // y_g0_g1_m_o = reshape(y_g_m_o, [G0, G1, M, O])
    // y_g0_m_g1_o = permute(y_g0_g1_m_o, [0, 2, 1, 3])
    ck::index_t M  = 512;
    ck::index_t N  = 512;
    ck::index_t K  = 64;
    ck::index_t O  = 64;
    ck::index_t G0 = 54; //54
    ck::index_t G1 = 16; //16

    float alpha = 1.f / std::sqrt(K);

    bool input_permute  = false;
    bool output_permute = false;

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
    else if(argc == 13)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M  = std::stoi(argv[4]);
        N  = std::stoi(argv[5]);
        K  = std::stoi(argv[6]);
        O  = std::stoi(argv[7]);
        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);

        alpha = std::stof(argv[10]);

        input_permute  = std::stoi(argv[11]);
        output_permute = std::stoi(argv[12]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        printf("arg11 to 12: input / output permute\n");
        exit(0);
    }

    const ck::index_t BatchCount = G0 * G1;

    std::vector<ck::index_t> q_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> q_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // Q layout [G0, M, G1, K]
            : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // Q layout [G0, G1, M, K]

    std::vector<ck::index_t> k_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> k_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // K layout [G0, N, G1, K]
            : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // K layout [G0, G1, N, K]

    std::vector<ck::index_t> v_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> v_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // V layout [G0, N, G1, O]
            : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // V layout [G0, G1, N, O]

    std::vector<ck::index_t> y_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> y_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // Y layout [G0, M, G1, O]
            : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // Y layout [G0, G1, M, O]

    // The softmax stat log-sum-exp (LSE) is used to speed up softmax calculation in backward pass
    // Pi = exp(Si) / sum(exp(S0) + exp(S1) + ...)
    //    = exp(Si) / exp(log(sum(exp() + ...)))
    //    = exp(Si - log(sum(exp() + ...)))
    //               ^^^^^^^^^^^^^^^^^^^^^
    //                       LSE
    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides{G1 * M, M, 1}; // LSE layout [G0, G1, M]

    Tensor<DataType> q_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
    Tensor<DataType> k_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
    Tensor<DataType> v_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);
    Tensor<DataType> y_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
    Tensor<DataType> ygrad_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
    Tensor<LSEDataType> lse_gs_ms(lse_gs_ms_lengths, lse_gs_ms_strides);

    std::cout << "q_gs_ms_ks: " << q_gs_ms_ks.mDesc << std::endl;
    std::cout << "k_gs_ns_ks: " << k_gs_ns_ks.mDesc << std::endl;
    std::cout << "v_gs_os_ns: " << v_gs_os_ns.mDesc << std::endl;
    std::cout << "y_gs_ms_os: " << y_gs_ms_os.mDesc << std::endl;
    std::cout << "lse_gs_ms_os: " << lse_gs_ms.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        break;
    case 2:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<DataType>{0.0, 1.0});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<DataType>{0.0, 1.0});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<DataType>{-0.5, 0.5});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_3<DataType>{-0.5, 0.5});
        break;
    case 3:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        break;
    case 4:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<DataType>{1});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_1<DataType>{2});
        break;
    case 5:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<DataType>{1});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Sequential<2>{}); // dy[g0, g1, m, o]
        // dO dot O = [0; 1; 2; ...]
        break;
    case 6:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<DataType>{1});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Sequential<3>{}); // dy[g0, g1, m, o]
        // assume mnko = 256
        // P = softmax(QK) = 0.0039 * ones
        // O = P V = 0.0039 * ones
        // dP = dO V = [0, 1, 2, ...; 0, 1, 2, ...; ...]
        // dO dot O = [127.5; ...]
        // dS = P * (dP - dO dot O)
        //
        break;
    default:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<DataType>{1});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_1<DataType>{1}); // dy[g0, g1, m, o]
        // assume mnko = 256
        // P = softmax(QK) = 0.0039 * ones
        // O = P V = 0.0039 * ones
        // dP = dO V = ones
        // dS = P * (dP - (dO dot O))
        //    = 0.0039 * ones * (ones - 0.0039*256)
        //    = 0.0039 * ones * (ones - 1)
        //    = 0
    }

    // calculate y & log-sum-exp beforehand
    Tensor<DataType> q_g_m_k({BatchCount, M, K});
    Tensor<DataType> k_g_n_k({BatchCount, N, K});
    Tensor<DataType> v_g_n_o({BatchCount, N, O});
    Tensor<AccDataType> s_g_m_n({BatchCount, M, N});
    Tensor<DataType> p_g_m_n({BatchCount, M, N});
    Tensor<DataType> y_g_m_o({BatchCount, M, O});
    Tensor<LSEDataType> lse_g_m({BatchCount, M});

    q_gs_ms_ks.ForEach(
        [&](auto& self, auto idx) { q_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx); });
    k_gs_ns_ks.ForEach(
        [&](auto& self, auto idx) { k_g_n_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx); });
    v_gs_os_ns.ForEach(
        [&](auto& self, auto idx) { v_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx); });
    lse_gs_ms.ForEach(
        [&](auto& self, auto idx) { lse_g_m(idx[0] * G1 + idx[1], idx[2]) = self(idx); });

    run_attention_fwd_host(q_g_m_k, k_g_n_k, v_g_n_o, alpha, s_g_m_n, p_g_m_n, y_g_m_o, lse_g_m);

    y_gs_ms_os.ForEach(
        [&](auto& self, auto idx) { self(idx) = y_g_m_o(idx[0] * G1 + idx[1], idx[2], idx[3]); });
    lse_gs_ms.ForEach(
        [&](auto& self, auto idx) { self(idx) = lse_g_m(idx[0] * G1 + idx[1], idx[2]); });

    // qkv gradients have the same descriptor as with qkv
    DeviceMem q_device_buf(sizeof(DataType) * q_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem k_device_buf(sizeof(DataType) * k_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem v_device_buf(sizeof(DataType) * v_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem y_device_buf(sizeof(DataType) * y_gs_ms_os.mDesc.GetElementSpaceSize());
    DeviceMem lse_device_buf(sizeof(LSEDataType) * lse_gs_ms.mDesc.GetElementSpaceSize());
    DeviceMem qgrad_device_buf(sizeof(DataType) * q_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem kgrad_device_buf(sizeof(DataType) * k_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem vgrad_device_buf(sizeof(DataType) * v_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem ygrad_device_buf(sizeof(DataType) * y_gs_ms_os.mDesc.GetElementSpaceSize());

    q_device_buf.ToDevice(q_gs_ms_ks.mData.data());
    k_device_buf.ToDevice(k_gs_ns_ks.mData.data());
    v_device_buf.ToDevice(v_gs_os_ns.mData.data());
    y_device_buf.ToDevice(y_gs_ms_os.mData.data());
    lse_device_buf.ToDevice(lse_gs_ms.mData.data());
    ygrad_device_buf.ToDevice(ygrad_gs_ms_os.mData.data());
    kgrad_device_buf.SetZero();
    vgrad_device_buf.SetZero();

    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(
        static_cast<DataType*>(q_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(k_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(v_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(y_device_buf.GetDeviceBuffer()),
        static_cast<LSEDataType*>(lse_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(ygrad_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(qgrad_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(kgrad_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(vgrad_device_buf.GetDeviceBuffer()),
        {}, // std::array<void*, 1> p_acc0_biases;
        {}, // std::array<void*, 1> p_acc1_biases;
        q_gs_ms_ks_lengths,
        q_gs_ms_ks_strides,
        k_gs_ns_ks_lengths,
        k_gs_ns_ks_strides,
        v_gs_os_ns_lengths,
        v_gs_os_ns_strides,
        y_gs_ms_os_lengths,
        y_gs_ms_os_strides,
        lse_gs_ms_lengths,
        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
        QKVElementOp{},
        QKVElementOp{},
        Scale{alpha},
        QKVElementOp{},
        YElementOp{});

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    // 5 GEMM ops in total:
    // S_MNK / dP_MNO Gemm (Gemm0 rcr)
    // dQ_MKN Gemm (Gemm1 rrr)
    // dV_NOM / dK_NKM Gemm (Gemm2 crr)
    // 3x MNK + 2x MNO
    std::size_t flop = (size_t(3) * M * N * K + size_t(2) * M * N * O) * 2 * BatchCount;
    // Q/K/V/Y, dQ/dK/dV/dY, LSE
    std::size_t num_btype = (sizeof(DataType) * M * K + sizeof(DataType) * K * N +
                             sizeof(DataType) * N * O + sizeof(DataType) * M * O) *
                                size_t(2) * BatchCount +
                            sizeof(LSEDataType) * M * BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        kgrad_device_buf.SetZero(); // reset global accum buffer and rerun
        vgrad_device_buf.SetZero();
        invoker.Run(argument, StreamConfig{nullptr, false});

        Tensor<DataType> qgrad_g_m_k({BatchCount, M, K});
        Tensor<DataType> kgrad_g_n_k({BatchCount, N, K});
        Tensor<DataType> vgrad_g_n_o({BatchCount, N, O});
        Tensor<DataType> sgrad_g_m_n({BatchCount, M, N});
        Tensor<DataType> pgrad_g_m_n({BatchCount, M, N});
        Tensor<DataType> ygrad_g_m_o({BatchCount, M, O});
        Tensor<DataType> ygrad_dot_y_g_m({BatchCount, M});

        ygrad_gs_ms_os.ForEach([&](auto& self, auto idx) {
            ygrad_g_m_o(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });

#if PRINT_HOST
        {
            std::cout << "q_g_m_k ref:\n" << q_g_m_k;
            std::cout << "k_g_n_k ref:\n" << k_g_n_k;
            std::cout << "v_g_n_o ref:\n" << v_g_n_o;
            std::cout << "ygrad_g_m_o ref:\n" << ygrad_g_m_o;
        }
#endif

        // Gradients
        auto ref_gemm_grad         = ReferenceGemmGradInstance{};
        auto ref_gemm_grad_invoker = ref_gemm_grad.MakeInvoker();
        using RefGemmGradArg       = ReferenceGemmGradInstance::Argument;

        // dP = dY * V^T
        auto v_g_o_n = v_g_n_o.Transpose({0, 2, 1});
        ref_gemm_grad_invoker.Run(RefGemmGradArg{
            ygrad_g_m_o, v_g_o_n, pgrad_g_m_n, PassThrough{}, PassThrough{}, Scale{1.f}});
#if PRINT_HOST
        {
            std::cout << "===== dP = dY * V^T\n";
            std::cout << "ygrad_g_m_o ref:\n" << ygrad_g_m_o;
            std::cout << "v_g_o_n ref:\n" << v_g_o_n;
            std::cout << "pgrad_g_m_n ref:\n" << pgrad_g_m_n;
        }
#endif

        // dS_i_j = P_i_j .* (dP_i_j - dY_i dot Y_i)
        sgrad_g_m_n.ForEach([&](auto& self, auto idx_gmn) {
            float ygrad_dot_y = 0;
            for(int o = 0; o < O; o++)
            {
                auto idx_gmo = idx_gmn;
                idx_gmo[2]   = o;
                ygrad_dot_y += ygrad_g_m_o(idx_gmo) * y_g_m_o(idx_gmo);
            }
            self(idx_gmn) = p_g_m_n(idx_gmn) * (pgrad_g_m_n(idx_gmn) - ygrad_dot_y);
        });
#if PRINT_HOST
        {
            std::cout << "===== dS_i_j = P_i_j .* (dP_i_j - dY_i dot Y_i)\n";
            std::cout << "p_g_m_n ref:\n" << p_g_m_n;
            std::cout << "pgrad_g_m_n ref:\n" << pgrad_g_m_n;
            std::cout << "y_g_m_o ref:\n" << y_g_m_o;
            std::cout << "ygrad_g_m_o ref:\n" << ygrad_g_m_o;
            std::cout << "sgrad_g_m_n ref:\n" << sgrad_g_m_n;
        }
#endif

        // dV = P^T * dY
        auto p_g_n_m = p_g_m_n.Transpose({0, 2, 1});
        ref_gemm_grad_invoker.Run(RefGemmGradArg{
            p_g_n_m, ygrad_g_m_o, vgrad_g_n_o, PassThrough{}, PassThrough{}, Scale{1.f}});
#if PRINT_HOST
        {
            std::cout << "===== dV = P^T * dY\n";
            std::cout << "p_g_n_m ref:\n" << p_g_n_m;
            std::cout << "ygrad_g_m_o ref:\n" << ygrad_g_m_o;
            std::cout << "vgrad_g_n_o ref:\n" << vgrad_g_n_o;
        }
#endif

        // dQ = alpha * dS * K
        ref_gemm_grad_invoker.Run(RefGemmGradArg{
            sgrad_g_m_n, k_g_n_k, qgrad_g_m_k, PassThrough{}, PassThrough{}, Scale{alpha}});
#if PRINT_HOST
        {
            std::cout << "===== dQ = alpha * dS * K\n";
            std::cout << "sgrad_g_m_n ref:\n" << sgrad_g_m_n;
            std::cout << "k_g_n_k ref:\n" << k_g_n_k;
            std::cout << "qgrad_g_m_k ref:\n" << qgrad_g_m_k;
        }
#endif

        // dK = alpha * dS^T * Q
        auto sgrad_g_n_m = sgrad_g_m_n.Transpose({0, 2, 1});
        ref_gemm_grad_invoker.Run(RefGemmGradArg{
            sgrad_g_n_m, q_g_m_k, kgrad_g_n_k, PassThrough{}, PassThrough{}, Scale{alpha}});
#if PRINT_HOST
        {
            std::cout << "===== dK = alpha * dS^T * Q\n";
            std::cout << "sgrad_g_n_m ref:\n" << sgrad_g_n_m;
            std::cout << "q_g_m_k ref:\n" << q_g_m_k;
            std::cout << "kgrad_g_n_k ref:\n" << kgrad_g_n_k;
        }
#endif

        Tensor<DataType> qgrad_gs_ms_ks_host_result(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
        Tensor<DataType> kgrad_gs_ns_ks_host_result(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
        Tensor<DataType> vgrad_gs_os_ns_host_result(v_gs_os_ns_lengths, v_gs_os_ns_strides);

        Tensor<DataType> qgrad_gs_ms_ks_device_result(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
        Tensor<DataType> kgrad_gs_ns_ks_device_result(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
        Tensor<DataType> vgrad_gs_os_ns_device_result(v_gs_os_ns_lengths, v_gs_os_ns_strides);

        qgrad_device_buf.FromDevice(qgrad_gs_ms_ks_device_result.mData.data());
        kgrad_device_buf.FromDevice(kgrad_gs_ns_ks_device_result.mData.data());
        vgrad_device_buf.FromDevice(vgrad_gs_os_ns_device_result.mData.data());

        // permute
        qgrad_gs_ms_ks_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = qgrad_g_m_k(g, idx[2], idx[3]);
        });
        kgrad_gs_ns_ks_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = kgrad_g_n_k(g, idx[2], idx[3]);
        });
        vgrad_gs_os_ns_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = vgrad_g_n_o(g, idx[3], idx[2]);
        });

        std::cout << "Checking qgrad:\n";
        pass &= ck::utils::check_err(qgrad_gs_ms_ks_device_result.mData,
                                     qgrad_gs_ms_ks_host_result.mData,
                                     "error",
                                     1e-2,
                                     1e-2);
        std::cout << "Checking kgrad:\n";
        pass &= ck::utils::check_err(kgrad_gs_ns_ks_device_result.mData,
                                     kgrad_gs_ns_ks_host_result.mData,
                                     "error",
                                     1e-2,
                                     1e-2);
        std::cout << "Checking vgrad:\n";
        pass &= ck::utils::check_err(vgrad_gs_os_ns_device_result.mData,
                                     vgrad_gs_os_ns_host_result.mData,
                                     "error",
                                     1e-2,
                                     1e-2);
    }

    return pass ? ((void)(std::cout << "pass\n"), 0) : ((void)(std::cout << "fail\n"), 1);
}

int main(int argc, char* argv[]) { return run(argc, argv); }
