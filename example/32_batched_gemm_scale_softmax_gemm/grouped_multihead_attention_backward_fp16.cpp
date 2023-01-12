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
#define USING_MASK 1

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_multihead_attention_backward_xdl_cshuffle.hpp"
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
    ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle<
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
        32,          // KPerBlock
        128,         // Gemm1NPerBlock
        64,          // Gemm1KPerBlock
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
        4,              // CShuffleNXdlPerWavePerShuffle
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
    float K = 128;
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
    else if(argc == 7)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        alpha = std::stof(argv[4]);

        input_permute  = std::stoi(argv[5]);
        output_permute = std::stoi(argv[6]);
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

    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    std::vector<DeviceGemmInstance::ProblemDesc> problem_descs;

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<const DataType*> p_q;
    std::vector<const DataType*> p_k;
    std::vector<const DataType*> p_v;
    std::vector<const DataType*> p_y;
    std::vector<const LSEDataType*> p_lse;
    std::vector<DataType*> p_qgrad;
    std::vector<DataType*> p_kgrad;
    std::vector<DataType*> p_vgrad;
    std::vector<const DataType*> p_ygrad;

    std::vector<Tensor<DataType>> q_g_m_ks;
    std::vector<Tensor<DataType>> k_g_n_ks;
    std::vector<Tensor<DataType>> v_g_n_os;
    std::vector<Tensor<AccDataType>> s_g_m_ns;
    std::vector<Tensor<DataType>> p_g_m_ns;
    std::vector<Tensor<DataType>> y_g_m_os;
    std::vector<Tensor<DataType>> q_tensors;
    std::vector<Tensor<DataType>> k_tensors;
    std::vector<Tensor<DataType>> v_tensors;
    std::vector<Tensor<DataType>> y_tensors;
    std::vector<Tensor<LSEDataType>> lse_tensors;
    std::vector<Tensor<DataType>> qgrad_tensors;
    std::vector<Tensor<DataType>> kgrad_tensors;
    std::vector<Tensor<DataType>> vgrad_tensors;
    std::vector<Tensor<DataType>> ygrad_tensors;

    std::vector<DeviceMemPtr> q_tensors_device;
    std::vector<DeviceMemPtr> k_tensors_device;
    std::vector<DeviceMemPtr> v_tensors_device;
    std::vector<DeviceMemPtr> y_tensors_device;
    std::vector<DeviceMemPtr> lse_tensors_device;
    std::vector<DeviceMemPtr> qgrad_tensors_device;
    std::vector<DeviceMemPtr> ygrad_tensors_device;
    std::vector<DeviceMemPtr> kgrad_tensors_device;
    std::vector<DeviceMemPtr> vgrad_tensors_device;
    std::size_t group_count = 3;
    std::size_t flop = 0, num_byte = 0;
    for(std::size_t i=0; i<group_count; i++){
        int M  = 128 * (rand() % 4 + 1);
        int N  = 128 * (rand() % 4 + 1);
        int K  = 128;
        int O  = 128;
        int G0 = rand() % 3 + 1;
        int G1 = rand() % 2 + 1;

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
        problem_descs.push_back({
            q_gs_ms_ks_lengths,
            q_gs_ms_ks_strides,
            k_gs_ns_ks_lengths,
            k_gs_ns_ks_strides,
            v_gs_os_ns_lengths,
            v_gs_os_ns_strides,
            y_gs_ms_os_lengths,
            y_gs_ms_os_strides,
            lse_gs_ms_lengths,
            lse_gs_ms_strides,
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
        });

        int BatchCount = G0 * G1;
        flop += (size_t(3) * M * N * K + size_t(2) * M * N * O) * 2 * BatchCount;
        // Q/K/V/Y, dQ/dK/dV/dY, LSE
        num_byte += (sizeof(DataType) * M * K + sizeof(DataType) * K * N +
                             sizeof(DataType) * N * O + sizeof(DataType) * M * O) *
                                size_t(2) * BatchCount +
                            sizeof(LSEDataType) * M * BatchCount;

        Tensor<DataType> q_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
        Tensor<DataType> k_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
        Tensor<DataType> v_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);
        Tensor<DataType> y_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
        Tensor<DataType> ygrad_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);
        Tensor<LSEDataType> lse_gs_ms(lse_gs_ms_lengths, lse_gs_ms_strides);
        if(i < 4){
            std::cout << "q_gs_ms_ks: " << q_gs_ms_ks.mDesc << std::endl;
            std::cout << "k_gs_ns_ks: " << k_gs_ns_ks.mDesc << std::endl;
            std::cout << "v_gs_os_ns: " << v_gs_os_ns.mDesc << std::endl;
            std::cout << "y_gs_ms_os: " << y_gs_ms_os.mDesc << std::endl;
            std::cout << "lse_gs_ms_os: " << lse_gs_ms.mDesc << std::endl;
        }
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

        q_g_m_ks.push_back(q_g_m_k);
        k_g_n_ks.push_back(k_g_n_k);
        v_g_n_os.push_back(v_g_n_o);
        s_g_m_ns.push_back(s_g_m_n);
        p_g_m_ns.push_back(p_g_m_n);
        y_g_m_os.push_back(y_g_m_o);
        q_tensors.push_back(q_gs_ms_ks);
        k_tensors.push_back(k_gs_ns_ks);
        v_tensors.push_back(v_gs_os_ns);
        y_tensors.push_back(y_gs_ms_os);
        lse_tensors.push_back(lse_gs_ms);
        ygrad_tensors.push_back(ygrad_gs_ms_os);
        q_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(DataType) * q_gs_ms_ks.GetElementSpaceSize()));
        k_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(DataType) * k_gs_ns_ks.GetElementSpaceSize()));
        v_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(DataType) * v_gs_os_ns.GetElementSpaceSize()));
        y_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(DataType) * y_gs_ms_os.GetElementSpaceSize()));
        lse_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(LSEDataType) * lse_gs_ms.GetElementSpaceSize()));
        qgrad_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(DataType) * q_gs_ms_ks.GetElementSpaceSize()));
        kgrad_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(DataType) * k_gs_ns_ks.GetElementSpaceSize()));
        vgrad_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(DataType) * v_gs_os_ns.GetElementSpaceSize()));
        ygrad_tensors_device.emplace_back(std::make_unique<DeviceMem>(sizeof(DataType) * y_gs_ms_os.GetElementSpaceSize()));
        q_tensors_device.back()->ToDevice(q_gs_ms_ks.data());
        k_tensors_device.back()->ToDevice(k_gs_ns_ks.data());
        v_tensors_device.back()->ToDevice(v_gs_os_ns.data());
        y_tensors_device.back()->ToDevice(y_gs_ms_os.data());
        lse_tensors_device.back()->ToDevice(lse_gs_ms.data());
        qgrad_tensors_device.back()->SetZero();
        kgrad_tensors_device.back()->SetZero();
        vgrad_tensors_device.back()->SetZero();
        ygrad_tensors_device.back()->ToDevice(ygrad_gs_ms_os.data());
        p_q.push_back(static_cast<DataType*>(q_tensors_device.back()->GetDeviceBuffer()));        
        p_k.push_back(static_cast<DataType*>(k_tensors_device.back()->GetDeviceBuffer()));        
        p_v.push_back(static_cast<DataType*>(v_tensors_device.back()->GetDeviceBuffer()));        
        p_y.push_back(static_cast<DataType*>(y_tensors_device.back()->GetDeviceBuffer()));        
        p_lse.push_back(static_cast<LSEDataType*>(lse_tensors_device.back()->GetDeviceBuffer()));
        p_kgrad.push_back(static_cast<DataType*>(kgrad_tensors_device.back()->GetDeviceBuffer()));
        p_vgrad.push_back(static_cast<DataType*>(vgrad_tensors_device.back()->GetDeviceBuffer()));
        p_ygrad.push_back(static_cast<DataType*>(ygrad_tensors_device.back()->GetDeviceBuffer()));
        p_qgrad.push_back(static_cast<DataType*>(qgrad_tensors_device.back()->GetDeviceBuffer()));
    }
    auto argument = gemm.MakeArgument(
        p_q,
        p_k,
        p_v,
        p_y,
        p_lse,
        p_ygrad,
        p_qgrad,
        p_kgrad,
        p_vgrad,
        {}, // std::array<void*, 1> p_acc0_biases;
        {}, // std::array<void*, 1> p_acc1_biases;
        problem_descs,
        QKVElementOp{},
        QKVElementOp{},
        Scale{alpha},
        QKVElementOp{},
        YElementOp{});
        
    DeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

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
        for(int i=0;i<group_count;i++){
            qgrad_tensors_device[i]->SetZero();
            kgrad_tensors_device[i]->SetZero();
            vgrad_tensors_device[i]->SetZero();
        }
        invoker.Run(argument, StreamConfig{nullptr, false});
        for(std::size_t i=0; i<group_count; i++){

            int G0 = v_tensors[i].GetLengths()[0];
            int G1 = v_tensors[i].GetLengths()[1];
            int O = v_tensors[i].GetLengths()[2];
            int N = v_tensors[i].GetLengths()[3];
            int M = q_tensors[i].GetLengths()[2];
            int K = q_tensors[i].GetLengths()[3];
            int BatchCount = G0 * G1;
            Tensor<DataType> qgrad_g_m_k({BatchCount, M, K});
            Tensor<DataType> kgrad_g_n_k({BatchCount, N, K});
            Tensor<DataType> vgrad_g_n_o({BatchCount, N, O});
            Tensor<DataType> sgrad_g_m_n({BatchCount, M, N});
            Tensor<DataType> pgrad_g_m_n({BatchCount, M, N});
            Tensor<DataType> ygrad_g_m_o({BatchCount, M, O});
            Tensor<DataType> ygrad_dot_y_g_m({BatchCount, M});
            ygrad_tensors[i].ForEach([&](auto& self, auto idx) {
                ygrad_g_m_o(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
            });
            auto ref_gemm_grad         = ReferenceGemmGradInstance{};
            auto ref_gemm_grad_invoker = ref_gemm_grad.MakeInvoker();
            using RefGemmGradArg       = ReferenceGemmGradInstance::Argument;
            // dP = dY * V^T
            auto v_g_o_n = v_g_n_os[i].Transpose({0, 2, 1});
            ref_gemm_grad_invoker.Run(RefGemmGradArg{
                ygrad_g_m_o, v_g_o_n, pgrad_g_m_n, PassThrough{}, PassThrough{}, Scale{1.f}});
            sgrad_g_m_n.ForEach([&](auto& self, auto idx_gmn) {
                float ygrad_dot_y = 0;
                for(int o = 0; o < O; o++)
                {
                    auto idx_gmo = idx_gmn;
                    idx_gmo[2]   = o;
                    ygrad_dot_y += ygrad_g_m_o(idx_gmo) * y_g_m_os[i](idx_gmo);
                }
                self(idx_gmn) = p_g_m_ns[i](idx_gmn) * (pgrad_g_m_n(idx_gmn) - ygrad_dot_y);
            });
            auto p_g_n_m = p_g_m_ns[i].Transpose({0, 2, 1});
            ref_gemm_grad_invoker.Run(RefGemmGradArg{
                p_g_n_m, ygrad_g_m_o, vgrad_g_n_o, PassThrough{}, PassThrough{}, Scale{1.f}});
            ref_gemm_grad_invoker.Run(RefGemmGradArg{
                sgrad_g_m_n, k_g_n_ks[i], qgrad_g_m_k, PassThrough{}, PassThrough{}, Scale{alpha}});
            auto sgrad_g_n_m = sgrad_g_m_n.Transpose({0, 2, 1});
            ref_gemm_grad_invoker.Run(RefGemmGradArg{
                sgrad_g_n_m, q_g_m_ks[i], kgrad_g_n_k, PassThrough{}, PassThrough{}, Scale{alpha}});

            Tensor<DataType> qgrad_gs_ms_ks_host_result(q_tensors[i].GetLengths(), q_tensors[i].GetStrides());
            Tensor<DataType> kgrad_gs_ns_ks_host_result(k_tensors[i].GetLengths(), k_tensors[i].GetStrides());
            Tensor<DataType> vgrad_gs_os_ns_host_result(v_tensors[i].GetLengths(), v_tensors[i].GetStrides());

            Tensor<DataType> qgrad_gs_ms_ks_device_result(q_tensors[i].GetLengths(), q_tensors[i].GetStrides());
            Tensor<DataType> kgrad_gs_ns_ks_device_result(k_tensors[i].GetLengths(), k_tensors[i].GetStrides());
            Tensor<DataType> vgrad_gs_os_ns_device_result(v_tensors[i].GetLengths(), v_tensors[i].GetStrides());

            qgrad_tensors_device[i]->FromDevice(qgrad_gs_ms_ks_device_result.data());
            kgrad_tensors_device[i]->FromDevice(kgrad_gs_ns_ks_device_result.data());
            vgrad_tensors_device[i]->FromDevice(vgrad_gs_os_ns_device_result.data());
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
    }

    return pass ? ((void)(std::cout << "pass\n"), 0) : ((void)(std::cout << "fail\n"), 1);
}

int main(int argc, char* argv[]) { return run(argc, argv); }
