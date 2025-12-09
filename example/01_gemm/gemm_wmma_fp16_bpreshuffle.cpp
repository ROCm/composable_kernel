// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_preshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/utility/scheduler_enum.hpp"

#include <cstddef>
#include <iostream>
#include <type_traits>

using F16 = ck::half_t;
using F32 = float;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = F16;
using ComputeTypeA     = F16;
using ComputeTypeB     = F16;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr bool PermuteA = false;
static constexpr bool PermuteB = false;
static constexpr int KPack     = 8; // int4 -> 32, fp8 -> 16, fp16 -> 8
// clang-format off
using DeviceOpInstance = 
    ck::tensor_operation::device::DeviceGemm_Wmma_CShuffleV3_BPreshuffle<
        ALayout,   BLayout,  CLayout,   
        ADataType, BDataType, CDataType, AccDataType, CShuffleDataType, 
        AElementOp, BElementOp, CElementOp, GemmDefault, 
        128,
        32, 128, 128,
        8, 8,
        16, 16,
        2, 2,
        S<16, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<16, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        1, 1, S<1, 16, 1, 8>, S<4, 4, 1>,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, ComputeTypeA, ComputeTypeB, PermuteA, PermuteB>;
// clang-format on

template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> b_k_n_preshuffled(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{0, 2});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "b_k_n_preshuffled: " << b_k_n_preshuffled.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    // do GEMM
    auto device_op = DeviceOpInstance{};

    // weight pre-shuffle
    int NPerWmma = device_op.GetPreShuffleParameters();
    int KLane    = ck::get_warp_size() / NPerWmma;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NPerWmma
    // N, K -> N0 K0 KLane NPerWmma KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NPerWmma;
            int n1 = n % NPerWmma;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NPerWmma * KLane * K0 + k0 * KPack * NPerWmma * KLane +
                              k1 * KPack * NPerWmma + n1 * KPack + k2;

            b_k_n_preshuffled(outputIndex) = b_k_n(n * K + k);
        }
    }

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_preshuffled.mData.data());
    c_m_n_device_buf.ToDevice(c_m_n_device_result.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    auto invoker = device_op.MakeInvoker();

    auto argument =
        device_op.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                               static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                               static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               StrideC,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               c_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        std::cerr << device_op.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    float ave_time =
        invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 0, 50, 50, false, 1});

    bool pass = true;
    if(config.do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        invoker.Run(argument, StreamConfig{nullptr, false, 0});
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());
    }

    if(config.time_kernel)
    {
        ave_time =
            invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 0, 20, 50, true, 50});

        std::size_t flop = 2_uz * M * N * K;
        std::size_t num_btype =
            sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << device_op.GetTypeString() << std::endl;
    }

    return pass;
}

bool run_gemm_splitk_example(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size{3840, 4096, 4096, 4096, 4096, 4096, 1};
    ExecutionConfig config;

    return parse_cmd_args(argc, argv, problem_size, config) && run_gemm(problem_size, config);
}

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
