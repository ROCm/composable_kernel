// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <numeric>
#include <unordered_map>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/reference_tensor_operation/gpu/reference_gemm.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;
};

struct ProblemSizeSplitK final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = -1;
    ck::index_t StrideB = -1;
    ck::index_t StrideC = -1;

    ck::index_t KBatch = 1;
};

struct ExecutionConfig final
{
    // 0 - no verification, 1 - CPU, 2 - GPU, 3 - CPU + GPU
    int do_verification = 1;
    int init_method     = 2;
    bool time_kernel    = false;
    int instance_index  = -1;
    int cold_niters     = 50;
    int nrepeat         = 100;
    int rotating_count  = 4;
    int verbosity       = 1;
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename ProblemType>
bool parse_cmd_args(int, char*[], ProblemType&, ExecutionConfig&)
{
    return false;
}

template <>
bool parse_cmd_args<ProblemSize>(int argc,
                                 char* argv[],
                                 ProblemSize& problem_size,
                                 ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);
    }
    else
    {
        std::cerr
            << "arg1: verification (0=no, 1=CPU, 2=GPU, 3=CPU and GPU)" << std::endl
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)" << std::endl
            << "arg3: time kernel (0=no, 1=yes)" << std::endl
            << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC (default: -1 or 0)"
            << std::endl;
        return false;
    }

    return true;
}

template <>
bool parse_cmd_args<ProblemSizeSplitK>(int argc,
                                       char* argv[],
                                       ProblemSizeSplitK& problem_size,
                                       ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);

        if(argc >= 11)
        {
            problem_size.KBatch = std::stoi(argv[10]);
        }
        if(argc >= 12)
        {
            config.instance_index = std::stoi(argv[11]);
        }
        if(argc >= 13)
        {
            config.cold_niters = std::stoi(argv[12]);
        }
        if(argc >= 14)
        {
            config.nrepeat = std::stoi(argv[13]);
        }
        if(argc >= 15)
        {
            config.rotating_count = std::stoi(argv[14]);
        }
    }
    else
    {
        std::cerr
            << "arg1: verification (0=no, 1=CPU, 2=GPU, 3=CPU and GPU)" << std::endl
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)" << std::endl
            << "arg3: time kernel (0=no, 1=yes)" << std::endl
            << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC (default: -1 or 0)"
            << std::endl
            << "arg10: KBatch" << std::endl
            << "arg11-14(optional): instance_index warmup_iters repeat_iters rotating_count"
            << std::endl;
        return false;
    }

    return true;
}

template <typename DataType, typename ComputeDataType = DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float> && std::is_same_v<ComputeDataType, ck::tf32_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType, typename ComputeDataType = DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float> && std::is_same_v<ComputeDataType, ck::tf32_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <bool KLast>
void preShuffleScaleBuffer_gfx950(ck::e8m0_bexp_t* src, ck::e8m0_bexp_t* dst, int MN, int K)
{
    int MNXdlPack = 2;
    int KXdlPack  = 2;

    int XdlMNThread = 16;
    int XdlKThread  = 64 / XdlMNThread;

    int K0 = K / KXdlPack / XdlKThread; // KRepeat

    // On gfx950, WarpSize=64:
    // The 4 16x128 building blocks will be packed into 1 32x256
    // The 8 16x16x128 mfma will be packed into 1 32x32x256

    // unfold the MN32xK(256/32) scale buffer
    //    4            16             2           2
    // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
    // Then, MNRepeat->KRepeat

    for(int n = 0; n < MN; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
            int tempn = n % (XdlMNThread * MNXdlPack);
            int n1    = tempn % XdlMNThread; // i XdlMNThread
            int n2    = tempn / XdlMNThread; // i MNXdlPack

            int k0    = k / (XdlKThread * KXdlPack); // i KRepeat
            int tempk = k % (XdlKThread * KXdlPack);
            int k1    = tempk % XdlKThread; // i XdlKThread
            int k2    = tempk / XdlKThread; // i KXdlPack

            int outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                              k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                              k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                              k2 * MNXdlPack + n2;
            // src[n * K + k] = ck::type_convert<ck::e8m0_bexp_t>(static_cast<float>(powf(2.0f,
            // 2-k)));

            if constexpr(KLast)
                dst[outputIndex] = src[n * K + k];
            else
                dst[outputIndex] = src[k * MN + n];
        }
    }
}

/**
 * Pre-shuffle scale buffer for gfx1250 16x16x128 wmma scale instruction
 *
 * @tparam ScaleType Scale data type
 * @tparam KStride Whether K is the leading dimension of the scale buffer
 */
template <typename ScaleType, ck::index_t ScaleBlockSize, bool KStride>
void preShuffleScaleBuffer_gfx1250(const ScaleType* src,
                                   ScaleType* dst,
                                   ck::index_t MN,
                                   ck::index_t K)
{

    static_assert(ScaleBlockSize == 32 && sizeof(ScaleType) == 1,
                  "wrong! only support 8-bit scale with ScaleBlockSize=32");

    constexpr ck::index_t MPerXdlops = 16;
    // constexpr ck::index_t NPerXdlops = 16;
    constexpr ck::index_t KPerXdlops = 128;

    int MNPack = 2; // 2 sets of scales in M/N direction
    int KPack  = 1; // 1 set of scales in K direction

    int MNStep = MPerXdlops;
    int KStep  = KPerXdlops / ScaleBlockSize; // scales per thread

    int K0 = K / KPack / KStep; // KRepeat - how many KStep blocks

    // On gfx1250, WarpSize=32:
    // -- The 2 16x128 building blocks will be packed into 1 32x128
    // -- The 4 16x16x128 wmma will be packed into 1 32x32x128

    // unfold the MN32xK(128/32) scale buffer
    //    4            16        1        2
    // To KStep  ->  MNStep -> KPack -> MNPack
    // or ???
    //    2         16        1        4
    //  MNPack -> MNStep -> KPack -> KStep
    for(int mn = 0; mn < MN; ++mn)
    {
        int iMNRepeat = mn / (MNStep * MNPack); // i MNRepeat (MN block id)
        int tempmn    = mn % (MNStep * MNPack); // position in MN block

        for(int k = 0; k < K; ++k)
        {
            int iKRepeat = k / (KStep * KPack); // i KRepeat
            int tempk    = k % (KStep * KPack); // position in KStep block

            int outputIndex = (iMNRepeat * MNPack * MNStep) * (KStep * KPack * K0) +
                              (iKRepeat * KStep * KPack) * (MNStep * MNPack) +
                              tempmn * (KStep * KPack) + tempk;

            if constexpr(KStride)
            {
                dst[outputIndex] = src[mn * K + k];
            }
            else
                dst[outputIndex] = src[k * MN + mn];
        }
    }
}

template <typename T>
void preShuffleBuffer(const T* src, T* dst, int N, int K, int NXdl, int KPack)
{
    int NLane = NXdl;
    int KLane = ck::get_warp_size() / NLane;
    int K_pk  = std::is_same_v<T, ck::f4x2_pk_t> ? K / 2 : K;
    int K0    = K_pk / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K_pk; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K_pk + k];
        }
    }
}

float i4_to_f32_gfx9(uint8_t i4)
{
    static std::unordered_map<uint8_t, float> u = {{0b1000, -0.5000f},
                                                   {0b1001, -0.4375f},
                                                   {0b1010, -0.3750f},
                                                   {0b1011, -0.3125f},
                                                   {0b1100, -0.2500f},
                                                   {0b1101, -0.1875f},
                                                   {0b1110, -0.1250f},
                                                   {0b1111, -0.0625f},
                                                   {0b0, +0.0000f},
                                                   {0b1, +0.0625f},
                                                   {0b10, +0.1250f},
                                                   {0b11, +0.1875f},
                                                   {0b100, +0.2500f},
                                                   {0b101, +0.3125f},
                                                   {0b110, +0.3750f},
                                                   {0b111, +0.4375f}};

    return u[i4];
}

inline void permute_b_pk_i4(Tensor<ck::pk_i4_t>& b_k_n_permute,
                            int N,
                            int K,
                            Tensor<float>& b_k_n_f32,
                            Tensor<float>& b_k_n_gfx9_f32)
{
    for(int n = 0; n < N; n++)
    {
        for(int k = 0; k < K; k++)
        {
            ck::pk_i4_t i4x2 = b_k_n_permute(k, n).data;
            uint8_t i4       = 0;

            if(k % 2 == 1)
                i4 = (i4x2.data >> 0) & 0xf;
            else
                i4 = (i4x2.data >> 4) & 0xf;

            b_k_n_f32(k, n)      = (((i4 & 0x0f) >> 0) - 8.f);
            b_k_n_gfx9_f32(k, n) = i4_to_f32_gfx9(i4);
        }
    }

    // vector pk_i4x4 permute
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < K; j += 8)
        {
            int input[8];

            for(int k = 0; k < 4; k++)
            {
                int i4x2         = b_k_n_permute(j + k * 2, i).data;
                input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
            }

            // permute 01234567->20643175
            {
                int hi   = input[2];
                int lo   = input[0];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 0, i) = i4x2;
            }

            {
                int hi   = input[6];
                int lo   = input[4];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 2, i) = i4x2;
            }

            {
                int hi   = input[3];
                int lo   = input[1];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 4, i) = i4x2;
            }

            {
                int hi   = input[7];
                int lo   = input[5];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 6, i) = i4x2;
            }
        }
    }
}

inline void permute_a_pk_i4(Tensor<ck::pk_i4_t>& a_m_k_permute,
                            int M,
                            int K,
                            Tensor<float>& a_m_k_f32,
                            Tensor<float>& a_m_k_gfx9_f32)
{
    for(int m = 0; m < M; m++)
    {
        for(int k = 0; k < K; k++)
        {
            ck::pk_i4_t i4x2 = a_m_k_permute(m, k).data;
            uint8_t i4       = 0;

            if(k % 2 == 1)
                i4 = (i4x2.data >> 0) & 0xf;
            else
                i4 = (i4x2.data >> 4) & 0xf;

            a_m_k_f32(m, k)      = (((i4 & 0x0f) >> 0) - 8.f);
            a_m_k_gfx9_f32(m, k) = i4_to_f32_gfx9(i4);
        }
    }

    // vector pk_i4x4 permute
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < K; j += 8)
        {
            int input[8];

            for(int k = 0; k < 4; k++)
            {
                int i4x2         = a_m_k_permute(i, j + k * 2).data;
                input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
            }

            // permute 01234567->20643175
            {
                int hi   = input[2];
                int lo   = input[0];
                int i4x2 = (hi << 4) | lo;

                a_m_k_permute(i, j + 0) = i4x2;
            }

            {
                int hi   = input[6];
                int lo   = input[4];
                int i4x2 = (hi << 4) | lo;

                a_m_k_permute(i, j + 2) = i4x2;
            }

            {
                int hi   = input[3];
                int lo   = input[1];
                int i4x2 = (hi << 4) | lo;

                a_m_k_permute(i, j + 4) = i4x2;
            }

            {
                int hi   = input[7];
                int lo   = input[5];
                int i4x2 = (hi << 4) | lo;

                a_m_k_permute(i, j + 6) = i4x2;
            }
        }
    }
}
