// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp"
// #include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_moe_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I4  = ck::pk_i4_t;
using F16 = ck::half_t;
// using BF16 = ck::bhalf_t;
using F8  = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = F8;
using B0DataType       = I4;
using EDataType        = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F32;
using D1DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType>;

using A0Layout = Row;
using B0Layout = Col;
using ELayout  = Row;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;

// for gate, a_scale, b_scale
struct MulABScale
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<EDataType, float, float, float>
                                                                           (EDataType& e,
                                                                            const float& c,
                                                                            const float& d0,
                                                                            const float& d1) const
    {
        e = ck::type_convert<EDataType>(c * d1 * d0);
    }
};

// for gate, a_scale, b_scale, fuse silu, 
struct MulABScaleSilu
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<EDataType, float, float>
                                                                           (EDataType& e,
                                                                            const float& c,
                                                                            const float& d0,
                                                                            const float& d1) const
    {
        // act
        float x0 = 0;
        ck::tensor_operation::element_wise::Silu{}(x0, c * d1 * d0);
        e = ck::type_convert<EDataType>(x0);
    }
};

using CDEElementOp = MulABScale;

#if 1
void preShuffleBuffer(const B0DataType* src, B0DataType* dst, int N, int K, int NXdl)
{
    int KPack = 32;
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;
            
            dst[outputIndex / 2] = src[(n * K + k) / 2];
        }
    }
}
#endif

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

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
#if 1
static constexpr ck::index_t MPerBlock = 128;
static constexpr ck::index_t MNPerXDL = 32;
static constexpr ck::index_t CShuffleMXDLPerWave = MPerBlock / 32;
static constexpr ck::index_t KPerBlock = 128 / sizeof(A0DataType);
static constexpr ck::index_t MXDLPerWave = MPerBlock / 32; //todo fix this constraint
static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);
static constexpr ck::index_t BK1 = 32 / sizeof(B0DataType);
static constexpr ck::index_t EVec = 16 / sizeof(EDataType);
static constexpr ck::index_t D0Vec = 1;
static constexpr ck::index_t D1Vec = 1;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemm<
            Row, Col, DsLayout, ELayout, 
            A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
            AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
            256,   MPerBlock,   128,    KPerBlock,
            AK1,   BK1,
            MNPerXDL,   MNPerXDL,
            MXDLPerWave,    1,
            S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
            S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, BK1, BK1, 0,
            CShuffleMXDLPerWave,    1,   S<1, 32, 1, 8>, S<EVec, D0Vec, D1Vec>,
            ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, true, A0DataType>;
// clang-format on
#else
static constexpr ck::index_t MPerBlock = 16;
// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemm<
            Row, Col, DsLayout, ELayout, 
            A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
            AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
            64,   16,   16,    128,
            16,   32,
            16,   16,
            1,    1,
            S<8, 8, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
            S<4, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 32, 32, 0,
            1,    1,   S<1, 16, 1, 4>, S<4, 1, 1>,
            ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, true, A0DataType>;
// clang-format on
#endif

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;

// tokens = 1
// topk = 1
// experts = 8
// per expert: 
    // GEMM shape
    ck::index_t N = 6144;
    ck::index_t K = 8192;
    ck::index_t experts = 8;
    ck::index_t sorted_tile_num = 8;
    ck::index_t sorted_tile_size = MPerBlock;
    ck::index_t SORTED_SIZE = sorted_tile_num * sorted_tile_size;
    ck::index_t tokens = 128;
    // ck::index_t tokens = 16;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 6)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        N = std::stoi(argv[4]);
        K = std::stoi(argv[5]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf(
            "arg4 to 5: N, K\n");
        exit(0);
    }

    ck::index_t StrideA              = K;
    ck::index_t StrideB              = K;
    ck::index_t StrideE              = N;
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr auto StrideDs = std::array<ck::index_t, NumDTensor>{0, 0};

    ck::index_t KBatch = 1;

    // const ck::index_t experts = 8;
    Tensor<ck::index_t> expert_ids(HostTensorDescriptor({experts}, {1}));
    Tensor<ck::index_t> sorted_token_ids(HostTensorDescriptor({SORTED_SIZE}, {1}));
    for (int i = 0; i < sorted_tile_num; i++) {
        expert_ids.mData[i] = i;
    }
    int token_per_tile = tokens / sorted_tile_num;
    int tokenid = 0;
    // sorted_token_ids.mData[0] = 0;
    for (int i = 0; i < SORTED_SIZE; i++) {
        int tile_off = i % sorted_tile_size;
        if(tile_off < token_per_tile)
            sorted_token_ids.mData[i] = tokenid++;
        else
            sorted_token_ids.mData[i] = tokens;
    }
    expert_ids.savetxt("expert_ids.txt", "int");
    sorted_token_ids.savetxt("sorted_token_ids.txt", "int");
    Tensor<A0DataType> a0_t_k(HostTensorDescriptor({tokens, K}, {K, 1}));
    Tensor<B0DataType> b0_e_n_k(HostTensorDescriptor({experts, K, N}, {N*K, 1, K}));
    Tensor<B0DataType> b0_preshuffled(HostTensorDescriptor({experts, K, N}, {N*K, 1, K}));
    Tensor<D0DataType> d0_t_n(HostTensorDescriptor({tokens, N}, {StrideDs[0], 0}));
    Tensor<D1DataType> d1_e_n(HostTensorDescriptor({experts, N}, {1, StrideDs[1]}));
    Tensor<EDataType> e_m_n_host_result(HostTensorDescriptor({SORTED_SIZE, N}, {N, 1}));
    Tensor<EDataType> e_m_n_device_result(HostTensorDescriptor({SORTED_SIZE, N}, {N, 1}));

    std::cout << "a0_t_k: " << a0_t_k.mDesc << std::endl;
    std::cout << "b0_e_n_k: " << b0_e_n_k.mDesc << std::endl;
    std::cout << "d1_e_n: " << d1_e_n.mDesc << std::endl;
    std::cout << "d0_t_n: " << d0_t_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_t_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{0, 2});
        d0_t_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{1, 3});
        d1_e_n.GenerateTensorValue(GeneratorTensor_2<D1DataType>{1, 3});
        break;
    case 2:
        a0_t_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{1});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{1});
        d0_t_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
        d1_e_n.GenerateTensorValue(GeneratorTensor_1<D1DataType>{1});
        break;
    case 3:
        a0_t_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{1});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        d0_t_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
        d1_e_n.GenerateTensorValue(GeneratorTensor_1<D1DataType>{1});
        break;
    case 4:
        a0_t_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{1});
        d0_t_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
        d1_e_n.GenerateTensorValue(GeneratorTensor_1<D1DataType>{1});
        break;
    default:
        a0_t_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        d0_t_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        d1_e_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
    }
    d0_t_n.savetxt("d0_t_n.txt", "int");
    d1_e_n.savetxt("d1_e_n.txt", "int");
    DeviceMem sorted_token_ids_dev(sizeof(ck::index_t) * sorted_token_ids.mDesc.GetElementSpaceSize());
    DeviceMem expert_ids_dev(sizeof(ck::index_t) * expert_ids.mDesc.GetElementSpaceSize());
    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_t_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_e_n_k.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_t_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_e_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());
    a0_t_k.savetxt("a.txt");

    sorted_token_ids_dev.ToDevice(sorted_token_ids.mData.data());
    expert_ids_dev.ToDevice(expert_ids.mData.data());
    a0_device_buf.ToDevice(a0_t_k.mData.data());
    d0_device_buf.ToDevice(d0_t_n.mData.data());
    d1_device_buf.ToDevice(d1_e_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};

#if 1
    preShuffleBuffer(b0_e_n_k.mData.data(), b0_preshuffled.mData.data(), N * experts, K, device_op.GetPreShuffleParameters());
#else
    // weight pre-shuffle
    int KPack = 32; // int4 -> 32, fp8 -> 16, fp16 -> 8
    int NLane = device_op.GetPreShuffleParameters();
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int e = 0; e < experts; ++e)
    {
        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < K; ++k)
            {
                int n0 = n / NLane;
                int n1 = n % NLane;

                int k0 = k / (KLane * KPack);
                tempk  = k % (KLane * KPack);
                int k1 = tempk / KPack;
                int k2 = tempk % KPack;

                int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                                k1 * KPack * NLane + n1 * KPack + k2;
                
                b0_preshuffled(e, outputIndex % K, outputIndex / K) = b0_e_n_k(e, k, n);
            }
        }
    }
#endif

    // vector pk_i4x4 permute
    for(int e = 0; e < experts; e++)
    {
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < K; j += 8)
            {
                int input[8];

                for(int k = 0; k < 4; k++)
                {
                    int i4x2         = b0_preshuffled(e, j + k * 2, i).data;
                    input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                    input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
                }

                // permute 01234567->20643175
                {
                    int hi   = input[2];
                    int lo   = input[0];
                    int i4x2 = (hi << 4) | lo;

                    b0_preshuffled(e, j + 0, i) = i4x2;
                }

                {
                    int hi   = input[6];
                    int lo   = input[4];
                    int i4x2 = (hi << 4) | lo;

                    b0_preshuffled(e, j + 2, i) = i4x2;
                }

                {
                    int hi   = input[3];
                    int lo   = input[1];
                    int i4x2 = (hi << 4) | lo;

                    b0_preshuffled(e, j + 4, i) = i4x2;
                }

                {
                    int hi   = input[7];
                    int lo   = input[5];
                    int i4x2 = (hi << 4) | lo;

                    b0_preshuffled(e, j + 6, i) = i4x2;
                }
            }
        }
    }

    b0_device_buf.ToDevice(b0_preshuffled.mData.data());

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids_dev.GetDeviceBuffer(),
                               expert_ids_dev.GetDeviceBuffer(),
                               a0_device_buf.GetDeviceBuffer(),
                               b0_device_buf.GetDeviceBuffer(),
                               std::array<const void*, NumDTensor>{d0_device_buf.GetDeviceBuffer(),
                                                                   d1_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               tokens,
                               SORTED_SIZE,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               StrideDs,
                               StrideE,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    if (time_kernel) {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

        std::size_t flop = std::size_t(2) * SORTED_SIZE * N * K;
        std::size_t num_btype =
            sizeof(A0DataType) * SORTED_SIZE * K + sizeof(B0DataType) * K * N * experts + sizeof(EDataType) * SORTED_SIZE * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << device_op.GetTypeString() << std::endl;
    }

    if(do_verification)
    {
        invoker.Run(argument, StreamConfig{nullptr, false, 0 ,0,1});

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        Tensor<CShuffleDataType> c_m_n({SORTED_SIZE, N});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMoeGemm<A0DataType,
                                                                                   B0DataType,
                                                                                   CShuffleDataType,
                                                                                   AccDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   PassThrough>;
        auto ref_moe_gemm           = ReferenceGemmInstance{};
        auto ref_invoker            = ref_moe_gemm.MakeInvoker();

        auto ref_argument = ref_moe_gemm.MakeArgument(
           sorted_token_ids, expert_ids, sorted_tile_size, a0_t_k, b0_e_n_k, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);
        for(int m = 0; m < SORTED_SIZE; ++m)
        {
            
            const int t = sorted_token_ids(m);
            const int e = expert_ids(m / sorted_tile_size);
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d0_t_n(t, n), d1_e_n(e, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());
        e_m_n_device_result.savetxt("out.txt");
        e_m_n_host_result.savetxt("ref.txt");

#if 0
        printf("A Matrix:\n");
        for(int t = 0; t < tokens; t++)
        {
            for(int k = 0; k < K; k++)
            {
                printf("%f,", ck::type_convert<float>(a0_t_k(t, k)));
            }
            printf("\n");
        }
        printf("\n");

        printf("B Matrix:\n");
        for(int e = 0; e < experts; e++)
        {
            for(int n = 0; n < N; n++)
            {
                for(int k = 0; k < K; k++)
                {
                    ck::pk_i4_t i4x2 = b0_e_n_k(e, k, n).data;
                    int8_t i4        = 0;
                    if(k % 2 == 1)
                        i4 = (i4x2.data >> 0) & 0xf;
                    else
                        i4 = (i4x2.data >> 4) & 0xf;

                    printf("%f,", i4_to_f32_gfx9(i4));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");

        printf("B preshuflled Matrix:\n");
        for(int e = 0; e < experts; e++)
        {
            for(int n = 0; n < N; n++)
            {
                for(int k = 0; k < K; k++)
                {
                    ck::pk_i4_t i4x2 = b0_preshuffled(e, k, n).data;
                    int8_t i4        = 0;
                    if(k % 2 == 1)
                        i4 = (i4x2.data >> 0) & 0xf;
                    else
                        i4 = (i4x2.data >> 4) & 0xf;

                    printf("%f,", i4_to_f32_gfx9(i4));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");

        printf("C device Matrix:\n");
        for(int m = 0; m < SORTED_SIZE; m++)
        {
            for(int n = 0; n < N; n++)
            {
                printf("%f,", ck::type_convert<float>(e_m_n_device_result(m, n)));
            }
            printf("\n");
        }
        printf("\n");

        printf("C host Matrix:\n");
        for(int m = 0; m < SORTED_SIZE; m++)
        {
            for(int n = 0; n < N; n++)
            {
                printf("%f,", ck::type_convert<float>(e_m_n_host_result(m, n)));
            }
            printf("\n");
        }
#endif

        return ck::utils::check_err(
                   e_m_n_device_result, e_m_n_host_result, "Error: Incorrect results!", 1e-3, 5e-2)
                   ? 0
                   : 1;
    }
    printf("end of kernel\n");

    return 0;
}
