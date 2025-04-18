// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp"
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

using F16 = ck::half_t;
// using BF16 = ck::bhalf_t;
using F8  = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = F8;
using B0DataType       = F8;
using EDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F32;
using D1DataType       = F32;
using D2DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType, D2DataType>;

using A0Layout = Row;
using B0Layout = Col;
using ELayout  = Row;
using D0Layout = Row;
using D1Layout = Col;
using D2Layout = ELayout;
using DsLayout = ck::Tuple<D0Layout, D1Layout, D2Layout>;

// for gate, a_scale, b_scale
struct MulABScale
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<EDataType, float, float, float>(
        EDataType& e, const float& c, const float& d0, const float& d1) const
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
    __host__ __device__ constexpr void operator()<EDataType, float, float>(EDataType& e,
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

struct MulABScaleExpertWeight
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1, const D2& d2) const;
    // for real kernel use
    template <>
    __host__ __device__ constexpr void operator()<EDataType, float, float, float, float>(
        EDataType& e, const float& c, const float& d0, const float& d1, const float& d2) const
    {
        // for real kernel use
        // warning: hack hack hack here!!!! ignore d0 right now as kernel mul d0 * d2 outside.
        // tofix:felix
        (void)d2;
        e = ck::type_convert<EDataType>(c * d1 * d0);
    }
    // for reference cpu
    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float, float>(
        float& e, const float& c, const float& d0, const float& d1, const float& d2) const
    {
        // for reference cpu
        e = ck::type_convert<EDataType>(c * d0 * d1 * d2);
    }
};

using CDEElementOp = MulABScaleExpertWeight; // combine MulRoutedWeight = true
// using DsLayout = DsLayoutGate;
// using DsDataType       = DsDataTypeGate;
// using CDEElementOp = MulABScale; // combine MulRoutedWeight = false

// using CDEElementOp = MulABScaleSiluMulGate;

void preShuffleBuffer(const B0DataType* src, B0DataType* dst, int N, int K, int NXdl)
{
    int KPack = 16 / sizeof(B0DataType);
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

            dst[outputIndex] = src[n * K + k];
        }
    }
}
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough;
using BElementOp = PassThrough;

static constexpr auto GemmSpec         = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr ck::index_t MPerBlock = 128;
static constexpr ck::index_t MXDLPerWave = 2;
static constexpr ck::index_t NXDLPerWave = 2;
static constexpr ck::index_t BLOCKSIZE   = 256;
static constexpr ck::index_t NPerBlock   = 128;
static constexpr ck::index_t MNPerXDL    = 32;
static constexpr ck::index_t KPerBlock   = 128 / sizeof(A0DataType);
static constexpr ck::index_t Nswizzle    = true;
static constexpr bool MulRoutedWeight    = false;
static constexpr ck::index_t AK1         = 16 / sizeof(A0DataType);
static constexpr ck::index_t BK1         = 16 / sizeof(B0DataType);
static constexpr ck::index_t EVec        = 16 / sizeof(EDataType);
static constexpr ck::index_t D0Vec       = 1;
static constexpr ck::index_t D1Vec       = 1;
static constexpr ck::index_t D2Vec       = 1;
// using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemm
    // clang-format off
        <      Row,      Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
               AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
               //threadnum, mblock, nblock, kblock
               BLOCKSIZE,   MPerBlock,   NPerBlock,    KPerBlock,
               // ak1, bk1
               AK1,   BK1,
               // mn_perxdl
               MNPerXDL,   MNPerXDL,
               // mn_xdlperwave 
               MXDLPerWave,    NXDLPerWave,
               // a,b: loadtranfer cluster, cluster order, srcorder,VECDIM, srcpervec, dstpervec, lds_extra
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, BK1, BK1, 0,
               //    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
               //    MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
                //  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
                2,    1,   S<1, 32, 1, 8>, S<EVec, D0Vec, D1Vec, D2Vec>,
               ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, Nswizzle, true, MulRoutedWeight, A0DataType>;

// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;

    // GEMM shape
    ck::index_t N               = 4096;
    ck::index_t K               = 4096;
    ck::index_t experts         = 8;
    ck::index_t sorted_tile_num = 8;
    ck::index_t valid_tile_num  = 8;
    ck::index_t tokens          = 128;
    ck::index_t topk            = 2;

    // ck::index_t tokens = batch * topk;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 7)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        N               = std::stoi(argv[4]);
        K               = std::stoi(argv[5]);
        tokens          = std::stoi(argv[6]);
    }
    else if(argc == 9)
    {

        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        N               = std::stoi(argv[4]);
        K               = std::stoi(argv[5]);
        tokens          = std::stoi(argv[6]);
        sorted_tile_num = std::stoi(argv[7]);
        valid_tile_num  = std::stoi(argv[8]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 5: N, K, tokens\n");
        exit(0);
    }

    ck::index_t sorted_size = sorted_tile_num * MPerBlock;
    ck::index_t valid_size  = valid_tile_num * MPerBlock;
    if(tokens * topk > valid_size)
    {
        printf("err config, tokens * topk > valid_size\n");
        exit(-1);
    }
    ck::index_t StrideA              = K;
    ck::index_t StrideB              = K;
    ck::index_t StrideE              = N;
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr auto StrideDs          = std::array<ck::index_t, NumDTensor>{0, 0, 0};

    ck::index_t KBatch = 1;

    // const ck::index_t experts = 8;
    Tensor<ck::index_t> expert_ids(HostTensorDescriptor({sorted_tile_num}, {1}));
    Tensor<ck::index_t> sorted_token_ids(HostTensorDescriptor({sorted_size}, {1}));
    Tensor<ck::index_t> max_token_id(HostTensorDescriptor({1 + sorted_tile_num}));
    // max_token_id.mData =  {valid_size, 2, 2, 1, 1, 2, 2, 2,2, 2, 2, 2, 2,1,0,0,0};
    // max_token_id.mData =  {valid_size, 0, 2, 3, 4, 6, 8, 10, 12, 13};
    // int eids[] = {0, 0,1, 2,3, 3, 4,4, 5, 5, 6, 6, 7, 3, 3, 3}; // {2, 1, 1, 2, 2, 2, 1, 2}
    // max_token_id.mData = {valid_size, 0, 2, 3, 4, 6, 8, 10, 12, 13};
    // int eids[] = {0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 3, 3, 3}; // {2, 1, 1, 2, 2, 2, 1, 2}
    max_token_id.mData = {valid_size, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    int eids[]         = {0, 1, 2, 3, 4, 5, 6, 7, 3, 3, 3}; // {2, 1, 1, 2, 2, 2, 1, 2}
    for(int i = 0; i < sorted_tile_num; i++)
    {
        expert_ids.mData[i] = eids[i];
    }
    int token_per_tile = (tokens * topk + valid_tile_num - 1) / valid_tile_num;
    int tokenid        = 0;
    // sorted_token_ids.mData[0] = 0;
    for(int i = 0; i < sorted_size; i++)
    {
        int tile_off = i % MPerBlock;
        if(tile_off < token_per_tile && tokenid < tokens * topk)
        {
            sorted_token_ids.mData[i] = (tokenid % tokens) | ((tokenid / tokens) << 24);
            tokenid++;
        }
        else
        {
            sorted_token_ids.mData[i] = tokens;
        }
    }
    // expert_ids.savetxt("expert_ids.txt", "int");
    // sorted_token_ids.savetxt("sorted_token_ids.txt", "int");
    Tensor<A0DataType> a0_t_k(HostTensorDescriptor({tokens, K}, {K, 1}));
    Tensor<B0DataType> b0_e_n_k(HostTensorDescriptor({experts, K, N}, {N * K, 1, K}));
    Tensor<B0DataType> b0_preshuffled(HostTensorDescriptor({experts, K, N}, {N * K, 1, K}));
    Tensor<D0DataType> d0_t_n(HostTensorDescriptor({tokens, N}, {StrideDs[0], 0}));
    Tensor<D1DataType> d1_e_n(HostTensorDescriptor({experts, N}, {1, StrideDs[1]}));
    Tensor<D2DataType> d2_e_n(HostTensorDescriptor({sorted_size, N}, {1, 0}));
    Tensor<EDataType> e_t_n_host_result(HostTensorDescriptor({tokens, topk, N}, {topk * N, N, 1}));
    Tensor<EDataType> e_t_n_device_result(
        HostTensorDescriptor({tokens, topk, N}, {topk * N, N, 1}));
    std::cout << "a0_t_k: " << a0_t_k.mDesc << std::endl;
    std::cout << "b0_e_n_k: " << b0_e_n_k.mDesc << std::endl;
    std::cout << "d1_e_n: " << d1_e_n.mDesc << std::endl;
    std::cout << "d0_t_n: " << d0_t_n.mDesc << std::endl;
    std::cout << "d2_e_n: " << d2_e_n.mDesc << std::endl;
    std::cout << "e_t_n: " << e_t_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_t_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        d0_t_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-2, 2});
        d1_e_n.GenerateTensorValue(GeneratorTensor_2<D1DataType>{-2, 2});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{-2, 2});
        break;
    case 2:
        a0_t_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        d0_t_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{});
        d1_e_n.GenerateTensorValue(GeneratorTensor_1<D1DataType>{});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{});
        break;
    case 3:
        a0_t_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        d0_t_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{});
        d1_e_n.GenerateTensorValue(GeneratorTensor_1<D1DataType>{});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{});
        break;
    default:
        a0_t_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        d0_t_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        d1_e_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0.0, 1.0});
    }
    DeviceMem sorted_token_ids_dev(sizeof(ck::index_t) *
                                   sorted_token_ids.mDesc.GetElementSpaceSize());
    DeviceMem expert_ids_dev(sizeof(ck::index_t) * expert_ids.mDesc.GetElementSpaceSize());
    DeviceMem max_token_id_dev(sizeof(ck::index_t) * max_token_id.mDesc.GetElementSpaceSize());
    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_t_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_e_n_k.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_t_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_e_n.mDesc.GetElementSpaceSize());
    DeviceMem d2_device_buf(sizeof(D2DataType) * d2_e_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_t_n_device_result.mDesc.GetElementSpaceSize());
    // a0_t_k.savetxt("a.txt");
    // d0_t_n.savetxt("d0_t_n.txt", "int");
    // d1_e_n.savetxt("d1_e_n.txt", "int");
    sorted_token_ids_dev.ToDevice(sorted_token_ids.mData.data());
    expert_ids_dev.ToDevice(expert_ids.mData.data());
    max_token_id_dev.ToDevice(max_token_id.mData.data());
    a0_device_buf.ToDevice(a0_t_k.mData.data());
    d0_device_buf.ToDevice(d0_t_n.mData.data());
    d1_device_buf.ToDevice(d1_e_n.mData.data());
    d2_device_buf.ToDevice(d2_e_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};

    int NPerXdl = device_op.GetPreShuffleParameters();

    preShuffleBuffer(b0_e_n_k.mData.data(), b0_preshuffled.mData.data(), N * experts, K, NPerXdl);

    b0_device_buf.ToDevice(b0_preshuffled.mData.data());

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids_dev.GetDeviceBuffer(),
                               expert_ids_dev.GetDeviceBuffer(),
                               max_token_id_dev.GetDeviceBuffer(),
                               a0_device_buf.GetDeviceBuffer(),
                               b0_device_buf.GetDeviceBuffer(),
                               std::array<const void*, NumDTensor>{d0_device_buf.GetDeviceBuffer(),
                                                                   d1_device_buf.GetDeviceBuffer(),
                                                                   d2_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               tokens,
                               topk,
                               sorted_size,
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
    if(time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

        std::size_t flop      = std::size_t(2) * tokens * topk * N * K;
        std::size_t num_btype = sizeof(A0DataType) * valid_tile_num * K +
                                sizeof(B0DataType) * K * N * experts +
                                sizeof(EDataType) * valid_tile_num * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << std::endl;
    }

    if(do_verification)
    {
        invoker.Run(argument, StreamConfig{nullptr, false, 0, 0, 1});

        e_device_buf.FromDevice(e_t_n_device_result.mData.data());

        Tensor<CShuffleDataType> c_t_k_n({tokens, topk, N}, {topk * N, N, 1});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMoeGemm<A0DataType,
                                                                                   B0DataType,
                                                                                   CShuffleDataType,
                                                                                   D2DataType,
                                                                                   AccDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   MulRoutedWeight>;
        auto ref_moe_gemm           = ReferenceGemmInstance{};
        auto ref_invoker            = ref_moe_gemm.MakeInvoker();

        auto ref_argument = ref_moe_gemm.MakeArgument(sorted_token_ids,
                                                      expert_ids,
                                                      max_token_id,
                                                      MPerBlock,
                                                      a0_t_k,
                                                      b0_e_n_k,
                                                      c_t_k_n,
                                                      d2_e_n,
                                                      PassThrough{},
                                                      PassThrough{},
                                                      PassThrough{});

        ref_invoker.Run(ref_argument);
        for(int m = 0; m < valid_size; ++m)
        {

            const int fuse_t  = sorted_token_ids.mData[m];
            const int t       = fuse_t & 0xffffff;
            const int topk_id = (fuse_t & 0xff000000) >> 24;

            if(t >= tokens)
            {
                continue;
            }
            const int e = expert_ids(m / MPerBlock);
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_t_n_host_result(t, topk_id, n),
                               c_t_k_n(t, topk_id, n),
                               d0_t_n(t, n),
                               d1_e_n(e, n),
                               1.f);
            }
        }

        e_device_buf.FromDevice(e_t_n_device_result.mData.data());
        // e_t_n_device_result.savetxt("out.txt");
        // e_t_n_host_result.savetxt("ref.txt");
        return ck::utils::check_err(
                   e_t_n_device_result, e_t_n_host_result, "Error: Incorrect results!", 1e-3, 5e-2)
                   ? 0
                   : 1;
    }

    return 0;
}
