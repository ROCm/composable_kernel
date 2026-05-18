// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_lds_direct_load.hpp"
#include "gemm_xdl_ck_tile_wrap.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using bf16        = ck::bhalf_t;
using fp16        = ck::half_t;
using index_t     = ck::index_t;
using fp8         = ck::f8_t;
using bf8         = ck::bf8_t;
using i8          = int8_t;
using pk_i4       = ck::pk_i4_t;
using pk_fp4      = ck::f4x2_pk_t;

using ADataType = PREC_DATATYPE;
using BDataType = PREC_DATATYPE;

template <typename PreDataType>
constexpr auto GetAccDataType()
{
    if constexpr(ck::is_same_v<PreDataType, int8_t>)
    {
        return int32_t{};
    }
    else
    {
        return float{};
    }
}
template <typename PreDataType>
constexpr auto GetCDataType()
{
    if constexpr(ck::is_same_v<PreDataType, int8_t>)
    {
        return int32_t{};
    }
    else
    {
        return ck::half_t{};
    }
}

template <typename PreDataType>
constexpr auto GetComputeDataType()
{
    if constexpr(ck::is_same_v<PreDataType, ck::pk_i4_t>)
    {
        return ck::f8_t{};
    }
    else
    {
        return PreDataType{};
    }
}

using AccDataType      = decltype(GetAccDataType<ADataType>());
using CShuffleDataType = decltype(GetCDataType<ADataType>());
using CDataType        = decltype(GetCDataType<ADataType>());
using ComputeDataType  = decltype(GetComputeDataType<ADataType>());

using ALayout = A_LAYOUT;
using BLayout = B_LAYOUT;
using CLayout = Row;

using AElementOp               = PassThrough;
using BElementOp               = PassThrough;
using CElementOp               = PassThrough;
static constexpr auto DataSize = sizeof(ADataType);

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
template <index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          index_t MinimumOccupancy>
using GemmV3 = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    PassThrough,
    PassThrough,
    PassThrough,
    GemmSpec,
    BlockSize,
    MPerBlock,
    NPerBlock,
    KPerBlock,
    AK1,
    BK1,
    MPerXDL,
    NPerXDL,
    MXdlPerWave,
    NXdlPerWave,
    ABlockTransferThreadClusterLengths_AK0_M_AK1,
    ABlockTransferThreadClusterArrangeOrder,
    ABlockTransferSrcAccessOrder,
    ABlockTransferSrcVectorDim,
    ABlockTransferSrcScalarPerVector,
    ABlockTransferDstScalarPerVector_AK1,
    0,
    BBlockTransferThreadClusterLengths_BK0_N_BK1,
    BBlockTransferThreadClusterArrangeOrder,
    BBlockTransferSrcAccessOrder,
    BBlockTransferSrcVectorDim,
    BBlockTransferSrcScalarPerVector,
    BBlockTransferDstScalarPerVector_BK1,
    0,
    CShuffleMXdlPerWavePerShuffle,
    CShuffleNXdlPerWavePerShuffle,
    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    CShuffleBlockTransferScalarPerVector_NPerBlock,
    BlkGemmPipeSched,
    BlkGemmPipelineVer,
    ComputeDataType,
    ComputeDataType,
    false,
    false,
    MinimumOccupancy>;

template <index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched,
          ck::PipelineVersion PipelineVer,
          index_t MinimumOccupancy>
using GemmAsync = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_LdsDirectLoad<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    PassThrough,
    PassThrough,
    PassThrough,
    GemmSpec,
    1,
    BlockSize,
    MPerBlock,
    NPerBlock,
    KPerBlock,
    AK1,
    BK1,
    MPerXDL,
    NPerXDL,
    MXdlPerWave,
    NXdlPerWave,
    ABlockTransferThreadClusterLengths_AK0_M_AK1,
    ABlockTransferSrcAccessOrder,
    ABlockTransferSrcVectorDim,
    ABlockTransferSrcScalarPerVector,
    1,
    BBlockTransferThreadClusterLengths_BK0_N_BK1,
    BBlockTransferSrcAccessOrder,
    BBlockTransferSrcVectorDim,
    BBlockTransferSrcScalarPerVector,
    1,
    CShuffleMXdlPerWavePerShuffle,
    CShuffleNXdlPerWavePerShuffle,
    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    CShuffleBlockTransferScalarPerVector_NPerBlock,
    LoopSched,
    ck::PipelineVersion::v4,
    ComputeDataType,
    MinimumOccupancy>;

template <index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t KPerXDL,
          index_t MWarp,
          index_t NWarp,
          index_t CShuffleNXdlPerWavePerShuffle,
          ck_tile::GemmPipelineScheduler PipelineScheduler,
          ck_tile::GemmPipeline PipelineVer,
          index_t ClusterSizeM,
          index_t ClusterSizeN,
          index_t MinimumOccupancy>
using GemmCkTile =
    ck::tensor_operation::device::DeviceGemm_Xdl_CkTileWrap<ALayout,
                                                            BLayout,
                                                            CLayout,
                                                            ADataType,
                                                            BDataType,
                                                            CDataType,
                                                            AccDataType,
                                                            CShuffleDataType,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough,
                                                            ck_tile::sequence<false, false, false>,
                                                            MPerBlock,
                                                            NPerBlock,
                                                            KPerBlock,
                                                            MPerXDL,
                                                            NPerXDL,
                                                            KPerXDL,
                                                            MWarp,
                                                            NWarp,
                                                            1,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            ComputeDataType,
                                                            ClusterSizeM,
                                                            ClusterSizeN,
                                                            PipelineScheduler,
                                                            PipelineVer,
                                                            MinimumOccupancy>;
#if CK_TILE_USE_WMMA
#if defined(CK_USE_GFX1250)
static constexpr ck::index_t KPerXDL = 64 / DataSize;
#else
static constexpr ck::index_t KPerXDL = 16;
#endif
#else
#if defined(CK_GFX950_SUPPORT)
static constexpr ck::index_t KPerXDL =
    std::is_same_v<ADataType, ck::f8_t> || std::is_same_v<ADataType, ck::bf8_t> ? 128 : 32;
#else
static constexpr ck::index_t KPerXDL = 32;
#endif
#endif

struct GemmConfig
{
    static constexpr index_t M_Warp_Tile = 16;
    static constexpr index_t N_Warp_Tile = 16;
    static constexpr index_t K_Warp_Tile = KPerXDL;
};

// pk int4 32
static constexpr ck::index_t AB_K1 =
    ck::math::max(static_cast<ck::index_t>(16 / DataSize), static_cast<ck::index_t>(8));
static constexpr ck::index_t KPack = AB_K1;
// clang-format off
             // Block|  MPer|  NPer|  KPer|             AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer| Block-wiseGemm|  Block-wiseGemm|
             //  Size| Block| Block| Block|                |    | XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|       Pipeline|        Pipeline|
             //      |      |      |      |                |    |    |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1| Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|      Scheduler|        Verision|
             //      |      |      |      |                |    |    |     |     |     |                |               |               |               |               |               |                |               |               |              |               |               |            |            |                                 |                |               |                |
#define GEMM_RCR_INSTANCE(GemmClass, Scheduler, Version, Occupancy)  \
        GemmClass<256,   256,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,   8,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,   4,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,   4,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   128,  256 / DataSize, AB_K1, AB_K1, 16, 16,   4,    4,     S<16,16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<16,16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   256,   256,  256 / DataSize, AB_K1, AB_K1, 16, 16,   8,   16,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   256,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,   8,   16,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   128,  256 / DataSize, AB_K1, AB_K1, 16, 16,   4,    8,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,   8,    8,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,   4,    8,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   64,    128,  128 / DataSize, AB_K1, AB_K1, 16, 16,   4,    4,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,          AB_K1,          AB_K1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1,  8>,               8,  Scheduler, Version, Occupancy>

#define GEMM_RRR_INSTANCE(GemmClass, Scheduler, Version, Occupancy)  \
        GemmClass<256,   256,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             4,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   128,  256 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<16,16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   256,   256,  256 / DataSize, AB_K1, AB_K1, 16, 16,    8,   16,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   256,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,   16,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   128,  256 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,    8,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,    64,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,          AB_K1,          AB_K1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           2,                   S<1,16, 1,  8>,               8,  Scheduler, Version, Occupancy>
        
#define GEMM_CRR_INSTANCE(GemmClass, Scheduler, Version, Occupancy) \
        GemmClass<256,   256,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,    8,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,          AB_K1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,          AB_K1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             4,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   128,  256 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   256,   256,  256 / DataSize, AB_K1, AB_K1, 16, 16,    8,   16,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   256,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,   16,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   128,  256 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,    8,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,    64,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,          AB_K1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,            1,             8,          AB_K1,         2,           2,                   S<1,16, 1,  8>,               8,  Scheduler, Version, Occupancy>

#define GEMM_CCR_INSTANCE(GemmClass, Scheduler, Version, Occupancy)  \
        GemmClass<256,   256,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,    8,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,          AB_K1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,          AB_K1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<256,   128,   128,  256 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<16,16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   256,   256,  256 / DataSize, AB_K1, AB_K1, 16, 16,    8,    16,    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   256,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,    16,    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   128,  256 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   256,  128 / DataSize, AB_K1, AB_K1, 16, 16,    8,    8,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,   128,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    8,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,          AB_K1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           4,                   S<1, 8, 1, 16>,               8,  Scheduler, Version, Occupancy>, \
        GemmClass<128,    64,   128,  128 / DataSize, AB_K1, AB_K1, 16, 16,    4,    4,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,          AB_K1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,         AB_K1,          AB_K1,         2,           2,                   S<1,16, 1,  8>,               8,  Scheduler, Version, Occupancy>

        //MPerBlock NPerBlock KPerBlock MPerXDL NPerXDL KPerXDL MWarp NWarp CShuffleNXdlPerWavePerShuffle PipelineScheduler PipelineVer ClusterSizeM ClusterSizeN Occupancy
#define GEMM_CK_TILE_INSTANCE(GemmClass, Scheduler, Version, ClusterSizeM, ClusterSizeN, Occupancy)  \
        GemmClass<256,   256,  128 / DataSize,  16,   16,  KPerXDL,  2,   4,   2, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<128,   256,  128 / DataSize,  16,   16,  KPerXDL,  2,   4,   2, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<128,   128,  128 / DataSize,  16,   16,  KPerXDL,  2,   4,   2, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<128,   128,  256 / DataSize,  16,   16,  KPerXDL,  2,   4,   2, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<256,   256,  256 / DataSize,  16,   16,  KPerXDL,  2,   2,   4, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<256,   256,  128 / DataSize,  16,   16,  KPerXDL,  2,   2,   4, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<128,   128,  256 / DataSize,  16,   16,  KPerXDL,  2,   2,   4, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<128,   256,  128 / DataSize,  16,   16,  KPerXDL,  2,   2,   4, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<128,   128,  128 / DataSize,  16,   16,  KPerXDL,  2,   2,   4, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>, \
        GemmClass<64,    128,  128 / DataSize,  16,   16,  KPerXDL,  2,   2,   2, Scheduler,       Version, ClusterSizeM, ClusterSizeN, Occupancy>

// NOTE: please increase NUM_SHARDS in cmake once you change the instance number.
using gemm_rcr_instances = std::tuple<  
    GEMM_RCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, 1),            // 0
    GEMM_RCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v2, 1),            // 10
    GEMM_RCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 1),            // 20
    GEMM_RCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 2),            // 30
    GEMM_RCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v5, 1),            // 40
    GEMM_RCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, 1),            // 50
    GEMM_RCR_INSTANCE(GemmAsync,       ck::LoopScheduler::Default,                ck::PipelineVersion::v4,          1),            // 60
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::BASIC_V1,       1, 1, 1), // 70
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::MEMORY,         1, 1, 1), // 80
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_V3,     1, 1, 1), // 90
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_V4,     1, 1, 1), // 100
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_ASYNC_V2,  1, 1, 1), // 110
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V1, 1, 1, 1), // 120
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V1, 2, 2, 1), // 130
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V2, 1, 1, 1), // 140
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V2, 2, 2, 1)  // 150
    >;

using gemm_rrr_instances = std::tuple<
    GEMM_RRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, 1),
    GEMM_RRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v2, 1),
    GEMM_RRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 1),
    GEMM_RRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 2),
    GEMM_RRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v5, 1),
    GEMM_RRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, 1),
    GEMM_RRR_INSTANCE(GemmAsync,       ck::LoopScheduler::Default,                ck::PipelineVersion::v4,          1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::BASIC_V1,       1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::MEMORY,         1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_V3,     1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_V4,     1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_ASYNC_V2,  1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V1, 1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V1, 2, 2, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V2, 1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V2, 2, 2, 1)
    >;

using gemm_crr_instances = std::tuple<
    GEMM_CRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, 1),
    GEMM_CRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v2, 1),
    GEMM_CRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 1),
    GEMM_CRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 2),
    GEMM_CRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v5, 1),
    GEMM_CRR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, 1),
    GEMM_CRR_INSTANCE(GemmAsync,       ck::LoopScheduler::Default,                ck::PipelineVersion::v4,          1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::BASIC_V1,       1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::MEMORY,         1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_V3,     1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_V4,     1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_ASYNC_V2,  1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V1, 1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V1, 2, 2, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V2, 1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V2, 2, 2, 1) 
    >;

using gemm_ccr_instances = std::tuple<
    GEMM_CCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, 1),
    GEMM_CCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v2, 1),
    GEMM_CCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 1),
    GEMM_CCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 2),
    GEMM_CCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v5, 1),
    GEMM_CCR_INSTANCE(GemmV3,          ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, 1),
    GEMM_CCR_INSTANCE(GemmAsync,       ck::LoopScheduler::Default,                ck::PipelineVersion::v4,          1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::BASIC_V1,       1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::MEMORY,         1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_V3,     1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_V4,     1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_ASYNC_V2,  1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V1, 1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V1, 2, 2, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V2, 1, 1, 1),
    GEMM_CK_TILE_INSTANCE(GemmCkTile,  ck_tile::GemmPipelineScheduler::Intrawave, ck_tile::GemmPipeline::COMPUTE_TDM_V2, 2, 2, 1)
    >;
// clang-format on

using DeviceOp = ck::tensor_operation::device::DeviceGemmV2<ALayout,
                                                            BLayout,
                                                            CLayout,
                                                            ADataType,
                                                            BDataType,
                                                            CDataType,
                                                            AElementOp,
                                                            BElementOp,
                                                            CElementOp>;

using gemm_xdl_benchmark_instances = std::vector<std::unique_ptr<DeviceOp>>;
