// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck/library/utility/gpu_verification.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if CK_TILE_USE_WMMA
#if defined(CK_USE_GFX1250)
    // is_8bit: all 8-bit types (including non-MX int8). is_mxtype: types with MX scale support.
    constexpr bool is_8bit = std::is_same_v<PrecType, ck_tile::fp8_t> ||
                             std::is_same_v<PrecType, ck_tile::bf8_t> ||
                             std::is_same_v<PrecType, ck_tile::int8_t>;
    constexpr bool is_mxtype = std::is_same_v<PrecType, ck_tile::fp8_t> ||
                               std::is_same_v<PrecType, ck_tile::bf8_t> ||
                               std::is_same_v<PrecType, ck_tile::pk_fp4_t>;
    if constexpr(is_mxtype && (M_Warp_Tile == 32 || M_Warp_Tile == 16))
    {
        return 128;
    }
    else
    {
        return is_8bit ? 64 : 32;
    }
#else
    return 16;
#endif
#else
    if constexpr(M_Warp_Tile == 32)
        return 64;
    else
        return 128;
#endif
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

// Deterministic per-element hash RNG for GPU data init. Returns a float in [-3, 3).
// The generic `fill_tensor_uniform_rand_fp_values` filler is NOT valid for ck_tile::pk_fp4_t
// (it converts a single float and duplicates it into both nibbles, and special-cases only the
// classic ck::f4x2_pk_t). We need two independent fp4 values per byte, so we fill directly.
// The narrow [-3,3) range keeps the fp16 GEMM output from overflowing at K up to 4096 (with the
// [0.25,1.0] scales used in RunAllGpu, worst case K*9 = 36864 < 65504).
__device__ inline float mx_fp4_fill_rand(unsigned int seed, unsigned long long idx)
{
    // splitmix64-style avalanche; deterministic given (seed, idx).
    unsigned long long z = (idx + 1ULL) * 0x9E3779B97F4A7C15ULL +
                           static_cast<unsigned long long>(seed) * 0xD1B54A32D192ED03ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const float u =
        static_cast<float>((z >> 40) & 0xFFFFFFULL) / static_cast<float>(0x1000000); // [0,1)
    return u * 6.0f - 3.0f;                                                          // [-3,3)
}

// Fill a packed-fp4 buffer with two independent, deterministic random fp4 values per byte.
// `num_packed` is the number of pk_fp4_t elements (= total fp4 values / 2).
__global__ void
fill_pk_fp4_uniform_kernel(ck_tile::pk_fp4_t* __restrict__ ptr, long num_packed, unsigned int seed)
{
    const long idx0 = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long nthr = static_cast<long>(gridDim.x) * blockDim.x;
    for(long i = idx0; i < num_packed; i += nthr)
    {
        const float lo_f = rintf(mx_fp4_fill_rand(seed, static_cast<unsigned long long>(i) * 2ULL));
        const float hi_f =
            rintf(mx_fp4_fill_rand(seed, static_cast<unsigned long long>(i) * 2ULL + 1ULL));
        const auto lo = ck_tile::float_to_mxfp4(lo_f, 1.0f);
        const auto hi = ck_tile::float_to_mxfp4(hi_f, 1.0f);
        ptr[i]        = ck_tile::pk_fp4_t::_pack(lo, hi);
    }
}

inline void fill_pk_fp4_uniform(ck_tile::pk_fp4_t* ptr,
                                long num_packed,
                                unsigned int seed,
                                hipStream_t stream = nullptr)
{
    constexpr int threads     = 256;
    constexpr long max_blocks = 65536; // grid-stride cap
    const long needed         = (num_packed + threads - 1) / threads;
    const long blocks         = needed < max_blocks ? needed : max_blocks;
    fill_pk_fp4_uniform_kernel<<<dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, stream>>>(
        ptr, num_packed, seed);
    ck_tile::hip_check_error(hipGetLastError());
}

enum struct MxGemmPipelineType
{
    CompTDMV1,
    CompTDMV2,
    CompAsync,
    CompEightWaves,
    WeightPreshuffle
};

template <MxGemmPipelineType PT, typename Problem>
struct MxGemmPipelineTypeSelector;

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV1, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMV1<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV1"; }
};

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV2, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV2"; }
};

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompAsync, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompAsync<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompAsync<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompAsync"; }
};

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompEightWaves, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompAsyncEightWaves<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompEightWaves"; }
};

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::WeightPreshuffle, Problem>
{
    using base_pipeline = ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<Problem>;
    using pipeline      = ck_tile::MXGemmPreshufflePipelineAGmemBGmemCRegV1<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrWeightPreshuffle"; }
};

template <MxGemmPipelineType PT, typename Problem, bool PermuteN>
struct MxGemmEpilogueTypeSelector
{
};

template <typename Problem>
struct MxGemmEpilogueTypeSelector<MxGemmPipelineType::CompTDMV1, Problem, false>
{
    using epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <typename Problem>
struct MxGemmEpilogueTypeSelector<MxGemmPipelineType::CompTDMV2, Problem, false>
{
    using epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <typename Problem>
struct MxGemmEpilogueTypeSelector<MxGemmPipelineType::CompAsync, Problem, false>
{
    using epilogue = ck_tile::CShuffleEpilogue<Problem>;
};

template <typename Problem>
struct MxGemmEpilogueTypeSelector<MxGemmPipelineType::CompEightWaves, Problem, false>
{
    using epilogue = ck_tile::CShuffleEpilogue<Problem>;
};

template <typename Problem, bool PermuteN>
struct MxGemmEpilogueTypeSelector<MxGemmPipelineType::WeightPreshuffle, Problem, PermuteN>
{
    using epilogue = std::conditional_t<PermuteN,
                                        ck_tile::PermuteNEpilogue<Problem>,
                                        ck_tile::CShuffleEpilogue<Problem>>;
};

template <MxGemmPipelineType PT>
struct MxGemmPipelineDefaultParams
{
    static constexpr bool PadM       = false;
    static constexpr bool PadN       = false;
    static constexpr bool PadK       = false;
    static constexpr bool Preshuffle = PT == MxGemmPipelineType::WeightPreshuffle;
};

template <ck_tile::index_t N_Warp_Tile_,
          ck_tile::index_t K_Warp_Tile_,
          ck_tile::index_t N_Tile_,
          ck_tile::index_t N_Warp_,
          typename BDataType_>
struct Config
{
    static constexpr ck_tile::index_t N_Warp_Tile = N_Warp_Tile_;
    static constexpr ck_tile::index_t K_Warp_Tile = K_Warp_Tile_;
    static constexpr ck_tile::index_t N_Tile      = N_Tile_;
    static constexpr ck_tile::index_t N_Warp      = N_Warp_;
    static constexpr ck_tile::index_t BContiguousItemsPerAccess =
        std::is_same_v<BDataType_, ck_tile::pk_fp4_t> ? 32 : 16;
};

template <typename Tuple, typename Derived>
class TestCkTileMxGemmPipeline : public ::testing::Test
{
    public:
    using ALayout                      = std::tuple_element_t<0, Tuple>;
    using BLayout                      = std::tuple_element_t<1, Tuple>;
    using CLayout                      = std::tuple_element_t<2, Tuple>;
    using ADataType                    = std::tuple_element_t<3, Tuple>;
    using BDataType                    = std::tuple_element_t<4, Tuple>;
    using AScaleDataType               = std::tuple_element_t<5, Tuple>;
    using BScaleDataType               = std::tuple_element_t<6, Tuple>;
    using AccDataType                  = std::tuple_element_t<7, Tuple>;
    using CDataType                    = std::tuple_element_t<8, Tuple>;
    static constexpr auto Scheduler    = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr auto PipelineType = std::tuple_element_t<14, Tuple>::value;
    static constexpr bool PermuteN =
        ck_tile::tuple_element_or_default_t<Tuple, 16, std::false_type>::value;

    static constexpr ck_tile::index_t M_Tile = std::tuple_element_t<9, Tuple>{};
    static constexpr ck_tile::index_t N_Tile = std::tuple_element_t<10, Tuple>{};
    static constexpr ck_tile::index_t K_Tile = std::tuple_element_t<11, Tuple>{};

    static constexpr ck_tile::index_t M_Warp_Tile = std::tuple_element_t<12, Tuple>{};
    static constexpr ck_tile::index_t N_Warp_Tile = std::tuple_element_t<13, Tuple>{};
    static constexpr ck_tile::index_t K_Warp_Tile = ck_tile::max(
        get_k_warp_tile<ADataType, M_Warp_Tile>(), get_k_warp_tile<BDataType, N_Warp_Tile>());

    using AComputeDataType = ADataType;
    using BComputeDataType = BDataType;

    using DsLayout   = ck_tile::tuple<>;
    using DsDataType = ck_tile::tuple<>;

    static constexpr bool Persistent = false;
    static constexpr bool ClusterLaunch =
        ck_tile::tuple_element_or_default_t<Tuple, 17, std::false_type>::value;

    static constexpr ck_tile::index_t ScaleBlockSize = std::tuple_element_t<15, Tuple>{};

    static constexpr ck_tile::index_t M_Warp =
        PipelineType == MxGemmPipelineType::WeightPreshuffle
            ? 1
            : (PipelineType == MxGemmPipelineType::CompEightWaves ? 4 : 2);
    static constexpr ck_tile::index_t N_Warp =
        PipelineType == MxGemmPipelineType::WeightPreshuffle ? 4 : 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    protected:
    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    void invoke_mx_gemm(const ck_tile::MxGemmHostArgs<1, 1, 0>& args,
                        const ck_tile::stream_config& s)
    {
        // if cluster launch is enabled, set cluster dim to 2x2x1
        constexpr ck_tile::index_t kClusterSizeM =
            std::conditional_t<ClusterLaunch, ck_tile::number<2>, ck_tile::number<1>>{};
        constexpr ck_tile::index_t kClusterSizeN =
            std::conditional_t<ClusterLaunch, ck_tile::number<2>, ck_tile::number<1>>{};
        constexpr ck_tile::index_t kClusterSizeK =
            std::conditional_t<ClusterLaunch, ck_tile::number<1>, ck_tile::number<1>>{};

        constexpr bool kPadM      = PadM;
        constexpr bool kPadN      = PadN;
        constexpr bool kPadK      = PadK;
        constexpr bool preshuffle = Preshuffle;

        constexpr bool DoubleSmemBuffer = true; // TDM pipeline requires double smem buffer

#if defined(CK_USE_GFX1250)
        constexpr ck_tile::index_t BlockedXDLNPerWarp = 1;
        constexpr bool TransposeC =
            std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor> &&
            M_Warp_Tile == N_Warp_Tile;
#elif defined(CK_USE_GFX950)
        constexpr ck_tile::index_t BlockedXDLNPerWarp = Preshuffle ? 2 : 1;
        constexpr bool TransposeC                     = false;
#endif
        static constexpr bool StructuredSparsity = false;
        static constexpr bool NumWaveGroup       = 1;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape = std::conditional_t<
            ClusterLaunch,
            ck_tile::ClusterTileGemmShape<
                ck_tile::sequence<kClusterSizeM, kClusterSizeN, kClusterSizeK>,
                ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>,
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>>;

        using TilePartitioner =
            std::conditional_t<ClusterLaunch,
                               ck_tile::GemmClusterTilePartitioner<GemmShape>,
                               ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                                          TileParitionerGroupNum,
                                                                          TileParitionerM01>>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC,
                                                                     StructuredSparsity,
                                                                     Persistent,
                                                                     NumWaveGroup,
                                                                     preshuffle>;

        using UniversalGemmProblem =
            ck_tile::MxGemmPipelineProblem<ADataType,
                                           BDataType,
                                           AccDataType,
                                           GemmShape,
                                           GemmUniversalTraits,
                                           Scheduler,
                                           ck_tile::element_wise::PassThrough,
                                           ck_tile::element_wise::PassThrough,
                                           AComputeDataType,
                                           BComputeDataType,
                                           AScaleDataType,
                                           BScaleDataType,
                                           ScaleBlockSize>;

        using GemmPipeline =
            typename MxGemmPipelineTypeSelector<PipelineType, UniversalGemmProblem>::pipeline;

        using GemmEpilogueProblem = std::conditional_t<
            PipelineType == MxGemmPipelineType::WeightPreshuffle && PermuteN,
            ck_tile::PermuteNEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             false, /*FixedVectorSize_*/
                                             1>,    /*VectorSizeC_*/
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             1,                  /*kNumWaveGroups_*/
                                             false,              /*FixedVectorSize_*/
                                             1,                  /*VectorSizeC_*/
                                             BlockedXDLNPerWarp, /*BlockedXDLN_PerWarp_*/
                                             DoubleSmemBuffer,   /*DoubleSmemBuffer*/
                                             AComputeDataType,   /*AComputeDataType_*/
                                             BComputeDataType,   /*BComputeDataType_*/
                                             !preshuffle>>;

        using GemmEpilogue = typename MxGemmEpilogueTypeSelector<PipelineType,
                                                                 GemmEpilogueProblem,
                                                                 PermuteN>::epilogue;

        using Kernel = ck_tile::MxGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping mx_gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching MxGemm kernel with args:" << " grid: {" << grids.x << ", "
                      << grids.y << ", " << grids.z << "}" << ", blocks: {" << blocks.x << ", "
                      << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        if constexpr(ClusterLaunch)
        {
            dim3 clusters = Kernel::ClusterSize();
            ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, clusters, grids, blocks, 0, kargs));
        }
        else
        {
            ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
    }

    public:
    std::vector<int> k_batches_;

    void SetUp() override
    {
        if constexpr(!Derived::check_data_type())
        {
            GTEST_SKIP() << "Unsupported data type combination for mx_gemm pipeline test.";
        }
        // for TDM it's used tdm_epilogue which don't support split-k
        if constexpr(PipelineType == MxGemmPipelineType::CompTDMV1 ||
                     PipelineType == MxGemmPipelineType::CompTDMV2 ||
                     std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor> ||
                     std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            // Only do k_batch = 1
            k_batches_ = {1};
        }
        else
        {
            // Otherwise, use k_batch = 1 and 2
            k_batches_ = {1, 2};
        }
    }

    template <bool PadM       = MxGemmPipelineDefaultParams<PipelineType>::PadM,
              bool PadN       = MxGemmPipelineDefaultParams<PipelineType>::PadN,
              bool PadK       = MxGemmPipelineDefaultParams<PipelineType>::PadK,
              bool Preshuffle = MxGemmPipelineDefaultParams<PipelineType>::Preshuffle>
    void Run(const int M,
             const int N,
             const int K,
             const int StrideA = 0,
             const int StrideB = 0,
             const int StrideC = 0)
    {
        if constexpr(Derived::check_data_type())
        {
            for(auto kb : k_batches_)
            {
                // skip test when split k' number is not evenly distributed
                if((K / K_Tile) % kb != 0)
                {
                    continue;
                }
                RunSingle<PadM, PadN, PadK, Preshuffle>(M, N, K, StrideA, StrideB, StrideC, kb);
            }
        }
    }

    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   int kbatch = 1)
    {
        using namespace ck_tile;

        // K must be a multiple of ScaleBlockSize
        if(K % ScaleBlockSize != 0)
        {
            GTEST_SKIP() << "K must be multiple of ScaleBlockSize for MX GEMM";
        }

        index_t stride_A = get_default_stride(M, K, StrideA, is_row_major(ALayout{}));
        index_t stride_B = get_default_stride(K, N, StrideB, is_row_major(BLayout{}));
        index_t stride_C = get_default_stride(M, N, StrideC, is_row_major(CLayout{}));

        // Create host tensors for A, B, C
        HostTensor<ADataType> a_m_k(
            host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        HostTensor<BDataType> b_k_n(
            host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        HostTensor<CDataType> c_m_n_dev_result(
            host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

        // Create host tensors for scale_a and scale_b
        // scale_a: (M, K/ScaleBlockSize) row-major
        // scale_b: (N, K/ScaleBlockSize) col-major
        const index_t num_scale_k = K / ScaleBlockSize;
        // Pre-shuffle interleaves 2 K-lanes (MNPack=2) with MPerXdlops=16 stride,
        // so M must be padded to at least MNPack * MPerXdlops = 32.
        constexpr index_t ScaleShuffleAlign = 32;
        const index_t scale_padded_M        = integer_least_multiple(
            static_cast<index_t>(M), static_cast<index_t>(ck_tile::max(M_Tile, ScaleShuffleAlign)));

        HostTensor<AScaleDataType> scale_a(
            {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

        const index_t scale_padded_N = integer_least_multiple(
            static_cast<index_t>(N), static_cast<index_t>(ck_tile::max(N_Tile, ScaleShuffleAlign)));
        // Pre-shuffle interleaves 2 K-lanes (MNPack=2) with MPerXdlops=16 stride,
        // so N must be padded to at least MNPack * NPerXdlops = 32.
        HostTensor<BScaleDataType> scale_b(
            {static_cast<std::size_t>(scale_padded_N), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

        // Fill data
        // For pk_fp4_t each byte packs two 4-bit elements; the generic filler
        // converts a single float and duplicates it into both nibbles.
        // Generate two independent random values per byte instead.
        if constexpr(std::is_same_v<ADataType, pk_fp4_t>)
        {
            std::mt19937 gen(11939);
            std::uniform_real_distribution<float> dis(-5.f, 5.f);
            for(auto& elem : a_m_k.mData)
            {
                auto lo = float_to_mxfp4(std::round(dis(gen)), 1.f);
                auto hi = float_to_mxfp4(std::round(dis(gen)), 1.f);
                elem    = pk_fp4_t::_pack(lo, hi);
            }
        }
        else
        {
            FillUniformDistributionIntegerValue<ADataType>{-5, 5, 11939}(a_m_k);
        }
        if constexpr(std::is_same_v<BDataType, pk_fp4_t>)
        {
            std::mt19937 gen(11940);
            std::uniform_real_distribution<float> dis(-5.f, 5.f);
            for(auto& elem : b_k_n.mData)
            {
                auto lo = float_to_mxfp4(std::round(dis(gen)), 1.f);
                auto hi = float_to_mxfp4(std::round(dis(gen)), 1.f);
                elem    = pk_fp4_t::_pack(lo, hi);
            }
        }
        else
        {
            FillUniformDistributionIntegerValue<BDataType>{-5, 5, 11940}(b_k_n);
        }

        {
            // Fill scale tensors with values uniformly drawn from [0.125, 2.0] = [2^-3, 2^1].
            // This spans 5 exponent bands centred around 1.0, keeping scales numerically
            // well-behaved without saturating the accumulator.
            //
            // Per-type raw byte ranges produced (raw bytes sampled uniformly within each):
            //   e8m0_t (bias=127, mant=0): raw in [124, 128] -> floats {0.125, 0.25, 0.5, 1.0, 2.0}
            //   e4m3_t (bias=7,   mant=3): raw in [32,  64]  -> floats  0.125 .. 2.0
            //   e5m3_t (bias=15,  mant=3): raw in [96,  128] -> floats  0.125 .. 2.0
            // No generated value exceeds 2.0 for any type.
            // A and B use different seeds so their scale values are uncorrelated.
            ck_tile::FillUniformScaleDistribution<AScaleDataType>{0.125f, 2.0f, 11941}(scale_a);
            ck_tile::FillUniformScaleDistribution<BScaleDataType>{0.125f, 2.0f, 11943}(scale_b);
        }

        // Pre-shuffle scale buffers for the hardware
#if defined(CK_USE_GFX1250)
        static constexpr index_t NXdlPackEff = 1;

        HostTensor<AScaleDataType> scale_a_shuffled(
            {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

        HostTensor<BScaleDataType> scale_b_shuffled(
            {static_cast<std::size_t>(scale_padded_N), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

        // Pre-shuffle for gfx1250 (WaveSize=32, WMMA)
        // Scales start in natural tensor layout and are pre-shuffled into the device layout
        // for both scale block sizes (the shuffle is the identity for ScaleBlockSize==16,
        // whose natural layout already matches the warp scale distribution).
        ck_tile::preShuffleScaleBuffer_gfx1250<AScaleDataType, ScaleBlockSize, true>(
            scale_a.mData.data(), scale_a_shuffled.mData.data(), scale_padded_M, num_scale_k);
        ck_tile::preShuffleScaleBuffer_gfx1250<BScaleDataType, ScaleBlockSize, true>(
            scale_b.mData.data(), scale_b_shuffled.mData.data(), scale_padded_N, num_scale_k);
#elif defined(CK_USE_GFX950)
        constexpr ck_tile::index_t MPerXdl      = M_Warp_Tile;
        constexpr ck_tile::index_t NPerXdl      = N_Warp_Tile;
        constexpr ck_tile::index_t KPerXdl      = K_Warp_Tile;
        constexpr ck_tile::index_t MIterPerWarp = M_Tile / (M_Warp * MPerXdl);
        constexpr ck_tile::index_t NIterPerWarp = N_Tile / (N_Warp * NPerXdl);
        constexpr ck_tile::index_t KIterPerWarp = K_Tile / KPerXdl;

        constexpr ck_tile::index_t MXdlPackEff =
            (MIterPerWarp >= 2 && MIterPerWarp % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t NXdlPackEff =
            (NIterPerWarp >= 2 && NIterPerWarp % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t KXdlPackEff =
            (KIterPerWarp >= 2 && KIterPerWarp % 2 == 0) ? 2 : 1;

        constexpr ck_tile::index_t XdlMNThread = M_Warp_Tile;
        constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

        HostTensor<AScaleDataType> scale_a_shuffled(
            {static_cast<std::size_t>(scale_padded_M / MXdlPackEff * 2),
             static_cast<std::size_t>(num_scale_k / KXdlPackEff * 2)},
            {static_cast<std::size_t>(num_scale_k / KXdlPackEff * 2), static_cast<std::size_t>(1)});

        HostTensor<BScaleDataType> scale_b_shuffled(
            {static_cast<std::size_t>(scale_padded_N / NXdlPackEff * 2),
             static_cast<std::size_t>(num_scale_k / KXdlPackEff * 2)},
            {static_cast<std::size_t>(num_scale_k / KXdlPackEff * 2), static_cast<std::size_t>(1)});

        ck_tile::preShuffleScaleBuffer_gfx950<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(
            scale_a.mData.data(), scale_a_shuffled.mData.data(), scale_padded_M, num_scale_k, true);

        if constexpr(PipelineType == MxGemmPipelineType::WeightPreshuffle && PermuteN)
        {
            ck_tile::preShuffleScaleBufferPermuteN_gfx950<N_Warp, N_Tile, XdlMNThread>(
                scale_b.mData.data(),
                scale_b_shuffled.mData.data(),
                scale_padded_N,
                num_scale_k,
                true);
        }
        else
        {
            ck_tile::
                preShuffleScaleBuffer_gfx950<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(
                    scale_b.mData.data(),
                    scale_b_shuffled.mData.data(),
                    scale_padded_N,
                    num_scale_k,
                    true);
        }
#endif

        // Allocate device memory
        DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());
        DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
        DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

        // Upload data to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();
        scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
        scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

        using GemmConfig = Config<N_Warp_Tile, K_Warp_Tile, N_Tile, N_Warp, BDataType>;

        const auto b_host_for_dev = [&]() {
            if constexpr(Preshuffle)
            {
                if constexpr(PermuteN)
                {
                    return ck_tile::shuffle_b_permuteN<GemmConfig, BDataType, NXdlPackEff>(b_k_n);
                }
                else
                {
                    return ck_tile::shuffle_b<GemmConfig>(b_k_n);
                }
            }
            else
            {
                return b_k_n;
            }
        }();
        DeviceMem b_k_n_dev_buf(b_host_for_dev.get_element_space_size_in_bytes());
        b_k_n_dev_buf.ToDevice(b_host_for_dev.data());

        // Create MxGemmHostArgs
        ck_tile::MxGemmHostArgs<1, 1, 0> args(
            {static_cast<const void*>(a_m_k_dev_buf.GetDeviceBuffer())},
            {static_cast<const void*>(scale_a_dev_buf.GetDeviceBuffer())},
            {static_cast<const void*>(b_k_n_dev_buf.GetDeviceBuffer())},
            {static_cast<const void*>(scale_b_dev_buf.GetDeviceBuffer())},
            {},
            c_m_n_dev_buf.GetDeviceBuffer(),
            kbatch,
            M,
            N,
            K,
            {stride_A},
            {stride_B},
            {},
            stride_C);

        invoke_mx_gemm<PadM, PadN, PadK, Preshuffle>(args, stream_config{nullptr, false});

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

        // Host reference computation using reference_mx_gemm
        // reference_mx_gemm expects scale_a(M, K/ScaleBlockSize) and scale_b(K/ScaleBlockSize, N)
        // We need to create scale_b in (K/ScaleBlockSize, N) format for the reference
        HostTensor<BScaleDataType> scale_b_ref(
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(N)},
            {static_cast<std::size_t>(1), static_cast<std::size_t>(num_scale_k)});
        // Copy scale_b data (our scale_b is (N, num_scale_k) row-major,
        // reference expects (num_scale_k, N) col-major, which is the same memory layout)
        // Truncate scale_a to actual N (not padded)
        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < num_scale_k; ++k)
            {
                scale_b_ref(k, n) = scale_b(n, k);
            }
        }

        // Truncate scale_a to actual M (not padded)
        HostTensor<AScaleDataType> scale_a_ref(
            {static_cast<std::size_t>(M), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});
        for(int m = 0; m < M; ++m)
        {
            for(int k = 0; k < num_scale_k; ++k)
            {
                scale_a_ref(m, k) = scale_a(m, k);
            }
        }

        HostTensor<CDataType> c_m_n_host_ref(
            host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        reference_mx_gemm<ADataType,
                          BDataType,
                          AScaleDataType,
                          BScaleDataType,
                          AccDataType,
                          CDataType>(a_m_k, b_k_n, c_m_n_host_ref, scale_a_ref, scale_b_ref);

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        bool pass = check_err(c_m_n_dev_result,
                              c_m_n_host_ref,
                              "Error: Incorrect results!",
                              rtol_atol.at(number<0>{}),
                              rtol_atol.at(number<1>{}));
        EXPECT_TRUE(pass);
    }

    // All-GPU validation path for the fp4 (pk_fp4_t) MX GEMM.
    //
    // Unlike Run(), this never materializes the A/B/C tensors on the host:
    //   - A/B are generated directly on device with a deterministic fp4 fill.
    //   - the reference is computed on device by reference_mx_gemm_gpu.
    //   - the comparison is done on device by ck::profiler::gpu_verify.
    // Only the tiny e8m0 scales touch the host (for pre-shuffle + an unshuffled copy that the
    // device reference consumes).
    void RunAllGpu(const int M, const int N, const int K, const int kbatch = 1)
    {
        if constexpr(!Derived::check_data_type())
            return;

        static_assert(std::is_same_v<ADataType, ck_tile::pk_fp4_t> &&
                          std::is_same_v<BDataType, ck_tile::pk_fp4_t>,
                      "RunAllGpu currently supports pk_fp4_t A/B only.");
        // The GPU reference (reference_mx_gemm_gpu) hardcodes these layouts; guard so it cannot be
        // silently misused with a layout it does not handle.
        static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor> &&
                          std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor> &&
                          std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>,
                      "RunAllGpu / reference_mx_gemm_gpu assume RowMajor-A, ColumnMajor-B, "
                      "RowMajor-C.");

        static_assert(PipelineType != MxGemmPipelineType::WeightPreshuffle);

#if !defined(CK_USE_GFX950)
        (void)M;
        (void)N;
        (void)K;
        (void)kbatch;
        GTEST_SKIP() << "RunAllGpu requires CK_USE_GFX950.";
#else
        using namespace ck_tile::literals;
        constexpr long kIntMax = 2147483647L; // INT_MAX

        auto f_get_default_stride =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(stride == 0)
                {
                    if constexpr(std::is_same_v<decltype(layout),
                                                ck_tile::tensor_layout::gemm::RowMajor>)
                        return col;
                    else
                        return row;
                }
                else
                    return stride;
            };

        constexpr ck_tile::index_t psize = ck_tile::numeric_traits<ADataType>::PackedSize; // 2
        static_assert(psize == 2,
                      "RunAllGpu byte-sizing and reference_mx_gemm_kernel's a_ptr[a_lin/2] "
                      "addressing assume pk_fp4_t PackedSize == 2.");

        bool pass     = true;
        long total_MN = 0;

        // Strides are K/N here (small); keep them as index_t to match the kernel args, and
        // make the size_t->index_t narrowing explicit.
        const ck_tile::index_t stride_A =
            static_cast<ck_tile::index_t>(f_get_default_stride(M, K, 0, ALayout{})); // K
        const ck_tile::index_t stride_B =
            static_cast<ck_tile::index_t>(f_get_default_stride(K, N, 0, BLayout{})); // K
        const ck_tile::index_t stride_C =
            static_cast<ck_tile::index_t>(f_get_default_stride(M, N, 0, CLayout{})); // N

        ASSERT_EQ(K % ScaleBlockSize, 0) << "K must be a multiple of ScaleBlockSize for MX GEMM";
        const ck_tile::index_t num_scale_k = K / ScaleBlockSize;
        ASSERT_EQ(num_scale_k % (K_Warp_Tile / ScaleBlockSize), 0)
            << "K must be a multiple of K_Warp_Tile (" << K_Warp_Tile
            << ") for MX GEMM. Pad the scale data.";
        const ck_tile::index_t scale_padded_M = ck_tile::integer_least_multiple(
            static_cast<ck_tile::index_t>(M), static_cast<ck_tile::index_t>(M_Tile));

        // int32-safety: the property under test for the M-decomposition. The predicate is
        // "largest 0-based element offset fits in a signed 32-bit int", i.e. offset <= INT_MAX.
        const long MN      = static_cast<long>(M) * N;
        const long A_elems = static_cast<long>(M) * K;
        const long B_elems = static_cast<long>(K) * N;
        const long C_off   = static_cast<long>(M - 1) * stride_C + (N - 1);
        const long A_off   = static_cast<long>(M - 1) * stride_A + (K - 1);
        const long B_off   = static_cast<long>(N - 1) * stride_B + (K - 1);
        const long c_bytes = MN * static_cast<long>(sizeof(CDataType));
        std::cout << "[int32-safety] M=" << M << " N=" << N << " K=" << K << " M*N=" << MN
                  << " A_elems=" << A_elems << " B_elems=" << B_elems << " C_off=" << C_off
                  << " A_off=" << A_off << " B_off=" << B_off << " C_bytes=" << c_bytes
                  << " (INT_MAX=" << kIntMax << ")" << std::endl;
        // Note (not an assert): the C *byte* span can exceed INT_MAX even when the element
        // count is int32-safe. We deliberately let the run proceed -- if any internal byte
        // offset overflows, gpu_verify will flag it, which is exactly what we want to discover.
        if(c_bytes > kIntMax)
            std::cout << "[int32-safety][note] C byte span (" << c_bytes
                      << ") exceeds INT_MAX; if verification fails, byte-offset overflow is the "
                         "prime suspect."
                      << std::endl;
        ASSERT_LE(B_off, kIntMax) << "B offset exceeds INT_MAX";
        total_MN += MN;

        const long a_bytes = (A_elems + psize - 1) / psize;
        const long b_bytes = (B_elems + psize - 1) / psize;

        // Bound peak device memory (A + B + 2*C + scales). Skip cleanly rather
        // than aborting via hip_check_error if the device cannot hold test shapes.
        {
            std::size_t free_b = 0, total_b = 0;
            ck_tile::hip_check_error(hipMemGetInfo(&free_b, &total_b));
            const std::size_t need = static_cast<std::size_t>(a_bytes) +
                                     static_cast<std::size_t>(b_bytes) +
                                     2u * static_cast<std::size_t>(c_bytes) + (64u << 20);
            if(free_b < need)
                GTEST_SKIP() << "insufficient device memory (need " << need << " B, free " << free_b
                             << " B)";
        }

        auto a_dev     = std::make_unique<ck_tile::DeviceMem>(static_cast<std::size_t>(a_bytes));
        auto b_dev     = std::make_unique<ck_tile::DeviceMem>(static_cast<std::size_t>(b_bytes));
        auto c_dev     = std::make_unique<ck_tile::DeviceMem>(static_cast<std::size_t>(c_bytes));
        auto c_ref_dev = std::make_unique<ck_tile::DeviceMem>(static_cast<std::size_t>(c_bytes));
        c_dev->SetZero();
        c_ref_dev->SetZero();

        // GPU fill A/B (deterministic, fp4-correct). Same device buffers feed both the kernel
        // and the reference, so the fill need not bit-match any host RNG.
        fill_pk_fp4_uniform(
            reinterpret_cast<ADataType*>(a_dev->GetDeviceBuffer()), a_bytes, 11939u);
        fill_pk_fp4_uniform(
            reinterpret_cast<BDataType*>(b_dev->GetDeviceBuffer()), b_bytes, 11940u);
        ck_tile::hip_check_error(hipDeviceSynchronize()); // surface fill faults at the fill site

        // e8m0 scales (tiny, host-built). The range is
        // deliberately narrow ([0.25,1.0] scales, [-3,3) fp4 fill) so that K up to 4096 cannot
        // overflow the fp16 output (worst case K*9 = 36864 < 65504); gpu_verify counts matched
        // infinities as errors, so an overflow would otherwise be a false failure.
        ck_tile::HostTensor<AScaleDataType> scale_a(
            {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});
        ck_tile::HostTensor<BScaleDataType> scale_b(
            {static_cast<std::size_t>(N), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});
        {
            std::mt19937 gen(11941u);
            std::uniform_real_distribution<float> dist(0.25f, 1.0f);
            for(auto& s : scale_a.mData)
                s = AScaleDataType{dist(gen)};
            for(auto& s : scale_b.mData)
                s = BScaleDataType{dist(gen)};
        }

        // gfx950 scale pre-shuffle. NOTE: this must stay in sync with the identical block in
        // Run() -- the kernel-input layout and the reference-input layout must agree.
        constexpr ck_tile::index_t MPerXdl      = M_Warp_Tile;
        constexpr ck_tile::index_t NPerXdl      = N_Warp_Tile;
        constexpr ck_tile::index_t KPerXdl      = K_Warp_Tile;
        constexpr ck_tile::index_t MIterPerWarp = M_Tile / (M_Warp * MPerXdl);
        constexpr ck_tile::index_t NIterPerWarp = N_Tile / (N_Warp * NPerXdl);
        constexpr ck_tile::index_t KIterPerWarp = K_Tile / KPerXdl;

        constexpr ck_tile::index_t MXdlPackEff =
            (MIterPerWarp >= 2 && MIterPerWarp % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t NXdlPackEff =
            (NIterPerWarp >= 2 && NIterPerWarp % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t KXdlPackEff =
            (KIterPerWarp >= 2 && KIterPerWarp % 2 == 0) ? 2 : 1;

        constexpr ck_tile::index_t XdlMNThread = M_Warp_Tile;
        constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

        ck_tile::HostTensor<AScaleDataType> scale_a_shuffled(
            {static_cast<std::size_t>(scale_padded_M / MXdlPackEff * 2),
             static_cast<std::size_t>(num_scale_k / KXdlPackEff * 2)},
            {static_cast<std::size_t>(num_scale_k / KXdlPackEff * 2), static_cast<std::size_t>(1)});
        ck_tile::HostTensor<BScaleDataType> scale_b_shuffled(
            {static_cast<std::size_t>(N / NXdlPackEff * 2),
             static_cast<std::size_t>(num_scale_k / KXdlPackEff * 2)},
            {static_cast<std::size_t>(num_scale_k / KXdlPackEff * 2), static_cast<std::size_t>(1)});

        ck_tile::preShuffleScaleBuffer_gfx950<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(
            scale_a.mData.data(), scale_a_shuffled.mData.data(), scale_padded_M, num_scale_k, true);
        ck_tile::preShuffleScaleBuffer_gfx950<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(
            scale_b.mData.data(), scale_b_shuffled.mData.data(), N, num_scale_k, true);

        // Device scale buffers: shuffled feed the kernel, unshuffled feed the reference.
        auto scale_a_shuf_dev = std::make_unique<ck_tile::DeviceMem>(
            scale_a_shuffled.get_element_space_size_in_bytes());
        auto scale_b_shuf_dev = std::make_unique<ck_tile::DeviceMem>(
            scale_b_shuffled.get_element_space_size_in_bytes());
        scale_a_shuf_dev->ToDevice(scale_a_shuffled.data());
        scale_b_shuf_dev->ToDevice(scale_b_shuffled.data());

        auto scale_a_ref_dev =
            std::make_unique<ck_tile::DeviceMem>(scale_a.get_element_space_size_in_bytes());
        auto scale_b_ref_dev =
            std::make_unique<ck_tile::DeviceMem>(scale_b.get_element_space_size_in_bytes());
        scale_a_ref_dev->ToDevice(scale_a.data());
        scale_b_ref_dev->ToDevice(scale_b.data());

        // Launch kernel
        ck_tile::MxGemmHostArgs<1, 1, 0> args(
            {static_cast<const void*>(a_dev->GetDeviceBuffer())},
            {static_cast<const void*>(scale_a_shuf_dev->GetDeviceBuffer())},
            {static_cast<const void*>(b_dev->GetDeviceBuffer())},
            {static_cast<const void*>(scale_b_shuf_dev->GetDeviceBuffer())},
            {},
            c_dev->GetDeviceBuffer(),
            kbatch,
            M,
            N,
            K,
            {stride_A},
            {stride_B},
            {},
            stride_C);

        invoke_mx_gemm<false, false, false, false>(args, ck_tile::stream_config{nullptr, false});

        ck_tile::hip_check_error(hipDeviceSynchronize());

        // GPU reference on the same device A/B buffers.
        ck_tile::reference_mx_gemm_gpu<ADataType,
                                       BDataType,
                                       AScaleDataType,
                                       BScaleDataType,
                                       AccDataType,
                                       CDataType>(
            reinterpret_cast<const ADataType*>(a_dev->GetDeviceBuffer()),
            reinterpret_cast<const BDataType*>(b_dev->GetDeviceBuffer()),
            reinterpret_cast<const AScaleDataType*>(scale_a_ref_dev->GetDeviceBuffer()),
            reinterpret_cast<const BScaleDataType*>(scale_b_ref_dev->GetDeviceBuffer()),
            reinterpret_cast<CDataType*>(c_ref_dev->GetDeviceBuffer()),
            M,
            N,
            K,
            num_scale_k,
            ScaleBlockSize);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        // GPU verify with explicit MX tolerance (auto tolerance defaults too tight for MX).
        const float max_acc = ck::profiler::gpu_reduce_max<CDataType>(c_ref_dev->GetDeviceBuffer(),
                                                                      static_cast<std::size_t>(MN));
        // The reference must be non-degenerate, else error_count==0 is a vacuous pass.
        ASSERT_GT(max_acc, 0.0f) << "GPU reference output is all-zero";
        const auto rtol_atol =
            calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(K, kbatch, max_acc);
        const auto res = ck::profiler::gpu_verify<CDataType>(c_dev->GetDeviceBuffer(),
                                                             c_ref_dev->GetDeviceBuffer(),
                                                             rtol_atol.at(ck_tile::number<0>{}),
                                                             rtol_atol.at(ck_tile::number<1>{}),
                                                             static_cast<std::size_t>(MN));

        // Positive liveness check on the *device* output. res.all_zero ANDs device- and
        // reference-zeroness, and the reference is never zero here, so it cannot detect a no-op
        // kernel on its own -- reduce the device buffer directly.
        const float c_dev_absmax = ck::profiler::gpu_reduce_max<CDataType>(
            c_dev->GetDeviceBuffer(), static_cast<std::size_t>(MN));

        std::cout << "[verify] errors=" << res.error_count << " max_error=" << res.max_error
                  << " c_dev_absmax=" << c_dev_absmax << " max_acc=" << max_acc
                  << " rtol=" << rtol_atol.at(ck_tile::number<0>{})
                  << " atol=" << rtol_atol.at(ck_tile::number<1>{}) << std::endl;

        EXPECT_EQ(res.error_count, 0ull) << "produced mismatched results";
        EXPECT_GT(c_dev_absmax, 0.0f) << "produced an all-zero device output";
        pass &= (res.error_count == 0 && c_dev_absmax > 0.0f);

        std::cout << "[int32-safety] aggregate total_M*N=" << total_MN << " (INT_MAX=" << kIntMax
                  << ") -> decomposition is the variable under test" << std::endl;
        EXPECT_TRUE(pass);
#endif // CK_USE_GFX950
    }
};
