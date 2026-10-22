// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

enum struct GemmPipelineType
{
    Mem,
    CompV3,
    CompV4,
    CompV6,
    CompAsync,
    CompAsyncV2,
    CompAsyncEightWaves,
    CompTDMV1,
    CompTDMV2,
    CompTDMProducerConsumer
};

// Pipelines that load A and B with TDM: no padding, no split-K (TdmEpilogue), ragged extents.
constexpr bool is_tdm_pipeline(GemmPipelineType pt)
{
    return pt == GemmPipelineType::CompTDMV1 || pt == GemmPipelineType::CompTDMV2 ||
           pt == GemmPipelineType::CompTDMProducerConsumer;
}

// Tail coverage loop shared by the async pipeline tests: calls run(M, N, K) for every {M, N}
// in MNs and every num_loop in num_loops, with K = num_loop * k_tile - k_short. A non-zero
// k_short makes K end inside the last K tile without changing num_loop = ceil(K / k_tile).
template <typename RunFn>
void run_tail_coverage(int k_tile,
                       const std::vector<std::pair<int, int>>& MNs,
                       int k_short,
                       RunFn&& run,
                       const std::vector<int>& num_loops = {1, 2, 3, 4, 5, 7, 32})
{
    for(const auto& [M, N] : MNs)
    {
        for(int nl : num_loops)
        {
            SCOPED_TRACE(::testing::Message() << "M=" << M << ", N=" << N << ", num_loop=" << nl
                                              << ", K=" << nl * k_tile - k_short);
            run(M, N, nl * k_tile - k_short);
        }
    }
}

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename PrecType,
          ck_tile::index_t M_Warp_Tile,
          GemmPipelineType PipelineType = GemmPipelineType::CompV3,
          bool IsAsync                  = false>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if CK_TILE_USE_WMMA
#if defined(CK_USE_GFX1250)
    constexpr bool is_8bit = std::is_same_v<PrecType, ck_tile::fp8_t> ||
                             std::is_same_v<PrecType, ck_tile::bf8_t> ||
                             std::is_same_v<PrecType, ck_tile::int8_t>;
    constexpr bool is_highprec =
        std::is_same_v<PrecType, ck_tile::fp32_t> || std::is_same_v<PrecType, ck_tile::fp64_t>;
    constexpr bool is_mxtype =
        std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::pk_fp4_t>;
    if constexpr(M_Warp_Tile == 32 && is_mxtype) // only mx data type can enter this branch
    {
        return 128;
    }
    else
    {
        return is_highprec ? 4 : (is_8bit ? 64 : 32);
    }
#else
    return 16;
#endif
#else
    if constexpr(PipelineType == GemmPipelineType::CompAsyncEightWaves || IsAsync)
        if constexpr(std::is_same_v<PrecType, ck_tile::int8_t>)
            return 64;
        else if constexpr(ck_tile::is_any_of<PrecType,
                                             ck_tile::fp8_t,
                                             ck_tile::pk_fp4_t,
                                             ck_tile::bf8_t>::value)
            return 128;
        else if constexpr(ck_tile::is_any_of<PrecType, ck_tile::fp16_t, ck_tile::bf16_t>::value)
            return 32;
        else
            static_assert(false, "CompAsyncEightWaves / Async pipeline: unsupported type");
    // CompAsyncConfig16x16x128
    else if constexpr(PipelineType == GemmPipelineType::CompAsync && M_Warp_Tile == 16)
        return 128;
    else if constexpr(M_Warp_Tile == 32)
        return 16;
    else
        return 32;
#endif
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <GemmPipelineType PT, typename Problem>
struct GemmPipelineTypeSelector;

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::Mem, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrMem<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrMem<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrMem"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV3, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompV3"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV4, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompV4<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompV4"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV6, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV6<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompV6<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompV6"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompAsync, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompAsync<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompAsync<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompAsync"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompAsyncV2, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompAsyncV2<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompAsyncV2<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompAsyncV2"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompAsyncEightWaves, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompAsyncEightWaves<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompAsyncEightWaves"; }
};
template <typename T>
using has_async = decltype(T::Async);

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompTDMV1, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMV1<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV1"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompTDMV2, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV2"; }
};

template <typename Problem>
struct GemmPipelineTypeSelector<GemmPipelineType::CompTDMProducerConsumer, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMProducerConsumer<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMProducerConsumer"; }
};

template <GemmPipelineType PT, typename Problem, typename Enable = void>
struct GemmEpilogueTypeSelector
{
    using epilogue = ck_tile::CShuffleEpilogue<Problem>;
};

template <GemmPipelineType PT, typename Problem>
struct GemmEpilogueTypeSelector<PT, Problem, std::enable_if_t<is_tdm_pipeline(PT)>>
{
    using epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <GemmPipelineType PT, typename Enable = void>
struct PipelineDefaultParams
{
    static constexpr bool PadM       = true;
    static constexpr bool PadN       = true;
    static constexpr bool PadK       = true;
    static constexpr bool Preshuffle = false;
};

template <GemmPipelineType PT>
struct PipelineDefaultParams<PT, std::enable_if_t<is_tdm_pipeline(PT)>>
{
    static constexpr bool PadM       = false;
    static constexpr bool PadN       = false;
    static constexpr bool PadK       = false;
    static constexpr bool Preshuffle = false;
};

template <typename Tuple, typename Derived>
class TestCkTileGemmPipeline : public ::testing::Test
{
    public:
    static constexpr bool IsAsync_v = [] {
        if constexpr(ck_tile::is_detected<has_async, Derived>{})
        {
            return Derived::Async;
        }
        else
        {
            return false;
        }
    }();

    using ALayout                      = std::tuple_element_t<0, Tuple>;
    using BLayout                      = std::tuple_element_t<1, Tuple>;
    using CLayout                      = std::tuple_element_t<2, Tuple>;
    using ADataType                    = std::tuple_element_t<3, Tuple>;
    using BDataType                    = std::tuple_element_t<4, Tuple>;
    using AccDataType                  = std::tuple_element_t<5, Tuple>;
    using CDataType                    = std::tuple_element_t<6, Tuple>;
    static constexpr auto Scheduler    = std::tuple_element_t<12, Tuple>::value;
    static constexpr auto PipelineType = std::tuple_element_t<13, Tuple>::value;
    static constexpr bool IsTdm        = is_tdm_pipeline(PipelineType);

    static constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, Tuple>{};
    static constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, Tuple>{};
    static constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, Tuple>{};

    static constexpr ck_tile::index_t M_Warp_Tile = std::tuple_element_t<10, Tuple>{};
    static constexpr ck_tile::index_t N_Warp_Tile = std::tuple_element_t<11, Tuple>{};
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::max(get_k_warp_tile<ADataType, M_Warp_Tile, PipelineType, IsAsync_v>(),
                     get_k_warp_tile<BDataType, N_Warp_Tile, PipelineType, IsAsync_v>());

    using AComputeDataType = ADataType;
    using BComputeDataType =
        std::conditional_t<std::is_same_v<BDataType, ck_tile::pk_int4_t>, ADataType, BDataType>;

    using DsLayout   = ck_tile::tuple<>;
    using DsDataType = ck_tile::tuple<>;

    static constexpr bool Persistent =
        ck_tile::tuple_element_or_default_t<Tuple, 14, std::false_type>::value;

    static constexpr bool ClusterLaunch =
        ck_tile::tuple_element_or_default_t<Tuple, 15, std::false_type>::value;

    // Waves per block as ck_tile::sequence<M_Warp, N_Warp, K_Warp>
    using BlockWarps = ck_tile::tuple_element_or_default_t<
        Tuple,
        16,
        ck_tile::sequence<PipelineType == GemmPipelineType::CompAsyncEightWaves ? 4 : 2, 2, 1>>;

    protected:
    // The kernel type the fixture instantiates for a given pad/preshuffle combination.
    // Factored out of invoke_gemm so host-side argument-validation tests can reach
    // Kernel::IsSupportedArgument without allocating device memory or launching.
    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    struct KernelFor
    {
        // CompTDMProducerConsumer stays at its BlockWarps default of [2, 2, 1] (4 consumer +
        // 2 producer waves).
        //
        // [4, 4, 1] was tried as the default and REVERTED: it passes SingleTile (64x64x32 and
        // 128x128x64) on hardware and the host ring replay is sound at ring_spec<3, 2, 16> for
        // every step count and interleaving, but LargeMatrix (2048x2048x2048) HANGS -- hard
        // enough that killing the test left a queue the driver could not evict, which wedged
        // the MES on a gpu_recovery=0 box. The host replay cannot see this because it models
        // the barrier handshake, not the memory the handshake is supposed to order.
        //
        // So the 16-consumer-wave path is NOT merely untuned, it is unsafe at large shapes, and
        // the cause is unknown. Do not restore it as the default, and do not add a pinned
        // [4, 4, 1] BlockWarps type-list entry, until the large-shape hang is root-caused -- a
        // gtest entry that hangs takes the whole device with it. The hang was observed while
        // named_barrier.hpp was missing its release/acquire fences; those are restored now, so
        // re-testing [4, 4, 1] against the fenced build is the first step in root-causing it.
        static constexpr ck_tile::index_t M_Warp = BlockWarps::at(ck_tile::number<0>{});
        static constexpr ck_tile::index_t N_Warp = BlockWarps::at(ck_tile::number<1>{});
        static constexpr ck_tile::index_t K_Warp = BlockWarps::at(ck_tile::number<2>{});

        static constexpr ck_tile::index_t kClusterSizeM = ClusterLaunch ? 2 : 1;
        static constexpr ck_tile::index_t kClusterSizeN = ClusterLaunch ? 2 : 1;
        static constexpr ck_tile::index_t kClusterSizeK = 1;

        // Must match what the tile engine emits (gemm_instance_builder.py,
        // DOUBLE_SMEM_BUFFER_PIPELINES), or the gtest exercises a trait
        // combination production never ships. CompTDMProducerConsumer does not
        // read Problem::DoubleSmemBuffer -- it sizes LDS from its own NumSlots
        // ring -- but CompTDMV1 and CompV4 both static_assert on it, so the two
        // sides are kept in step deliberately rather than by accident.
        //
        // TransposeC below is the one deliberate exception: the tile engine pins
        // comp_tdm_pc to false (verified ISA-identical on gfx1250, where every warp tile
        // is 16x16 and gemm_universal C is RowMajor) to keep it directly comparable to
        // comp_tdm_v2; the fixture keeps the generic gfx1250 rule.
        static constexpr bool DoubleSmemBuffer =
            (PipelineType == GemmPipelineType::CompV4 ||
             PipelineType == GemmPipelineType::CompAsync ||
             PipelineType == GemmPipelineType::CompAsyncV2 ||
             PipelineType == GemmPipelineType::CompTDMV1 ||
             PipelineType == GemmPipelineType::CompTDMV2 ||
             PipelineType == GemmPipelineType::CompTDMProducerConsumer);

#if defined(CK_USE_GFX1250)
        static constexpr bool TransposeC =
            std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor> &&
            M_Warp_Tile == N_Warp_Tile;
#else
        static constexpr bool TransposeC = false;
#endif
        static constexpr bool StructuredSparsity       = false;
        static constexpr ck_tile::index_t NumWaveGroup = 1;

        static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        static constexpr ck_tile::index_t TileParitionerM01      = 4;
        static constexpr ck_tile::index_t VectorSize             = 16;

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

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<PadM,
                                             PadN,
                                             PadK,
                                             DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             CLayout,
                                             TransposeC,
                                             StructuredSparsity,
                                             Persistent,
                                             NumWaveGroup,
                                             Preshuffle,
                                             VectorSize,
                                             ck_tile::DataCachePrefetchKind::None,
                                             ck_tile::DataCachePrefetchKind::None,
                                             IsAsync_v>;

        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  Scheduler,
                                                  ck_tile::element_wise::PassThrough,
                                                  ck_tile::element_wise::PassThrough,
                                                  AComputeDataType,
                                                  BComputeDataType>;

        using GemmPipeline =
            typename GemmPipelineTypeSelector<PipelineType, UniversalGemmProblem>::pipeline;

        using GemmEpilogue = typename GemmEpilogueTypeSelector<
            PipelineType,
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
                                             1,                /*kNumWaveGroups_*/
                                             false,            /*FixedVectorSize_*/
                                             1,                /*VectorSizeC_*/
                                             1,                /*BlockedXDLN_PerWarp_*/
                                             DoubleSmemBuffer, /*DoubleSmemBuffer*/
                                             AComputeDataType, /*AComputeDataType_*/
                                             BComputeDataType /*BComputeDataType_*/>>::epilogue;

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    };

    public:
    // Host-side argument validation without any allocation or launch. The pointers are
    // never dereferenced: IsSupportedArgument only inspects extents and strides. This is
    // the only way to exercise rejection paths whose trigger (a >2GB byte extent) cannot
    // be allocated in a test.
    template <bool PadM       = PipelineDefaultParams<PipelineType>::PadM,
              bool PadN       = PipelineDefaultParams<PipelineType>::PadN,
              bool PadK       = PipelineDefaultParams<PipelineType>::PadK,
              bool Preshuffle = PipelineDefaultParams<PipelineType>::Preshuffle>
    static bool IsArgSupported(const ck_tile::index_t M,
                               const ck_tile::index_t N,
                               const ck_tile::index_t K,
                               const ck_tile::index_t stride_A,
                               const ck_tile::index_t stride_B,
                               const ck_tile::index_t stride_C,
                               const ck_tile::index_t kbatch = 1)
    {
        using Kernel = typename KernelFor<PadM, PadN, PadK, Preshuffle>::Kernel;

        ck_tile::GemmHostArgs args = {
            nullptr, nullptr, nullptr, kbatch, M, N, K, stride_A, stride_B, stride_C};

        return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
    }

    protected:
    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    void invoke_gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
    {
        constexpr int kBlockPerCu = 1;

        // ===============================================

        using Kernel = typename KernelFor<PadM, PadN, PadK, Preshuffle>::Kernel;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 blocks = Kernel::BlockSize();
        dim3 grids;
        if constexpr(Persistent)
        {
            grids = Kernel::MaxOccupancyGridSize(s);
            // At most two workgroups, so multi-tile cases loop and reuse LDS and barriers.
            if constexpr(IsTdm)
            {
                grids.x = std::min(grids.x, 2u);
            }
        }
        else
        {
            grids = Kernel::GridSize(args.M, args.N, args.k_batch);
        }

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:" << " grid: {" << grids.x << ", " << grids.y
                      << ", " << grids.z << "}" << ", blocks: {" << blocks.x << ", " << blocks.y
                      << ", " << blocks.z << "}" << std::endl;
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
            GTEST_SKIP() << "Unsupported data type combination for gemm pipeline test.";
        }
        if constexpr(PipelineType == GemmPipelineType::CompTDMProducerConsumer)
        {
            if(!ck_tile::is_gfx125_supported())
            {
                // A skip reports as PASS; CK_TILE_REQUIRE_GFX125=1 turns it into a failure on
                // runners that are meant to be gfx1250.
                const char* required = std::getenv("CK_TILE_REQUIRE_GFX125");
                if(required != nullptr && required[0] == '1')
                {
                    FAIL() << "CK_TILE_REQUIRE_GFX125=1 but the device reports '"
                           << ck_tile::get_device_name() << "'";
                }
                GTEST_SKIP() << "hardware named barriers require gfx1250; device reports '"
                             << ck_tile::get_device_name() << "'";
            }
        }
        // TDM pipelines use cluster launch (multicast), not supported on gfx1250 A0 (revision 0)
        if constexpr(PipelineType == GemmPipelineType::CompTDMV1 ||
                     PipelineType == GemmPipelineType::CompTDMV2)
        {
            if(ck_tile::get_device_revision() == 0)
            {
                GTEST_SKIP() << "TDM pipeline cluster launch is not supported on gfx1250 "
                                "asicRevision=0";
            }
        }
        // pk_fp4_t with M_Warp_Tile=32 uses dedicated WMMA F4 instructions that are
        // not supported on gfx1250 asicRevision=0. M_Warp_Tile=16 uses the generic
        // F8F6F4 path which is not affected.
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_fp4_t> &&
                     std::is_same_v<BDataType, ck_tile::pk_fp4_t> && M_Warp_Tile == 32)
        {
            if(ck_tile::is_gfx125_supported() && ck_tile::get_device_revision() == 0)
            {
                GTEST_SKIP() << "Unsupported data type combination for gemm pipeline test on "
                                "gfx1250 asicRevision=0.";
            }
        }
        // for TDM it used tdm_epilogue which don't support split-k
        if constexpr(PipelineType == GemmPipelineType::CompV4 ||
                     PipelineType == GemmPipelineType::CompAsyncV2 ||
                     PipelineType == GemmPipelineType::CompAsyncEightWaves || IsAsync_v || IsTdm ||
                     std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Only do k_batch = 1 when pipeline is CompV4, BDataType is I4 or async pipeline.
            // CompAsyncV2 pins k_batch = 1 so the tests control num_loop (tail) exactly.
            k_batches_ = {1};
        }
        else
        {
            // Otherwise, use k_batch = 1 and 2
            k_batches_ = {1, 2};
        }
    }

    template <bool PadM       = PipelineDefaultParams<PipelineType>::PadM,
              bool PadN       = PipelineDefaultParams<PipelineType>::PadN,
              bool PadK       = PipelineDefaultParams<PipelineType>::PadK,
              bool Preshuffle = PipelineDefaultParams<PipelineType>::Preshuffle>
    void Run(const int M,
             const int N,
             const int K,
             const int StrideA = 0,
             const int StrideB = 0,
             const int StrideC = 0)
    {
        // Some unsupported tests don't compile, so we check here before attempting to.
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
        ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, StrideA, is_row_major(ALayout{}));
        ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, StrideB, is_row_major(BLayout{}));
        ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, StrideC, is_row_major(CLayout{}));

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, 11940}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
            permute_vectors_i4x4_b(b_k_n_dev);
            b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
        }
        else
        {
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::GemmHostArgs args = {a_m_k_dev_buf.GetDeviceBuffer(),
                                      b_k_n_dev_buf.GetDeviceBuffer(),
                                      c_m_n_dev_buf.GetDeviceBuffer(),
                                      kbatch,
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      stride_C};

        invoke_gemm<PadM, PadN, PadK, Preshuffle>(args, ck_tile::stream_config{nullptr, false});

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool pass = true;

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));
        EXPECT_TRUE(pass);
    }
};
