// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

// 0: the machine scheduler orders each region (a half K step or a K step) with the hot loop
// hints. 1: the source order is the schedule (pinned per-step waits, no hints), built with
// -mllvm -enable-misched=false.
#ifndef CK_TILE_TDM_SCHEDULED_SOURCE_ORDER
#define CK_TILE_TDM_SCHEDULED_SOURCE_ORDER 0
#endif

namespace ck_tile {

namespace mx_tdm_scheduled {

// The GL2 prefetch's cache policy: th:TH_LOAD_NT | scope:SCOPE_SE (the enum has it on gfx12 only).
#if defined(__gfx12__)
inline constexpr auto kGl2PrefetchCoherence = amd_buffer_coherence_enum::SE_NT;
#else
inline constexpr auto kGl2PrefetchCoherence = amd_buffer_coherence_enum::coherence_default;
#endif
} // namespace mx_tdm_scheduled

/// The K loop's tile-count dispatch, on the host as flatmm's: tile t uses LDS buffer t % 2 and
/// the loop runs tile pairs, so a whole-tile count has one tile besides the pairs
/// (TailNumber::One) or none (Two). The caller instantiates the kernel whose Problem::TailNum
/// TailHandler passes; the loop runs any count of pairs, so the hot loop is not dispatched.
template <typename Problem>
struct BaseMxGemmPipelineAgBgCrCompTDMScheduled
{
    static constexpr index_t PrefetchStages = 2;

    /// num_loop counts whole K tiles, the K tail excluded.
    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return num_loop % PrefetchStages == 1 ? TailNumber::One : TailNumber::Two;
    }

    /// The TailNumber of an M x N x K problem: of its whole K tiles, the K tail excluded.
    CK_TILE_HOST_DEVICE static constexpr TailNumber GetTailNumber(index_t K)
    {
        return GetBlockLoopTailNum(K / Problem::BlockGemmShape::kK);
    }

    /// Calls run_func(bool_constant<true>, integral_constant<TailNumber, tail_number>).
    /// With CK_TILE_FORCE_SINGLE_TAIL_HANDLER, one variant, TailNumber::Full, serves every count:
    /// its odd tile and the K tail's buffer are chosen at run time.
    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        ignore = has_hot_loop;
#if !defined(CK_TILE_FORCE_SINGLE_TAIL_HANDLER)
        if(tail_number == TailNumber::One)
            return run_func(bool_constant<true>{},
                            integral_constant<TailNumber, TailNumber::One>{});
        return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Two>{});
#else
        ignore = tail_number;
        return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Full>{});
#endif
    }
};

/**
 * @brief MX GEMM main loop for gfx1250: TDM loads of A, B and their K-major scales into two LDS
 *        buffers, scaled WMMAs scheduled explicitly, optional cluster multicast and GL2 prefetch.
 *
 * PipelineImpl has five roles: TdmLoader (descriptors built once, each tile patches only its LDS
 * address, global address and remaining sizes; even waves load one B component, odd waves one A
 * component), Gl2Prefetcher, LdsReader (16 B reads), TileCompute (PrefetchLdsRead and
 * SplitBLdsRead) and Pipeline (PrefetchStages 2: tile k refills its own buffer with
 * tile k+2 once the cluster has read it; tile k+1 is waited for at tile k's last K step).
 *
 * The schedule is one intrawave loop, so Problem::Scheduler must be
 * GemmPipelineScheduler::Intrawave. The machine scheduler orders each region (a half K step, a K
 * step) with V1-style hot loop hints that spread the next K step's LDS reads over the WMMAs;
 * fences pin the barriers and TDM loads, whose descriptors are computed a region ahead.
 * CK_TILE_TDM_SCHEDULED_SOURCE_ORDER keeps the source order instead. The tuning knobs are the
 * policy's (MxGemmPipelineAgBgCrCompTDMScheduledPolicy).
 *
 * M, N and K are run time; the Problem's traits (MxTdmScheduledGemmTraits) carry the code paths
 * that depend on them (edges, wide offsets, a K tail). Tile t uses LDS buffer t % 2 and the loop
 * runs tile pairs; Problem::TailNum is the variant that runs the odd tile before the pairs
 * (TailNumber::One) or none (Two), so every tile's buffer is a compile-time constant. The host
 * picks it from the whole-tile count with the base's TailHandler and launches that kernel.
 */
template <typename Problem_, typename Policy_ = MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>>
struct MxGemmPipelineAgBgCrCompTDMScheduled
    : public BaseMxGemmPipelineAgBgCrCompTDMScheduled<remove_cvref_t<Problem_>>
{
    using Base      = BaseMxGemmPipelineAgBgCrCompTDMScheduled<remove_cvref_t<Problem_>>;
    using Problem   = remove_cvref_t<Problem_>;
    using Policy    = remove_cvref_t<Policy_>;
    using Traits    = typename Problem::Traits;
    using ADataType = typename Problem::ADataType;
    using BDataType = typename Problem::BDataType;
    using ODataType = typename Problem::ODataType;

    using AsDataType     = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType     = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using AsLayout       = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout       = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using CLayout        = remove_cvref_t<typename Problem::CLayout>;
    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;

    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    /// This pipeline with Problem::TailNum T: the kernel the base's TailHandler picks.
    template <TailNumber T>
    using WithTailNum =
        MxGemmPipelineAgBgCrCompTDMScheduled<typename Problem::template WithTailNum<T>, Policy>;

    static_assert(mx_tdm_scheduled::InvalidTuning<Policy::template GetReason<Problem>()>::value,
                  "comp_tdm_scheduled: the Problem and policy break a rule of the policy's "
                  "GetReason");

    static constexpr auto Scheduler = Problem::Scheduler;
    static_assert(Scheduler == GemmPipelineScheduler::Intrawave,
                  "comp_tdm_scheduled is one explicitly scheduled intrawave loop: "
                  "Problem::Scheduler must be GemmPipelineScheduler::Intrawave");

    static constexpr index_t kBlockSize     = Problem::kBlockSize;
    static constexpr index_t BlockSize      = Problem::kBlockSize;
    static constexpr bool kPadM             = Problem::kPadM;
    static constexpr bool kPadN             = Problem::kPadN;
    static constexpr bool kPadK             = Problem::kPadK;
    static constexpr bool Preshuffle        = false;
    static constexpr bool LargeTensors      = Problem::LargeTensors;
    static constexpr index_t ScaleBlockSize = Problem::ScaleBlockSize;
    static constexpr index_t MXdlPackEff    = 1;
    static constexpr index_t NXdlPackEff    = 1;
    static constexpr index_t KXdlPackEff    = 4;
    /// One MX GEMM of single A, B and C tensors with K-major scales: the multi-D, grouped and
    /// batched kernels reject this pipeline.
    static constexpr bool kPlainGemmOnly = true;

    // TDM reads A and B, so the vector sizes are 1 and the launch check of the vector lengths is
    // skipped, as for the other TDM pipelines.
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return 1;
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return 1;
    }
    static constexpr bool skipCheckValidLaunchParams = true;
    /// The scales are K-major, [ceil(K / 128)][MN] int32 packs of four E8M0 scales: the kernel's
    /// scale windows are (MN, K / 128) views of that layout.
    static constexpr bool kKMajorScales = true;
    /// Runs k_batch == 1 only.
    static constexpr bool kSupportsSplitK = false;
    /// kPadM / kPadN (EdgeM / EdgeN) are false only when the tiles divide M / N.
    static constexpr bool kExactPadMN = true;
    /// The cluster multicast takes grid x as N tiles and grid y as M tiles
    /// (GemmTile2DNMajorPartitioner).
    static constexpr bool kGridXWalksN = true;

    /// The workgroup's threads: kBlockSize of the device pass (4 waves of 32).
    static constexpr index_t kThreads = Policy::template GetBlockSize<Problem>();
    /// The LDS the workgroup may take, for the pipeline and the epilogue together.
    static constexpr index_t kLdsCapacityBytes = Policy::kLdsCapacityBytes;
    static constexpr index_t kTileM            = BlockGemmShape::kM;
    static constexpr index_t kTileN            = BlockGemmShape::kN;
    static constexpr index_t kTileK            = BlockGemmShape::kK;
    static constexpr index_t kWarpM            = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t kWarpN            = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t kRm               = Policy::template GetWaveRepeatM<Problem>();
    static constexpr index_t kRn               = Policy::template GetWaveRepeatN<Problem>();
    static constexpr index_t kKSteps           = Policy::template GetKSteps<Problem>();
    static constexpr index_t kAtomM            = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr index_t kKPerByte         = Policy::template GetKPerByte<Problem>();
    static constexpr bool kPrefetchLdsRead     = Policy::template GetPrefetchLdsRead<Problem>();
    static constexpr bool kSplitBLdsRead       = Policy::template GetSplitBLdsRead<Problem>();
    static constexpr index_t kHalf             = kRn / 2;
    static constexpr bool kHasCluster          = Policy::template HasCluster<Problem>();
    static constexpr index_t kClusterX         = BlockGemmShape::kclusterN;
    static constexpr index_t kClusterY         = BlockGemmShape::kclusterM;
    static constexpr index_t kClusterZ         = BlockGemmShape::kclusterK;
    static_assert(kClusterZ == 1, "comp_tdm_scheduled: the cluster spans no K");
    static constexpr index_t kGl2Distance     = Policy::kGl2PrefetchDistance;
    static constexpr index_t kReadsPerStep    = Policy::template GetReadsPerStep<Problem>();
    static constexpr index_t kLdsRowBytes     = Policy::template GetLdsRowBytes<Problem>();
    static constexpr index_t kLdsBufferStride = Policy::template GetLdsBufferStride<Problem>();
    static constexpr index_t kLatePending     = Policy::template GetLatePendingReads<Problem>();
    /// The sched_fence mask of the pipeline's fences: LDS reads and WMMAs stay put, ALU work may
    /// cross.
    static constexpr int32_t kAluMayCross =
        LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::SALU | LLVMSchedGroupMask::TRANS;
    static constexpr bool kSourceOrder = CK_TILE_TDM_SCHEDULED_SOURCE_ORDER != 0;
    /// The fence mask around the barriers: with the machine scheduler SALU stays put too, as the
    /// barriers are SALU.
    static constexpr int32_t kBarrierFence =
        kSourceOrder ? kAluMayCross : LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::TRANS;
    /// The refill's cluster barrier in two halves: the arrival once every wave has drained its
    /// reads of the buffer, the wait after the last K step's WMMAs (fill_arrive, fill_wait).
    static constexpr bool kSplitCluster = kHasCluster && !kSourceOrder;
    /// A K step's ds_loads (unclamped): kReadsA with LocalPrefetchA (scales and A), kReadsBFrag
    /// per B fragment.
    static constexpr index_t kReadsBFrag = Policy::template GetChunksPerLane<Problem>();
    static constexpr index_t kReadsA =
        (kRm < 8 ? 1 : kRm / 4) + (kRn < 8 ? 1 : kRn / 4) + kReadsBFrag * (kAtomM / 16) * kRm;
    /// Half B's reads before its first half's WMMAs, and the s_wait_dscnt of a refill after them.
    static constexpr index_t kReadsHalfB   = kReadsA + kReadsBFrag * kHalf;
    static constexpr index_t kPendingHalfB = kReadsHalfB < 63 ? kReadsHalfB : 63;
    static_assert(min(kReadsA + kReadsBFrag * kRn, index_t{63}) == kReadsPerStep,
                  "comp_tdm_scheduled: the read counts are the policy's GetReadsPerStep");
    using BlockGemm  = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using CBlockTile = typename BlockGemm::CBlockTile;
    using AccTile    = typename BlockGemm::AccTile;
    using ATile      = typename BlockGemm::ATile;
    using BTile      = typename BlockGemm::BTile;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST static std::string GetName()
    {
        return std::string("pipeline_AgBgCrCompTDMScheduled") +
               Policy::template GetTuningName<Problem>();
    }

    /// The kernel-name spelling of an A or B type.
    template <typename T>
    CK_TILE_HOST static const char* GetDtypeName()
    {
        return std::is_same_v<T, pk_fp4_t> ? "fp4" : std::is_same_v<T, fp8_t> ? "fp8" : "bf8";
    }
    /// The kernel-name type tag: mx, A's type, then B's if it differs.
    CK_TILE_HOST static std::string GetDtypeTag()
    {
        std::string tag = std::string("mx") + GetDtypeName<ADataType>();
        if(!std::is_same_v<ADataType, BDataType>)
            tag += GetDtypeName<BDataType>();
        return tag;
    }
    CK_TILE_HOST static std::string GetTileName()
    {
        return "_t" + std::to_string(kTileM) + "x" + std::to_string(kTileN) + "x" +
               std::to_string(kTileK);
    }

    /// The name of an MxGemmKernel of this pipeline: data types, tile, tuning and traits.
    [[nodiscard]] CK_TILE_HOST static std::string GetKernelName()
    {
        std::string name = "mx_gemm_tdm_scheduled_" + GetDtypeTag() + GetTileName() +
                           Policy::template GetTuningName<Problem>(true);
        if(Traits::kEdgeM || Traits::kEdgeN)
            name += "_edge";
        if(Traits::kWideA || Traits::kWideB || Traits::kWideD)
            name += "_wide";
        if(Traits::kHasTail)
            name += "_tail" + std::to_string(Traits::kTailSteps);
        if(Problem::TailNum == TailNumber::One)
            name += "_odd";
        else if(Problem::TailNum == TailNumber::Two)
            name += "_even";
        return name;
    }

    /// nullptr if this instance runs an M x N x K problem, else why not: a rule that needs the
    /// problem's shape (the others hold at compile time), or a shape path the traits lack.
    CK_TILE_HOST static const char* GetUnsupportedReason(index_t M, index_t N, index_t K)
    {
        using R  = MxTdmScheduledReason;
        R reason = R::Ok;
        // A, B and D offsets switch to i64 past 2^31; the scales' stay i32. A partial last tile's
        // rows reach the next whole tile.
        const long_index_t span_m  = integer_least_multiple(long_index_t{M}, kTileM);
        const long_index_t span_n  = integer_least_multiple(long_index_t{N}, kTileN);
        const long_index_t mn_span = span_m > span_n ? span_m : span_n;
        // The refill loads two tiles ahead and the GL2 prefetch kGl2Distance; both compute their
        // K in i32 before clamping it to K.
        constexpr long_index_t kKLookahead =
            long_index_t{(kGl2Distance > 2 ? kGl2Distance : 2) + 1} * kTileK;
        if(M <= 0 || N <= 0 || K <= 0)
            reason = R::ShapePositive;
        else if(K % 32)
            reason = R::KScaleBlock;
        else if(mn_span * integer_divide_ceil(long_index_t{K}, Policy::kAtomK) * 4 > (1ll << 31))
            reason = R::ScaleSpan;
        else if(K < kTileK)
            reason = R::KShort;
        else if(long_index_t{K} > numeric<index_t>::max() - kKLookahead)
            reason = R::KLong;
        else if(K % kTileK > kTileK - Policy::kAtomK)
            reason = R::KTailFull;
        else if(integer_divide_ceil(N, kTileN) % kClusterX ||
                integer_divide_ceil(M, kTileM) % kClusterY)
            reason = R::ClusterGrid;
        if(reason != R::Ok)
            return mx_tdm_scheduled_reason_message(reason);
        const MxTdmScheduledShapeTraits need = MxTdmScheduledSelectTraits<Problem>(M, N, K);
        if(need.edge_m && !Traits::kEdgeM)
            return "M is not a multiple of tile_m: needs an EdgeM instance";
        if(need.edge_n && !Traits::kEdgeN)
            return "N is not a multiple of tile_n: needs an EdgeN instance";
        if(need.wide_a && !Traits::kWideA)
            return "A's offsets pass 2^31: needs a WideA instance";
        if(need.wide_b && !Traits::kWideB)
            return "B's offsets pass 2^31: needs a WideB instance";
        if(need.wide_d && !Traits::kWideD)
            return "D's offsets pass 2^31: needs a WideD instance";
        if(need.tail_steps != Traits::kTailSteps)
            return "the K tail (K mod tile_k, in 128-wide K steps) is not the instance's TailSteps";
        if(Problem::TailNum != TailNumber::Full && Base::GetTailNumber(K) != Problem::TailNum)
            return "the whole K tile count's parity is not the instance's TailNum "
                   "(the base's TailHandler picks it)";
        return nullptr;
    }

    /// The DRAM step between the iterate-mode blocks of A's or B's data load: iterate_rows rows of
    /// row_bytes. It passes 2^32 at long K, and the descriptor field holds 48 bits.
    template <bool IsA>
    CK_TILE_HOST_DEVICE static constexpr uint64_t GetIterateGlobalIncrement(index_t row_bytes)
    {
        constexpr index_t tile_rows = Policy::template GetLdsOperand<Problem, IsA>().iterate_rows;
        return static_cast<uint64_t>(row_bytes) * static_cast<uint64_t>(tile_rows);
    }

    template <typename ADramWindow,
              typename BDramWindow,
              typename AScaleDramWindow,
              typename BScaleDramWindow>
    struct PipelineImpl
    {
        // A's and B's data and K-major scale DRAM windows of the tile, and the problem from the
        // data windows' views.
        ADramWindow a_dram;
        BDramWindow b_dram;
        AScaleDramWindow a_scale_dram;
        BScaleDramWindow b_scale_dram;
        index_t M;
        index_t N;
        index_t K;
        // The calling thread's place: thread, wave, the tile's first row and column (the A and B
        // windows' origins) and the cluster (0 without one).
        index_t tid;
        index_t wave;
        index_t wave_m;
        index_t wave_n;
        index_t block_m;
        index_t block_n;
        index_t cluster_wx = 0;
        index_t cluster_wy = 0;
        CK_TILE_LDS_ADDR uint8_t* smem;
        index_t packed_k;
        index_t k_atoms;
        uint16_t a_mask = 0;
        uint16_t b_mask = 0;
        TDMGroups data{};
        TDMGroups scale{};
        // Group 0 of each load is rebuilt from these: patching the hoisted descriptor's own
        // words keeps it live, and every load then copies it into a fresh register quad.
        uint32_t data_lds     = 0;
        uint32_t scale_lds    = 0;
        uint64_t data_global  = 0;
        uint64_t scale_global = 0;
        index_t scale_stride  = 0;
        // Scale row bytes for the GL2 prefetch: as M * 4 / N * 4, LLVM factors the 4 out of
        // row * stride + offset and the address takes an extra VALU op.
        index_t a_scale_stride;
        index_t b_scale_stride;
        BlockGemm block_gemm;

        CK_TILE_DEVICE PipelineImpl(const ADramWindow& a_dram_,
                                    const BDramWindow& b_dram_,
                                    const AScaleDramWindow& a_scale_dram_,
                                    const BScaleDramWindow& b_scale_dram_,
                                    CK_TILE_LDS_ADDR uint8_t* smem_)
            : a_dram(a_dram_),
              b_dram(b_dram_),
              a_scale_dram(a_scale_dram_),
              b_scale_dram(b_scale_dram_),
              M(Lengths(a_dram_)[number<0>{}]),
              N(Lengths(b_dram_)[number<0>{}]),
              K(Lengths(a_dram_)[number<1>{}]),
              tid(get_thread_id()),
              // Wave-uniform, so the TDM descriptors built from it stay scalar.
              wave(get_warp_id()),
              wave_m(integer_divide_nonnegative<kWarpN>(wave)),
              wave_n(integer_modulo_nonnegative<kWarpN>(wave)),
              block_m(a_dram_.get_window_origin()[number<0>{}]),
              block_n(b_dram_.get_window_origin()[number<0>{}]),
              smem(smem_),
              packed_k(integer_divide_nonnegative<kKPerByte>(K)),
              // The last K step is partial (and its scales zero-padded) only with a K tail.
              k_atoms(integer_divide_nonnegative<Policy::kAtomK>(
                  Traits::kHasTail ? K + Policy::kAtomK - 1 : K)),
              a_scale_stride(detail::opaque_scalar(M * 4)),
              b_scale_stride(detail::opaque_scalar(N * 4)),
              block_gemm(smem_, wave_m, wave_n)
        {
            if constexpr(kHasCluster)
            {
                cluster_wx = static_cast<index_t>(get_cluster_workgroup_id_x());
                cluster_wy = static_cast<index_t>(get_cluster_workgroup_id_y());
            }
            const dim3 block_id_in_cluster{
                static_cast<uint32_t>(cluster_wx), static_cast<uint32_t>(cluster_wy), 0};
            b_mask = Policy::template GetTDMWorkgroupMask<MultiCastDirection::kN, Problem>(
                block_id_in_cluster);
            a_mask = Policy::template GetTDMWorkgroupMask<MultiCastDirection::kM, Problem>(
                block_id_in_cluster);
        }

        template <typename Window>
        CK_TILE_DEVICE static const uint8_t* Bytes(const Window& window)
        {
            return reinterpret_cast<const uint8_t*>(
                window.get_bottom_tensor_view().get_buffer_view().p_data_);
        }
        template <typename Window>
        CK_TILE_DEVICE static auto Lengths(const Window& window)
        {
            return detail::tdm_real_lengths(
                window.get_bottom_tensor_view().get_tensor_descriptor());
        }

        CK_TILE_DEVICE static constexpr index_t BufferBase(index_t parity)
        {
            return parity * kLdsBufferStride;
        }

        /// The K origin of a load at k: past the end of K the last tile is issued again (so the
        /// buffer holds data the GEMM never reads), where a clipped window load would write
        /// zeros.
        CK_TILE_DEVICE index_t ClampK(index_t k) const { return k < K ? k : K - kTileK; }

        template <bool IsA>
        using DramWindowType = std::conditional_t<IsA, ADramWindow, BDramWindow>;
        template <bool IsA>
        using ScaleDramWindowType = std::conditional_t<IsA, AScaleDramWindow, BScaleDramWindow>;

        template <bool IsA>
        CK_TILE_DEVICE const auto& DramWindow() const
        {
            if constexpr(IsA)
                return a_dram;
            else
                return b_dram;
        }

        /// Component `component` of A's or B's tile as a TDM load window: a sub-window of the
        /// operand's DRAM window at the component's first row, one box of
        /// MakeDramTdmTileDistribution. The view is rebased at that row so the window's offset
        /// is K only: a tensor descriptor offset is 32-bit and counts elements, which pass 2^31
        /// at half the byte offset with packed types (a 65536 x 65536 FP4 A). The row's byte
        /// offset is 32-bit as WideA / WideB say.
        template <bool IsA>
        CK_TILE_DEVICE auto DramComponentWindow(index_t component) const
        {
            using DataType             = remove_cvref_t<typename DramWindowType<IsA>::DataType>;
            constexpr auto L           = Policy::template GetLdsOperand<Problem, IsA>();
            constexpr bool wide        = IsA ? Traits::kWideA : Traits::kWideB;
            constexpr index_t packed   = numeric_traits<DataType>::PackedSize;
            constexpr index_t box_rows = L.iterate_rows > 0 ? L.iterate_rows : L.rows;
            constexpr auto lengths     = make_tuple(number<box_rows>{}, number<kTileK>{});
            constexpr auto dstr = Policy::template MakeDramTdmTileDistribution<Problem, IsA>();
            using RowOffset     = std::conditional_t<wide, long_index_t, index_t>;
            const auto& dram    = DramWindow<IsA>();
            const auto& view    = dram.get_bottom_tensor_view();
            const index_t row0  = dram.get_window_origin()[number<0>{}] + component * L.rows;
            const index_t k0    = dram.get_window_origin()[number<1>{}];
            const index_t row_stride =
                view.get_tensor_descriptor().calculate_offset(make_multi_index(1, 0));
            const auto rebased = make_naive_tensor_view<address_space_enum::global>(
                view.get_buffer_view().p_data_ +
                    static_cast<RowOffset>(row0) * (row_stride / packed),
                make_tuple(Lengths(dram)[number<0>{}] - row0, K),
                make_tuple(row_stride, 1));
            return make_tile_window(rebased, lengths, {0, k0}, dstr);
        }

        /// Where component `component` of A's or B's tile lands in buffer 0: an LDS window over
        /// the policy's TDM layout of the operand.
        template <bool IsA>
        CK_TILE_DEVICE auto LdsComponentWindow(index_t component) const
        {
            using DataType             = remove_cvref_t<typename DramWindowType<IsA>::DataType>;
            constexpr auto L           = Policy::template GetLdsOperand<Problem, IsA>();
            constexpr index_t box_rows = L.iterate_rows > 0 ? L.iterate_rows : L.rows;
            constexpr auto desc        = Policy::template MakeLdsTdmBlockDescriptor<Problem, IsA>();
            // The view starts at the component, offset in words as the block GEMM's reads offset
            // their rows, so the two share the wave's base.
            auto* const base = reinterpret_cast<CK_TILE_LDS_ADDR uint32_t*>(smem + L.base) +
                               component * (L.stride / 4);
            const auto view = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<CK_TILE_LDS_ADDR DataType*>(base), desc);
            return make_tile_window(view, make_tuple(number<box_rows>{}, number<kTileK>{}), {0, 0});
        }

        template <bool IsA>
        CK_TILE_DEVICE const auto& ScaleDramWindow() const
        {
            if constexpr(IsA)
                return a_scale_dram;
            else
                return b_scale_dram;
        }

        /// Component `component` of A's or B's K-major scales as a TDM load window: the tile's K
        /// steps by the component's rows, one box of MakeDramTdmTileDistribution.
        template <bool IsA>
        CK_TILE_DEVICE auto ScaleDramComponentWindow(index_t component) const
        {
            constexpr auto L = Policy::template GetLdsOperand<Problem, IsA, true>();
            constexpr auto dstr =
                Policy::template MakeDramTdmTileDistribution<Problem, IsA, true>();
            const auto& dram = ScaleDramWindow<IsA>();
            return make_tile_window(dram.get_bottom_tensor_view(),
                                    make_tuple(number<kKSteps>{}, number<L.rows>{}),
                                    {dram.get_window_origin()[number<0>{}],
                                     dram.get_window_origin()[number<1>{}] + component * L.rows},
                                    dstr);
        }

        /// Where component `component` of A's or B's scales lands in buffer 0: an LDS window over
        /// the policy's TDM layout of the scales, starting at the component.
        template <bool IsA>
        CK_TILE_DEVICE auto LdsScaleComponentWindow(index_t component) const
        {
            using DataType      = remove_cvref_t<typename ScaleDramWindowType<IsA>::DataType>;
            constexpr auto L    = Policy::template GetLdsOperand<Problem, IsA, true>();
            constexpr auto desc = Policy::template MakeLdsScaleTdmBlockDescriptor<Problem, IsA>();
            const auto view     = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<CK_TILE_LDS_ADDR DataType*>(smem + L.base + component * L.stride),
                desc);
            return make_tile_window(view, make_tuple(number<kKSteps>{}, number<L.rows>{}), {0, 0});
        }

        // ---------------------------------------------------------------- TdmLoader

        /// The TDMConfig of A's or B's data or scales: its LDS padding and cluster multicast mask.
        template <bool IsA, bool IsScale>
        CK_TILE_DEVICE TDMConfig Config() const
        {
            return PipelineImplBase::template MakeTdmConfig<IsA, IsScale>(IsA ? a_mask : b_mask);
        }

        /// The groups of A's or B's component load into buffer 0, from the component's DRAM
        /// window into its LDS window, as tile_window::make_tdm_groups builds them. With
        /// iterate-mode blocks, the block count is what the component's rows left need.
        template <bool IsA>
        CK_TILE_DEVICE TDMGroups DataDescriptor(index_t component) const
        {
            constexpr auto L       = Policy::template GetLdsOperand<Problem, IsA>();
            constexpr bool edge    = IsA ? Traits::kEdgeM : Traits::kEdgeN;
            constexpr bool iterate = L.iterate_rows > 0;
            const auto dram_window = DramComponentWindow<IsA>(component);

            TDMConfig config = Config<IsA, false>();
            if constexpr(iterate)
            {
                constexpr index_t tile_rows = L.iterate_rows;
                constexpr index_t count     = L.rows / tile_rows;
                const auto& view            = dram_window.get_bottom_tensor_view();
                index_t iterations          = count;
                if constexpr(edge)
                {
                    // Each iteration moves tile_rows rows whatever tensor_dim1 says, so walk
                    // ceil(rows left / tile_rows), at least one: with no rows left, tensor_dim1 0
                    // keeps that one off memory.
                    const index_t rows_left = max(index_t{0},
                                                  Lengths(dram_window)[number<0>{}] -
                                                      dram_window.get_window_origin()[number<0>{}]);
                    const index_t held      = max(min(rows_left, L.rows), 1);
                    iterations = integer_divide_nonnegative<tile_rows>(held + tile_rows - 1);
                }
                const index_t row_bytes = integer_divide_nonnegative<kKPerByte>(
                    view.get_tensor_descriptor().calculate_offset(make_multi_index(1, 0)));
                config.iterate_enable = true;
                config.iter_config    = {GetIterateGlobalIncrement<IsA>(row_bytes),
                                         static_cast<uint32_t>(L.iterate_stride),
                                         static_cast<uint16_t>(iterations - 1)};
            }
            return dram_window.make_tdm_groups(config, LdsComponentWindow<IsA>(component));
        }

        /// The groups of A's or B's component scale load into buffer 0, from the component's
        /// K-major scale window into its LDS window.
        template <bool IsA>
        CK_TILE_DEVICE TDMGroups ScaleDescriptor(index_t component) const
        {
            return ScaleDramComponentWindow<IsA>(component).make_tdm_groups(
                Config<IsA, true>(), LdsScaleComponentWindow<IsA>(component));
        }

        /// Both descriptors for tile 0, buffer 0, of this wave's loads; every word uniform. Even
        /// waves issue B's loads and odd waves A's, each one component: wave / 2.
        CK_TILE_DEVICE void Hoist()
        {
            const bool is_b_issuer  = integer_modulo_nonnegative<2>(wave) == 0;
            const index_t component = integer_divide_nonnegative<2>(wave);
            const auto select       = [&](const TDMGroups& b, const TDMGroups& a) {
                return amd_wave_read_first_lane(is_b_issuer ? b : a);
            };
            data      = select(DataDescriptor<false>(component), DataDescriptor<true>(component));
            scale     = select(ScaleDescriptor<false>(component), ScaleDescriptor<true>(component));
            data_lds  = data.g0.lds_addr;
            scale_lds = scale.g0.lds_addr;
            data_global  = data.g0.getGlobalAddr();
            scale_global = scale.g0.getGlobalAddr();
            scale_stride = is_b_issuer ? N * 4 : M * 4;
        }

        /// This wave's data and scale loads of the tile at k0 into buffer parity.
        CK_TILE_DEVICE void Issue(index_t k0, index_t parity) const
        {
            const index_t k_byte      = integer_divide_nonnegative<kKPerByte>(k0);
            const index_t remaining_k = packed_k - k_byte;
            PipelineImplBase::template GlobalPrefetchTDM<0>(
                data,
                data_lds + static_cast<uint32_t>(BufferBase(parity)),
                data_global + static_cast<uint32_t>(k_byte),
                static_cast<uint32_t>(remaining_k));
            const index_t k_step     = integer_divide_nonnegative<Policy::kAtomK>(k0);
            const index_t steps_left = k_atoms - k_step;
            PipelineImplBase::template GlobalPrefetchTDM<1>(
                scale,
                scale_lds + static_cast<uint32_t>(BufferBase(parity)),
                scale_global + static_cast<uint32_t>(k_step * scale_stride),
                static_cast<uint32_t>(steps_left));
        }

        /// Issue split in two, the descriptor arithmetic ahead of the refill's barrier and the
        /// loads after it.
        struct Refill
        {
            TDMGroups data;
            TDMGroups scale;
        };
        CK_TILE_DEVICE Refill PrepareIssue(index_t k0, index_t parity) const
        {
            const index_t k_byte     = integer_divide_nonnegative<kKPerByte>(k0);
            const index_t k_step     = integer_divide_nonnegative<Policy::kAtomK>(k0);
            const uint32_t lds_shift = static_cast<uint32_t>(BufferBase(parity));
            Refill r{data, scale};
            r.data.g0 =
                TDM_GROUP0{data_lds + lds_shift, data_global + static_cast<uint32_t>(k_byte), 0, 0};
            r.data.g1.tensorDim0(static_cast<uint32_t>(packed_k - k_byte));
            r.scale.g0 = TDM_GROUP0{scale_lds + lds_shift,
                                    scale_global + static_cast<uint32_t>(k_step * scale_stride),
                                    0,
                                    0};
            r.scale.g1.tensorDim1(static_cast<uint32_t>(k_atoms - k_step));
            return r;
        }
        CK_TILE_DEVICE static void IssuePrepared(const Refill& r)
        {
            amd_tdm_load(r.data.g0, r.data.g1, r.data.g2);
            amd_tdm_load(r.scale.g0, r.scale.g1, r.scale.g2);
        }

        // ---------------------------------------------------------------- Gl2Prefetcher

        /// Bytes between A's or B's rows, from the DRAM window's descriptor as the TDM loads take
        /// them.
        template <bool IsA>
        CK_TILE_DEVICE index_t DramRowBytes() const
        {
            return integer_divide_nonnegative<kKPerByte>(
                DramWindow<IsA>().get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                    make_multi_index(1, 0)));
        }

        /// The GL2 prefetch of what the cluster's TDM loads of the tile at k_ahead read, as
        /// tile_window::prefetch_for_tdm with DataCachePrefetchKind::L2 but for the whole cluster
        /// at once: A's and B's data and scale lines are split over every thread of the cluster,
        /// one 256 B line each, with SE_NT coherence. A line out of a partial tile is clamped to
        /// the last line inside instead of predicated off, and one prefetch per round covers
        /// every category: any branch in the loop body makes LLVM drain the LDS read pipeline.
        /// AheadParity is the parity of the prefetched tile's index: with 128 B rows one line
        /// holds an even tile and the next one, so the data lines are only prefetched for even
        /// tiles. Past the end of K the last tile is prefetched again.
        template <index_t AheadParity>
        CK_TILE_DEVICE void prefetch_for_tdm_cluster(index_t k_ahead) const
        {
            if constexpr(kGl2Distance != 0)
            {
                constexpr index_t line = Policy::kGl2LineBytes;
                k_ahead                = k_ahead < K ? k_ahead : K - kTileK;
                const index_t k_byte   = integer_divide_nonnegative<kKPerByte>(k_ahead);
                constexpr index_t data_lines =
                    kLdsRowBytes / line == 0 && AheadParity == 0 ? 1 : kLdsRowBytes / line;
                const index_t cluster_m0 = kHasCluster ? block_m - cluster_wy * kTileM : block_m;
                const index_t cluster_n0 = kHasCluster ? block_n - cluster_wx * kTileN : block_n;
                constexpr index_t b_scale_row     = kClusterX * kTileN * 4;
                constexpr index_t a_scale_row     = kClusterY * kTileM * 4;
                constexpr index_t b_lines_per_row = b_scale_row / line;
                constexpr index_t a_lines_per_row = a_scale_row / line;
                // Categories in order: A data, B data, B scales, A scales. Scales are prefetched
                // only for an operand whose cluster scale row is whole lines.
                constexpr index_t counts[4] = {
                    data_lines ? kClusterY * kTileM * data_lines : 0,
                    data_lines ? kClusterX * kTileN * data_lines : 0,
                    b_scale_row % line == 0 ? kKSteps * b_lines_per_row : 0,
                    a_scale_row % line == 0 ? kKSteps * a_lines_per_row : 0};
                constexpr index_t starts[4] = {
                    0, counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2]};
                constexpr index_t total = starts[3] + counts[3];
                constexpr index_t pool  = kClusterX * kClusterY * kThreads;
                const index_t thread =
                    kHasCluster ? (cluster_wx + cluster_wy * kClusterX) * kThreads + tid : tid;
                // A partial last tile has fewer K steps than the k_steps rows prefetched for it;
                // a partial M/N tile has lines past the end.
                const index_t k_step = integer_divide_nonnegative<Policy::kAtomK>(k_ahead);

                const auto address = [&](auto cat, index_t index) -> const uint8_t* {
                    constexpr index_t kCat = decltype(cat)::value;
                    if constexpr(kCat < 2)
                    {
                        constexpr bool is_a = kCat == 0;
                        constexpr bool edge = is_a ? Traits::kEdgeM : Traits::kEdgeN;
                        constexpr bool wide = is_a ? Traits::kWideA : Traits::kWideB;
                        // Dependent on kCat: a tile without data lines (data_lines 0) never
                        // instantiates the divide.
                        constexpr index_t lines = is_a ? data_lines : data_lines;
                        const index_t row0      = is_a ? cluster_m0 : cluster_n0;
                        const index_t unclamped = row0 + integer_divide_nonnegative<lines>(index);
                        const index_t row = edge ? min(unclamped, (is_a ? M : N) - 1) : unclamped;
                        const index_t offset =
                            k_byte + integer_modulo_nonnegative<lines>(index) * line;
                        const index_t row_bytes = DramRowBytes<is_a>();
                        const uint8_t* base     = Bytes(DramWindow<is_a>());
                        if constexpr(wide)
                            return base + (static_cast<uint64_t>(static_cast<uint32_t>(row)) *
                                               static_cast<uint32_t>(row_bytes) +
                                           static_cast<uint32_t>(offset));
                        else
                            return base + static_cast<uint32_t>(row * row_bytes + offset);
                    }
                    else
                    {
                        constexpr bool is_a = kCat == 3;
                        constexpr bool clamp =
                            (is_a ? Traits::kEdgeM : Traits::kEdgeN) || Traits::kHasTail;
                        constexpr index_t lines_per_row = is_a ? a_lines_per_row : b_lines_per_row;
                        const index_t row_bytes         = is_a ? a_scale_stride : b_scale_stride;
                        const index_t mn0               = is_a ? cluster_m0 : cluster_n0;
                        const index_t row =
                            k_step + integer_divide_nonnegative<lines_per_row>(index);
                        const index_t offset =
                            mn0 * 4 + integer_modulo_nonnegative<lines_per_row>(index) * line;
                        const index_t unclamped = row * row_bytes + offset;
                        const index_t element =
                            clamp ? min(unclamped, k_atoms * row_bytes - 1) : unclamped;
                        return Bytes(ScaleDramWindow<is_a>()) + static_cast<uint32_t>(element);
                    }
                };

                static_for<0, (total + pool - 1) / pool, 1>{}([&](auto round) {
                    constexpr index_t round_start = decltype(round)::value * pool;
                    constexpr index_t remaining   = total - round_start;
                    const index_t index =
                        remaining >= pool
                            ? thread + round_start
                            : integer_modulo_nonnegative<remaining>(thread) + round_start;
                    // Ranges are visited last to first, so each lane ends up with the first
                    // range whose end lies above its index.
                    const uint8_t* pointer = nullptr;
                    bool first             = true;
                    static_for<0, 4, 1>{}([&](auto rev) {
                        constexpr index_t cat   = 3 - decltype(rev)::value;
                        constexpr index_t start = starts[cat];
                        constexpr index_t count = counts[cat];
                        if constexpr(count > 0 && start < round_start + pool &&
                                     start + count > round_start)
                        {
                            const index_t local = start ? index - start : index;
                            const uint8_t* here = address(number<cat>{}, local);
                            pointer =
                                first
                                    ? here
                                    : detail::select_address(index < start + count, here, pointer);
                            first = false;
                        }
                    });
                    amd_global_prefetch<mx_tdm_scheduled::kGl2PrefetchCoherence>(pointer);
                });
            }
            else
            {
                ignore = k_ahead;
            }
        }

        // ---------------------------------------------------------------- Pipeline hooks

        template <index_t StaticParity>
        CK_TILE_DEVICE void PrefetchAhead(index_t k0) const
        {
            if constexpr(kGl2Distance != 0)
                prefetch_for_tdm_cluster<(StaticParity + kGl2Distance) % 2>(k0 +
                                                                            kGl2Distance * kTileK);
        }

        /// The GL2 prefetch ahead of tile k0, then (with local prefetch) the wait and barrier that
        /// publish tile k0 + 1 before its first read.
        /// Gl2: the GL2 prefetch is here, not a region earlier (gl2_early).
        template <index_t StaticParity, bool Gl2 = true>
        CK_TILE_DEVICE void publish_next_tile(index_t k0) const
        {
            if constexpr(Gl2)
                PrefetchAhead<StaticParity>(k0);
            sched_fence<kBarrierFence>();
            if constexpr(kPrefetchLdsRead)
            {
                // Tile k+1 is read next, from this point: it is the only TDM load in flight, and
                // the barrier below publishes it.
                s_wait_tensorcnt<0>();
                if constexpr(kSplitCluster)
                {
                    // Every read of tile k's buffer is issued: with them drained, the same
                    // barrier is the refill's arrival.
                    s_wait_dscnt_pinned<0>();
                    cluster_barrier_arrive();
                }
                else
                    s_barrier();
                sched_fence<kBarrierFence>();
            }
        }

        /// kSplitCluster without local prefetch: this wave's reads of the buffer drain, then the
        /// workgroup barrier and the workgroup's arrival on the cluster barrier.
        CK_TILE_DEVICE static void fill_arrive()
        {
            sched_fence<kBarrierFence>();
            s_wait_dscnt_pinned<0>();
            cluster_barrier_arrive();
            // kBarrierFence lets the step's WMMAs move above the arrival.
            sched_fence<>();
        }
        /// kSplitCluster: once every workgroup of the cluster has arrived (has read the buffer),
        /// refills it with the descriptors of prepare_fill. Called after the last K step's
        /// WMMAs, which run between the arrival and the wait.
        CK_TILE_DEVICE static void fill_wait(const Refill& refill)
        {
            sched_fence<kBarrierFence>();
            cluster_barrier_wait();
            sched_fence<kBarrierFence>();
            IssuePrepared(refill);
            sched_fence<kAluMayCross>();
        }

        /// Refills buffer parity with the tile two K tiles on, once the cluster and the
        /// workgroup have read it: this wave's reads of the buffer drain, then one barrier
        /// (cluster_barrier() includes the workgroup's) publishes every wave's. Pending is the
        /// count of the other buffer's reads issued since this buffer's last read. Without
        /// kSplitCluster (no cluster, or the source order).
        template <index_t Pending>
        CK_TILE_DEVICE static void fill_barrier()
        {
            sched_fence<kBarrierFence>();
            s_wait_dscnt_pinned<Pending>();
            if constexpr(kHasCluster)
                cluster_barrier();
            else
                s_barrier();
            if constexpr(!kSourceOrder)
                sched_fence<kBarrierFence>();
        }
        CK_TILE_DEVICE void fill_stage(index_t k0, index_t parity) const
        {
            fill_barrier<kLatePending>();
            Issue(ClampK(k0 + 2 * kTileK), parity);
            sched_fence<kAluMayCross>();
        }
        /// fill_stage with the descriptors of PrepareIssue(ClampK(k0 + 2 * kTileK), parity).
        template <index_t Pending>
        CK_TILE_DEVICE static void fill_stage(const Refill& refill)
        {
            fill_barrier<Pending>();
            IssuePrepared(refill);
            sched_fence<kAluMayCross>();
        }
        CK_TILE_DEVICE Refill prepare_fill(index_t k0, index_t parity) const
        {
            return PrepareIssue(ClampK(k0 + 2 * kTileK), parity);
        }

        /// With the machine scheduler, the GL2 prefetch of a tile of Steps > 1 K steps is issued
        /// in step Steps - 2, so that its address arithmetic spreads over that step's WMMAs.
        template <index_t Steps, index_t Step>
        static constexpr bool gl2_early = !kSourceOrder && Steps > 1 && Step + 2 == Steps;
        template <index_t Steps>
        static constexpr bool gl2_at_publish = kSourceOrder || Steps == 1;

        /// The next tile has landed (only the two newer loads stay in flight) and every wave sees
        /// it.
        CK_TILE_DEVICE static void wait_next_tile()
        {
            s_wait_tensorcnt_barrier<2>();
            compiler_memory_fence();
        }

        // ---------------------------------------------------------------- TileCompute

        /// The WMMAs of B fragments [First, First + Count): half-B order (rn outer, rm inner) with
        /// BOuter, else rm outer.
        template <bool BOuter, index_t First, index_t Count>
        CK_TILE_DEVICE void gemm(AccTile& acc, const ATile& a, const BTile& b) const
        {
            block_gemm.template Run<First, Count, BOuter, 0, kRm * Count>(acc, a, b);
        }

        /// The wait for a K step's reads before its WMMAs: pinned in the source order, else left
        /// to the compiler's waitcnt insertion.
        template <index_t N>
        CK_TILE_DEVICE static void step_wait()
        {
            if constexpr(kSourceOrder)
                s_wait_dscnt_pinned<N>();
        }

        /// Closes a scheduling region of Reads ds_loads (the next K step's) and Wmmas WMMAs, as
        /// V1's HotLoopScheduler: the reads go to the first Window WMMAs, ceil(Reads / Window) per
        /// WMMA or one per ceil(Window / Reads) WMMAs, and sched_barrier(0) ends the region.
        /// Front-loaded reads keep their distance to the next step's WMMAs.
        template <index_t Reads, index_t Wmmas, index_t WindowWanted = Wmmas>
        CK_TILE_DEVICE static void hot_loop_scheduler()
        {
            constexpr index_t Window = WindowWanted < 1 ? 1 : WindowWanted;
            if constexpr(!kSourceOrder)
            {
                if constexpr(Reads > 0 && Reads <= Window)
                    static_for<0, Reads, 1>{}([](auto) {
                        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::DS_READ, 1, 0);
                        __builtin_amdgcn_sched_group_barrier(
                            LLVMSchedGroupMask::MFMA, (Window + Reads - 1) / Reads, 0);
                    });
                else if constexpr(Reads > Window)
                    static_for<0, Window, 1>{}([](auto) {
                        __builtin_amdgcn_sched_group_barrier(
                            LLVMSchedGroupMask::DS_READ, (Reads + Window - 1) / Window, 0);
                        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
                    });
                if constexpr(Reads > 0 && Window < Wmmas)
                    __builtin_amdgcn_sched_group_barrier(
                        LLVMSchedGroupMask::MFMA, Wmmas - Window, 0);
                __builtin_amdgcn_sched_barrier(0);
            }
        }

        /// PrefetchLdsRead and SplitBLdsRead: all reads of K step s + 1 before the first wait but
        /// B's second half, which follows the first half's WMMAs.
        /// The WMMAs run B outer (rn outer, rm inner), so each half's WMMAs use only that half's
        /// B fragments and each B fragment feeds kRm WMMAs in a row.
        /// The same split applies to A (A's halves, rm outer); only B's is implemented.
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void k_tile_half_b(AccTile& acc,
                                          ATile& a_tile,
                                          BTile& b_tile,
                                          index_t k0,
                                          index_t parity,
                                          index_t buffer,
                                          index_t next_buffer) const
        {
            Refill refill;
            static_for<0, Steps, 1>{}([&](auto s) {
                constexpr index_t step = decltype(s)::value;
                constexpr bool last    = step + 1 == Steps;
                const ATile cur        = a_tile;
                const BTile cur_b      = b_tile;
                if constexpr(!last)
                {
                    a_tile = block_gemm.LocalPrefetchA(buffer, step + 1);
                    block_gemm.template LocalPrefetchB<0, kHalf>(b_tile, buffer, step + 1);
                }
                else
                {
                    publish_next_tile<StaticParity, gl2_at_publish<Steps>>(k0);
                    if constexpr(!kSourceOrder)
                        refill = prepare_fill(k0, parity);
                    a_tile = block_gemm.LocalPrefetchA(next_buffer, 0);
                    block_gemm.template LocalPrefetchB<0, kHalf>(b_tile, next_buffer, 0);
                }
                step_wait<kReadsPerStep>();
                gemm<true, 0, kHalf>(acc, cur, cur_b);
                hot_loop_scheduler<kReadsHalfB, kRm * kHalf>();
                if constexpr(!last)
                    block_gemm.template LocalPrefetchB<kHalf, kHalf>(b_tile, buffer, step + 1);
                else if constexpr(kSourceOrder)
                {
                    block_gemm.template LocalPrefetchB<kHalf, kHalf>(b_tile, next_buffer, 0);
                    fill_stage(k0, parity);
                }
                else
                {
                    // The refill first, so that B's second half spreads over the WMMAs below.
                    if constexpr(!kSplitCluster)
                        fill_stage<kPendingHalfB>(refill);
                    block_gemm.template LocalPrefetchB<kHalf, kHalf>(b_tile, next_buffer, 0);
                }
                if constexpr(gl2_early<Steps, step>)
                    PrefetchAhead<StaticParity>(k0);
                step_wait<kReadsPerStep>();
                gemm<true, kHalf, kHalf>(acc, cur, cur_b);
                hot_loop_scheduler<kReadsBFrag * kHalf, kRm * kHalf>();
                if constexpr(last && kSplitCluster)
                    fill_wait(refill);
            });
        }

        /// PrefetchLdsRead, B whole.
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void k_tile_full(AccTile& acc,
                                        ATile& a_tile,
                                        BTile& b_tile,
                                        index_t k0,
                                        index_t parity,
                                        index_t buffer,
                                        index_t next_buffer) const
        {
            Refill refill;
            static_for<0, Steps, 1>{}([&](auto s) {
                constexpr index_t step = decltype(s)::value;
                constexpr bool last    = step + 1 == Steps;
                const ATile cur        = a_tile;
                const BTile cur_b      = b_tile;
                if constexpr(!last)
                {
                    a_tile = block_gemm.LocalPrefetchA(buffer, step + 1);
                    block_gemm.template LocalPrefetchB<0, kRn>(b_tile, buffer, step + 1);
                }
                else if constexpr(kSourceOrder)
                {
                    publish_next_tile<StaticParity>(k0);
                    a_tile = block_gemm.LocalPrefetchA(next_buffer, 0);
                    block_gemm.template LocalPrefetchB<0, kRn>(b_tile, next_buffer, 0);
                    fill_stage(k0, parity);
                }
                else
                {
                    // The refill first, so that the next tile's reads spread over the WMMAs.
                    if constexpr(Steps == 1)
                        refill = prepare_fill(k0, parity);
                    publish_next_tile<StaticParity, gl2_at_publish<Steps>>(k0);
                    if constexpr(!kSplitCluster)
                        fill_stage<0>(refill);
                    a_tile = block_gemm.LocalPrefetchA(next_buffer, 0);
                    block_gemm.template LocalPrefetchB<0, kRn>(b_tile, next_buffer, 0);
                }
                if constexpr(gl2_early<Steps, step>)
                {
                    PrefetchAhead<StaticParity>(k0);
                    refill = prepare_fill(k0, parity);
                }
                step_wait<kReadsPerStep>();
                gemm<false, 0, kRn>(acc, cur, cur_b);
                // The last step's barrier drains these reads first: they take half the WMMAs.
                hot_loop_scheduler<kReadsA + kReadsBFrag * kRn,
                                   kRm * kRn,
                                   step + 2 == Steps ? kRm * kRn / 2 : kRm * kRn>();
                if constexpr(last && kSplitCluster)
                    fill_wait(refill);
            });
        }

        /// PrefetchLdsRead off: each K step reads, drains and computes. The tile's last read is the
        /// last K step's: its refill (GL2 prefetch, barrier, TDM) follows the drain, before the
        /// step's WMMAs; with kSplitCluster only the arrival does, and the wait and TDM follow
        /// the WMMAs.
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void
        k_tile_direct(AccTile& acc, index_t k0, index_t parity, index_t buffer) const
        {
            Refill refill;
            static_for<0, Steps, 1>{}([&](auto s) {
                constexpr index_t step = decltype(s)::value;
                constexpr bool last    = step + 1 == Steps;
                const ATile h          = block_gemm.LocalPrefetchA(buffer, step);
                BTile b;
                block_gemm.template LocalPrefetchB<0, kRn>(b, buffer, step);
                step_wait<0>();
                if constexpr(last && kSourceOrder)
                {
                    publish_next_tile<StaticParity>(k0);
                    fill_stage(k0, parity);
                }
                else if constexpr(last)
                {
                    if constexpr(Steps == 1)
                        refill = prepare_fill(k0, parity);
                    publish_next_tile<StaticParity, gl2_at_publish<Steps>>(k0);
                    if constexpr(kSplitCluster)
                        fill_arrive();
                    else
                        fill_stage<0>(refill);
                }
                if constexpr(gl2_early<Steps, step>)
                {
                    PrefetchAhead<StaticParity>(k0);
                    refill = prepare_fill(k0, parity);
                }
                gemm<false, 0, kRn>(acc, h, b);
                // The step's reads feed its own WMMAs: no hints, the region only ends the step.
                if constexpr(!kSourceOrder)
                    __builtin_amdgcn_sched_barrier(0);
                if constexpr(last && kSplitCluster)
                    fill_wait(refill);
            });
        }

        /// One K tile at k0 in buffer parity: StaticParity is its parity when known at compile
        /// time (the GL2 short-row choice), Steps its K step count.
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void
        k_tile(AccTile& acc, ATile& a_tile, BTile& b_tile, index_t k0, index_t parity) const
        {
            const index_t buffer      = BufferBase(parity);
            const index_t next_buffer = BufferBase(1 - parity);
            if constexpr(!kPrefetchLdsRead)
            {
                wait_next_tile();
                k_tile_direct<Steps, StaticParity>(acc, k0, parity, buffer);
            }
            else if constexpr(kSplitBLdsRead)
                k_tile_half_b<Steps, StaticParity>(
                    acc, a_tile, b_tile, k0, parity, buffer, next_buffer);
            else
                k_tile_full<Steps, StaticParity>(
                    acc, a_tile, b_tile, k0, parity, buffer, next_buffer);
        }

        /// The K loop: full_tiles whole K tiles, then the K tail with Traits::kHasTail.
        /// TailNum is the base's for full_tiles; TailNumber::Full has the odd tile and the K
        /// tail's buffer chosen at run time.
        template <TailNumber TailNum>
        CK_TILE_DEVICE void operator()(AccTile& acc, index_t full_tiles)
        {
            // builds the TDM descriptors once, before the loop and patch per tile
            Hoist();
            if constexpr(kHasCluster)
                cluster_barrier();
            else
                s_barrier();
            // TDM loads of tile 0 (data and scales) into buffer 0
            Issue(0, 0);
            // With one tile, buffer 1 gets tile 0 again and is never read (a).
            Issue(ClampK(kTileK), 1);
            // Issue GL2 prefetch hints for tile2 and later.
            static_for<2, (kGl2Distance > 2 ? kGl2Distance : 2), 1>{}([&](auto ahead) {
                prefetch_for_tdm_cluster<decltype(ahead)::value % 2>(ahead * kTileK);
            });
            if constexpr(kPrefetchLdsRead)
                wait_next_tile();
            clear_tile(acc);
            // The register tiles of the next K step, read one step ahead.
            ATile a_tile{};
            BTile b_tile{};
            if constexpr(kPrefetchLdsRead)
            {
                a_tile = block_gemm.LocalPrefetchA(BufferBase(0), 0);
                block_gemm.template LocalPrefetchB<0, kRn>(b_tile, BufferBase(0), 0);
            }
            // Tile t uses LDS buffer t % 2, so each tile's buffer is a compile-time constant and
            // its LDS offsets fold into immediates. TailNumber::Two runs pairs in buffers 0 and
            // 1; One runs the odd tile, tile 0, first and then pairs in buffers 1 and 0. Either
            // way nothing but the K tail follows a loop that may run no pair: an odd tile after
            // it, reached around the loop too, costs about 130 VGPRs (maf).
            const index_t pairs_end = TailNum == TailNumber::Full
                                          ? integer_divide_nonnegative<2>(full_tiles) * (2 * kTileK)
                                          : full_tiles * kTileK;
            if constexpr(TailNum == TailNumber::One)
            {
                k_tile<kKSteps, 0>(acc, a_tile, b_tile, 0, 0);
                for(index_t k0 = kTileK; k0 < pairs_end; k0 += 2 * kTileK)
                {
                    k_tile<kKSteps, 1>(acc, a_tile, b_tile, k0, 1);
                    k_tile<kKSteps, 0>(acc, a_tile, b_tile, k0 + kTileK, 0);
                }
            }
            else
            {
                for(index_t k0 = 0; k0 < pairs_end; k0 += 2 * kTileK)
                {
                    k_tile<kKSteps, 0>(acc, a_tile, b_tile, k0, 0);
                    k_tile<kKSteps, 1>(acc, a_tile, b_tile, k0 + kTileK, 1);
                }
            }
            if constexpr(TailNum == TailNumber::Full)
            {
                if(integer_modulo_nonnegative<2>(full_tiles))
                    k_tile<kKSteps, 0>(acc, a_tile, b_tile, pairs_end, 0);
            }
            if constexpr(Traits::kHasTail)
            {
                // The K tail is one more tile: its TDM loads are clipped at K and only its first
                // tail_steps K steps are read. With TailNumber::Full the GL2 prefetch takes its
                // parity as even, which only changes which lines are hinted.
                static_assert(Traits::kTailSteps < kKSteps,
                              "comp_tdm_scheduled: tail steps < k_steps");
                constexpr index_t kTailParity = TailNum == TailNumber::One ? 1 : 0;
                k_tile<Traits::kTailSteps, kTailParity>(
                    acc,
                    a_tile,
                    b_tile,
                    full_tiles * kTileK,
                    TailNum == TailNumber::Full ? integer_modulo_nonnegative<2>(full_tiles)
                                                : kTailParity);
            }
            // No workgroup may exit while a TDM load into its LDS is in flight: with multicast,
            // peers' loads write here too.
            s_wait_tensorcnt<0>();
            if constexpr(kHasCluster)
                cluster_barrier();
        }
    };

    /// The workgroup's tile of the M x N x K problem: A and B are the (M, K) and (N, K) windows
    /// of their tiles, the scales (M, K / 128) and (N, K / 128) windows of K-major packs. The TDM
    /// loads read the operands from the windows' buffers and clip at the unpadded extents; the
    /// tile origin is the A and B windows'. num_loop is the K tile count, the K tail included;
    /// the tail's K steps come from the shape traits. `p_smem` is GetSmemSize() bytes of LDS.
    template <typename AsDramBlockWindowTmp,
              typename AElementFunction,
              typename BsDramBlockWindowTmp,
              typename BElementFunction,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction&,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction&,
                                   const ScaleADramBlockWindowTmp& scale_a_window,
                                   const ScaleBDramBlockWindowTmp& scale_b_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        static_assert(std::is_same_v<AElementFunction, element_wise::PassThrough> &&
                          std::is_same_v<BElementFunction, element_wise::PassThrough>,
                      "comp_tdm_scheduled reads A and B as stored");
        static_assert(std::tuple_size_v<AsDramBlockWindowTmp> == 1 &&
                          std::tuple_size_v<BsDramBlockWindowTmp> == 1 &&
                          std::tuple_size_v<ScaleADramBlockWindowTmp> == 1 &&
                          std::tuple_size_v<ScaleBDramBlockWindowTmp> == 1,
                      "comp_tdm_scheduled runs single A and B tensors");
        CBlockTile c_block_tile;
#if defined(__gfx125__)
        using ADramWindow      = remove_cvref_t<decltype(a_dram_block_window_tmp[number<0>{}])>;
        using BDramWindow      = remove_cvref_t<decltype(b_dram_block_window_tmp[number<0>{}])>;
        using AScaleDramWindow = remove_cvref_t<decltype(scale_a_window[number<0>{}])>;
        using BScaleDramWindow = remove_cvref_t<decltype(scale_b_window[number<0>{}])>;
        PipelineImpl<ADramWindow, BDramWindow, AScaleDramWindow, BScaleDramWindow> impl(
            a_dram_block_window_tmp[number<0>{}],
            b_dram_block_window_tmp[number<0>{}],
            scale_a_window[number<0>{}],
            scale_b_window[number<0>{}],
            c_style_pointer_cast<CK_TILE_LDS_ADDR uint8_t*>(p_smem));
        AccTile acc;
        impl.template operator()<Problem::TailNum>(acc, Traits::kHasTail ? num_loop - 1 : num_loop);
        c_block_tile = BlockGemm::ToCBlockTile(acc);
#else
        ignore = a_dram_block_window_tmp;
        ignore = b_dram_block_window_tmp;
        ignore = scale_a_window;
        ignore = scale_b_window;
        ignore = num_loop;
        ignore = p_smem;
#endif
        return c_block_tile;
    }
};

} // namespace ck_tile
