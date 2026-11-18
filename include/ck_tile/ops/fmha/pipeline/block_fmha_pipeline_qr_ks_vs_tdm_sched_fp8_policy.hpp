// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_load.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"

namespace ck_tile {

// Tuning that also selects the segmented D256 LDS layout; other instances keep their type.
template <typename Tuning_>
struct FmhaTdmSchedFp8LdsSegTuning : Tuning_
{
};

template <typename Tuning>
inline constexpr bool kFmhaTdmSchedFp8IsLdsSegTuning = false;
template <typename Tuning>
inline constexpr bool kFmhaTdmSchedFp8IsLdsSegTuning<FmhaTdmSchedFp8LdsSegTuning<Tuning>> = true;

template <typename Geometry_,
          typename Tuning_,
          typename TileSchedule_ = FmhaTdmSchedFp8DefaultScheduleFor<Geometry_>>
struct BlockFmhaPipelineQRKSVSTdmSchedFp8Policy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Geometry   = Geometry_;
    using Tuning     = Tuning_;
    using Schedule   = TileSchedule_;
    using Layout     = FmhaTdmSchedFp8Layout<Geometry>;
    using KLoad      = FmhaTdmSchedFp8Load<Geometry::kKSuLoadCount, Geometry::kKLdsReadBytes>;
    using VLoad      = FmhaTdmSchedFp8TransposeLoad<Geometry::kVStageLoadCount>;
    static constexpr index_t kBlockPerCu = Tuning::kBlockPerCu;

    static constexpr index_t kLdsRows         = Geometry::kN;
    static constexpr index_t kKValidWidth     = Geometry::kHeadDimQK;
    static constexpr index_t kKPhysicalStride = Layout::kKPhysicalStride;
    static constexpr index_t kVLogicalWidth   = Geometry::kDv;
    static constexpr index_t kVPhysicalStride = Layout::kVPhysicalStride;

    using KPadding =
        std::conditional_t<Layout::kKPadByTdm,
                           detail::LdsPaddingConfig<true, kKValidWidth, Layout::kRowPadBytes>,
                           detail::LdsPaddingConfig<false, 0, 0>>;
    // Width of the K TDM box: the row, or the padded row whose tail reads zeros past hdim_q.
    static constexpr index_t kKDramBoxWidth = Layout::kKPadByTdm ? kKValidWidth : kKPhysicalStride;
    // D128/D256 Q rows take the K row padding so per-lane row reads spread across LDS banks.
    static constexpr bool kPadQRows = kKValidWidth >= 128 && Layout::kKPadByTdm;
    using VPadding = detail::LdsPaddingConfig<true, kVLogicalWidth, Layout::kRowPadBytes>;
    static_assert(sizeof(typename Geometry::KDataType) == 1 &&
                      sizeof(typename Geometry::VDataType) == 1 &&
                      kKPhysicalStride == kKValidWidth + Layout::kRowPadBytes &&
                      kVPhysicalStride == kVLogicalWidth + Layout::kRowPadBytes,
                  "TDM row padding must match the LDS read stride");
    static constexpr index_t kKFootprintElements = kLdsRows * kKPhysicalStride;
    static constexpr index_t kVFootprintElements = kLdsRows * kVPhysicalStride;
    static constexpr index_t kKFootprintBytes =
        kKFootprintElements * sizeof(typename Geometry::KDataType);
    static constexpr index_t kVFootprintBytes =
        kVFootprintElements * sizeof(typename Geometry::VDataType);
    // The output rescale is fused into softmax part 2; taking the early slot skips the post pass.
    static constexpr bool kEarlyOutputRescale      = true;
    static constexpr bool kLeadFirstLoad           = false;
    static constexpr index_t kKPrefetchTensorCount = Tuning::kKWait;
    static constexpr index_t kVPrefetchTensorCount = Tuning::kVWait;
    static constexpr bool kPrefetchTailDrain       = Tuning::kTailDrain;
    static_assert(kKPrefetchTensorCount == 0 || kKPrefetchTensorCount == 1);
    static_assert(kVPrefetchTensorCount == 0 || kVPrefetchTensorCount == 1);

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQkTailDsCount()
    {
        return Tuning::template GetQkTailDsCount<Stage>();
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetPvTailDsCount()
    {
        return Tuning::template GetPvTailDsCount<Stage>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitQkStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kQkStages);
        s_wait_dscnt<kFmhaTdmSchedFp8DsWait<GetQkTailDsCount<Stage>()>>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitPvStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        s_wait_dscnt<kFmhaTdmSchedFp8DsWait<GetPvTailDsCount<Stage>()>>();
    }

    // FP8 Q is staged in the V1 buffer, which D192 widens to its unpadded rows.
    static constexpr index_t kQFootprintBytes =
        Geometry::kM * (kPadQRows ? kKPhysicalStride : kKValidWidth);
    static constexpr index_t kV1FootprintBytes = max(kVFootprintBytes, kQFootprintBytes);
    // Segmented: K/V lower halves packed from 0 (Q fills the gap), upper halves a segment away.
    static constexpr bool kLdsSegLayout   = kFmhaTdmSchedFp8IsLdsSegTuning<Tuning_>;
    static constexpr index_t kLdsSegHalf  = kLdsRows / 2 * kKPhysicalStride;
    static constexpr index_t kLdsSegShift = kLdsSegLayout ? 4 * kLdsSegHalf : 0;
    static_assert(!kLdsSegLayout ||
                  (Geometry::kHeadDimQK == 256 && Geometry::kWaves == 4 &&
                   kKPhysicalStride == kVPhysicalStride && kQFootprintBytes == 2 * kLdsSegHalf &&
                   kLdsSegShift + kLdsSegHalf >= 65536));
    static constexpr index_t kLdsOffsetK0 = 0;
    static constexpr index_t kLdsOffsetK1 =
        kLdsSegLayout ? kLdsSegHalf : kLdsOffsetK0 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV0 =
        kLdsSegLayout ? 2 * kLdsSegHalf : kLdsOffsetK1 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV1 =
        kLdsSegLayout ? 3 * kLdsSegHalf : kLdsOffsetV0 + kVFootprintBytes;
    static constexpr index_t kLdsArenaSize =
        kLdsSegLayout ? 9 * kLdsSegHalf : kLdsOffsetV1 + kV1FootprintBytes;

    // fp32 S/O fragments are input-type independent, so the 16-bit softmax/output maps apply.
    using OutputFragments = FmhaTdmSchedOutputFragmentsFor<Geometry::kPvNIter / 2>;

    static_assert(kFmhaTdmSchedFp8GeometryOk<Geometry>);
    using ScoreSoftmax = FmhaTdmSchedSplitSoftmaxFor<Geometry::kQkStages>;
    using ScoreMapping = typename ScoreSoftmax::Mapping;
    static_assert(ScoreMapping::kNumMIter == Geometry::kQkMIter &&
                      ScoreMapping::kNumNIter == Geometry::kN / Geometry::kQkWmmaN &&
                      ScoreMapping::kNumSu == Geometry::kQkStages &&
                      ScoreMapping::kNIterPerSu == Geometry::kQkSuNIter &&
                      ScoreMapping::kNumMsb == Geometry::kQkMIter * Geometry::kQkSuNIter &&
                      ScoreMapping::kElementsPerWmma ==
                          Geometry::QkWmma::kCM0PerLane * Geometry::QkWmma::kCM1PerLane,
                  "split-softmax score mapping does not match the QK geometry");
    static_assert(OutputFragments::kNumFragments == Geometry::kPvMIter * Geometry::kPvNIter &&
                      OutputFragments::kNumDmsb % Geometry::kPvMIter == 0 &&
                      OutputFragments::kElementsPerFragment ==
                          Geometry::PvWmma::kCM0PerLane * Geometry::PvWmma::kCM1PerLane,
                  "output fragments do not match the PV geometry");
    // One PV stage covers the KV tile, so every PV row is an output fragment.
    static_assert(Geometry::kPvWmmasPerStage == OutputFragments::kNumFragments);

    static_assert(kLdsOffsetK0 % 16 == 0 && kLdsOffsetK1 % 16 == 0 && kLdsOffsetV0 % 16 == 0 &&
                  kLdsOffsetV1 % 16 == 0);

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemmSu()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Geometry::kQkSuColumns,
                                                    Problem::BlockFmhaShape::kQKHeaddim>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::QDataType,
                                            typename Problem::KDataType,
                                            typename Problem::SaccDataType,
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                            true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::SaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeKSuRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemmSu<Problem>())>;
        static_assert(BlockGemm::NIterPerWarp == Geometry::kQkSuNIter &&
                      BlockGemm::KIterPerWarp == Geometry::kQkKIter);
        return make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
    }

    // Marks a GEMM whose WMMA scale operands stay neutral.
    struct NoScale
    {
    };

    // Marks a softmax whose QK stages share one scale.
    struct NoColumnScale
    {
    };

    // K64 has no scaled WMMA: zero-pad to K128, whose low eight dwords keep the K64 lane order.
    template <typename CWarpTensor, typename AWarpTensor, typename BWarpTensor>
    CK_TILE_DEVICE static void RunPaddedScaledWmma(CWarpTensor& c_warp_tensor,
                                                   const AWarpTensor& a_warp_tensor,
                                                   const BWarpTensor& b_warp_tensor,
                                                   int32_t a_scale,
                                                   int32_t b_scale)
    {
        using QDataType     = typename Geometry::QDataType;
        using PaddedGemm    = WarpGemmDispatcher<QDataType, QDataType, float, 16, 16, 128, true>;
        using Half          = int32_t __attribute__((ext_vector_type(8)));
        constexpr auto zero = number<0>{};
        constexpr auto one  = number<1>{};
        static_assert(sizeof(typename AWarpTensor::DataType) == 1 &&
                      AWarpTensor::get_thread_buffer_size() == sizeof(Half) &&
                      BWarpTensor::get_thread_buffer_size() == sizeof(Half));

        typename PaddedGemm::AWarpTensor a;
        typename PaddedGemm::BWarpTensor b;
        typename PaddedGemm::CWarpTensor c;
        a.get_thread_buffer().template set_as<Half>(
            zero, bit_cast<Half>(a_warp_tensor.get_thread_buffer()));
        a.get_thread_buffer().template set_as<Half>(one, Half{});
        b.get_thread_buffer().template set_as<Half>(
            zero, bit_cast<Half>(b_warp_tensor.get_thread_buffer()));
        b.get_thread_buffer().template set_as<Half>(one, Half{});
        c.get_thread_buffer() = c_warp_tensor.get_thread_buffer();
        PaddedGemm{}(c, a, b, a_scale, b_scale);
        c_warp_tensor.get_thread_buffer() = c.get_thread_buffer();
    }

    template <index_t WmmaOrdinal,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename AScaleFn = NoScale,
              typename BScaleFn = NoScale>
    CK_TILE_DEVICE static void RunQkSuWmma(CBlockTensor& c_block_tensor,
                                           const ABlockTensor& a_block_tensor,
                                           const BBlockTensor& b_block_tensor,
                                           const AScaleFn& a_scale = {},
                                           const BScaleFn& b_scale = {})
    {
        using WarpGemm    = typename BlockGemm::WarpGemm;
        using AWarpDstr   = typename WarpGemm::AWarpDstr;
        using BWarpDstr   = typename WarpGemm::BWarpDstr;
        using CWarpDstr   = typename WarpGemm::CWarpDstr;
        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_assert(WmmaOrdinal >= 0 && WmmaOrdinal < BlockGemm::NIterPerWarp *
                                                            BlockGemm::KIterPerWarp *
                                                            BlockGemm::MIterPerWarp);
        constexpr index_t n_iter =
            WmmaOrdinal / (BlockGemm::KIterPerWarp * BlockGemm::MIterPerWarp);
        constexpr index_t k_iter =
            (WmmaOrdinal / BlockGemm::MIterPerWarp) % BlockGemm::KIterPerWarp;
        constexpr index_t m_iter = WmmaOrdinal % BlockGemm::MIterPerWarp;

        AWarpTensor a_warp_tensor;
        a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, k_iter>{}, a_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

        BWarpTensor b_warp_tensor;
        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<n_iter, k_iter>{}, b_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

        CWarpTensor c_warp_tensor;
        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

        if constexpr(std::is_same_v<AScaleFn, NoScale>)
            WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        else if constexpr(WarpGemm::kK == 128)
            WarpGemm{}(c_warp_tensor,
                       a_warp_tensor,
                       b_warp_tensor,
                       a_scale(number<m_iter>{}, number<k_iter>{}),
                       b_scale(number<n_iter>{}, number<k_iter>{}));
        else
            RunPaddedScaledWmma(c_warp_tensor,
                                a_warp_tensor,
                                b_warp_tensor,
                                a_scale(number<m_iter>{}, number<k_iter>{}),
                                b_scale(number<n_iter>{}, number<k_iter>{}));
        c_block_tensor.set_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    }

    template <FmhaTdmSchedSchedulePoint Point>
    CK_TILE_DEVICE static void EmitSchedulePoint()
    {
        if constexpr(Point == FmhaTdmSchedSchedulePoint::AfterTokens)
            __builtin_amdgcn_sched_barrier(0);
        else
            __builtin_amdgcn_sched_barrier(LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::TRANS);
    }

    // Fence emitter of the placement executor: (stage, wmma, point).
    struct SchedulePointEmitter
    {
        template <typename Stage, typename Wmma, typename Point>
        CK_TILE_DEVICE void operator()(Stage, Wmma, Point) const
        {
            EmitSchedulePoint<Point::value>();
        }
    };

    // QK stage WMMAs interleaved with reads of the next K group (last stage: the PV V tile).
    template <index_t Stage,
              typename Mode,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename AScaleFn = NoScale,
              typename BScaleFn = NoScale>
    CK_TILE_DEVICE static void RunQkScheduledStage(const BlockGemm&,
                                                   CBlockTensor& c_block_tensor,
                                                   const ABlockTensor& a_block_tensor,
                                                   const BBlockTensor& b_block_tensor,
                                                   NextBBlockTensor& next_b_block_tensor,
                                                   const BTileWindow& b_lds_window,
                                                   const AScaleFn& a_scale = {},
                                                   const BScaleFn& b_scale = {})
    {
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumQkStages);
        static_assert((Stage < SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount) ||
                      (Stage == SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount));

        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
            {
                static_assert(Stage < Geometry::kQkStages - 1);
                KLoad::template LoadInstruction<decltype(access)::value>(next_b_block_tensor,
                                                                         b_lds_window);
            }
            else
            {
                static_assert(Stage == Geometry::kQkStages - 1);
                VLoad::template LoadAccess<decltype(access)::value>(next_b_block_tensor,
                                                                    b_lds_window);
            }
        };
        SchedulePointEmitter emit_point;
        auto emit_wmma = [&](auto, auto wmma) {
            RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
                c_block_tensor, a_block_tensor, b_block_tensor, a_scale, b_scale);
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        WaitQkStageTail<Stage>();
    }

    // Marks a lead read that has no group after it.
    struct NoLeadGroup
    {
    };

    // QK stage reading the next group's second half, then the first half of the group after.
    template <index_t Stage,
              typename Mode,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename LeadBBlockTensor,
              typename LeadTileWindow,
              typename AScaleFn = NoScale,
              typename BScaleFn = NoScale>
    CK_TILE_DEVICE static void RunQkScheduledLeadStage(const BlockGemm&,
                                                       CBlockTensor& c_block_tensor,
                                                       const ABlockTensor& a_block_tensor,
                                                       const BBlockTensor& b_block_tensor,
                                                       NextBBlockTensor& next_b_block_tensor,
                                                       const BTileWindow& b_lds_window,
                                                       LeadBBlockTensor& lead_b_block_tensor,
                                                       const LeadTileWindow& lead_lds_window,
                                                       const AScaleFn& a_scale = {},
                                                       const BScaleFn& b_scale = {})
    {
        using SelectedSchedule  = typename Mode::SelectedPolicy::Schedule;
        using Executor          = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind              = FmhaTdmSchedLoadKind;
        constexpr index_t kHalf = Geometry::kKSuLoadCount / 2;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumQkStages - 1 &&
                      BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount);

        auto emit_token = [&](auto kind, auto access) {
            static_assert(decltype(kind)::value == Kind::KRead);
            constexpr index_t a = decltype(access)::value;
            if constexpr(a < kHalf)
                KLoad::template LoadInstruction<a + kHalf>(next_b_block_tensor, b_lds_window);
            else if constexpr(!std::is_same_v<LeadTileWindow, NoLeadGroup>)
                KLoad::template LoadInstruction<a - kHalf>(lead_b_block_tensor, lead_lds_window);
        };
        SchedulePointEmitter emit_point;
        auto emit_wmma = [&](auto, auto wmma) {
            RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
                c_block_tensor, a_block_tensor, b_block_tensor, a_scale, b_scale);
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        WaitQkStageTail<Stage>();
    }

    // Marks a PV stage without streamed V reads (all but D256).
    struct NoStreamedV
    {
    };

    // PV stage WMMAs interleaved with next-K group 0 reads and, at D256, the streamed V chunks.
    template <index_t Stage,
              typename Mode,
              bool KDwordStore,
              typename NextKTensor,
              typename KTileWindow,
              typename VTensor,
              typename VTileWindow,
              typename WmmaEmitter>
    CK_TILE_DEVICE static void RunPvStageWithWmma(NextKTensor& next_k_tensor,
                                                  const KTileWindow& k_lds_window,
                                                  VTensor& v_tensor,
                                                  const VTileWindow& v_lds_window,
                                                  WmmaEmitter& emit_wmma)
    {
        using SelectedSchedule   = typename Mode::SelectedPolicy::Schedule;
        using Executor           = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind               = FmhaTdmSchedLoadKind;
        constexpr bool kStreamsV = !std::is_same_v<VTileWindow, NoStreamedV>;
        static_assert(Stage == SelectedSchedule::kNumPvStages - 1 &&
                      KTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount);
        if constexpr(kStreamsV)
            static_assert(VTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount);
        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
                KLoad::template LoadInstruction<decltype(access)::value, KDwordStore>(next_k_tensor,
                                                                                      k_lds_window);
            else
            {
                static_assert(kStreamsV && decltype(access)::value >= 0, "V reads need a V window");
                VLoad::template LoadAccess<decltype(access)::value>(v_tensor, v_lds_window);
            }
        };
        SchedulePointEmitter emit_point;
        // The output rescale is fused into softmax, so PV rows emit none.
        auto no_rescale = [](auto) {};
        Executor::template ExecutePvStage<Stage>(emit_wmma, emit_token, emit_point, no_rescale);
        WaitPvStageTail<Stage>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCountdownLoop()
    {
        return !Problem::FmhaMask::IsMasking && !Problem::kHasSink &&
               Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool DeferTensorReady()
    {
        return !Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{}) == 1;
    }

    // Max of N values by max3 levels; a two-value remainder carries to the next level.
    template <index_t N>
    CK_TILE_DEVICE static float TreeMax(const array<float, N>& v)
    {
        if constexpr(N == 1)
            return v[number<0>{}];
        else if constexpr(N == 2)
            return max(v[number<0>{}], v[number<1>{}]);
        else
        {
            constexpr index_t kTriples = N / 3;
            constexpr index_t kRest    = N % 3;
            array<float, kTriples + kRest> next{};
            static_for<0, kTriples, 1>{}([&](auto i) {
                constexpr index_t k = 3 * decltype(i)::value;
                next(i) = max(max(v[number<k>{}], v[number<k + 1>{}]), v[number<k + 2>{}]);
            });
            static_for<0, kRest, 1>{}([&](auto i) {
                constexpr index_t k          = decltype(i)::value;
                next(number<kTriples + k>{}) = v[number<3 * kTriples + k>{}];
            });
            return TreeMax<kTriples + kRest>(next);
        }
    }

    // Max of N values by Chains independent max3 chains, then a tree over the chain results.
    template <index_t Chains, index_t N>
    CK_TILE_DEVICE static float ChainedMax(const array<float, N>& v)
    {
        constexpr index_t kPerChain = (N + Chains - 1) / Chains;
        array<float, Chains> acc{};
        static_for<0, Chains, 1>{}([&](auto c) {
            constexpr index_t kBegin = decltype(c)::value * kPerChain;
            constexpr index_t kEnd   = min(kBegin + kPerChain, N);
            acc(c)                   = v[number<kBegin>{}];
            static_for<0, (kEnd - kBegin) / 2, 1>{}([&](auto i) {
                constexpr index_t k = kBegin + 1 + 2 * decltype(i)::value;
                if constexpr(k + 1 < kEnd)
                    acc(c) = max(max(acc[c], v[number<k>{}]), v[number<k + 1>{}]);
                else
                    acc(c) = max(acc[c], v[number<k>{}]);
            });
        });
        return TreeMax(acc);
    }

    // Copies the score pairs of row half Half, QK stages [SuBegin, SuEnd), into values from First.
    template <index_t Half,
              index_t SuBegin,
              index_t SuEnd,
              index_t First,
              typename ScoreTensor,
              index_t N>
    CK_TILE_DEVICE static void GatherRowPairs(const ScoreTensor& score, array<float, N>& values)
    {
        using Mapping = typename ScoreSoftmax::Mapping;
        static_assert(First + (SuEnd - SuBegin) * 4 * Mapping::kPairsPerFragment <= N);
        static_for<SuBegin, SuEnd, 1>{}([&](auto su) {
            static_for<0, 2, 1>{}([&](auto h) {
                static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto p) {
                    constexpr index_t at =
                        First + (((su - SuBegin) * 2 + h) * Mapping::kPairsPerFragment + p) * 2;
                    const auto x =
                        Mapping::template LoadPair<Half * 2 + h,
                                                   su * Mapping::kPairsPerFragment + p>(score);
                    values(number<at>{})     = x[0];
                    values(number<at + 1>{}) = x[1];
                });
            });
        });
    }

    // Log2 softmax scale of row band Half: one per workgroup, or one per lane row band.
    template <index_t Half, typename LogScale>
    CK_TILE_DEVICE static float BandScale(const LogScale& log2e_scale)
    {
        if constexpr(std::is_same_v<LogScale, float>)
            return log2e_scale;
        else
            return log2e_scale[number<Half>{}];
    }

    template <typename Problem,
              typename ScoreTensor,
              typename RowTensor,
              typename LogScale,
              typename ColScale = NoColumnScale>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(ScoreTensor& score,
                                                     RowTensor& row_max,
                                                     LogScale log2e_scale,
                                                     float& delta_m0,
                                                     float& delta_m1,
                                                     const ColScale& col_scale = {})
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum != BlockAttentionBiasEnum::ALIBI);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        const float old_max_m0 = row_max.get_thread_buffer()[number<0>{}];
        const float old_max_m1 = row_max.get_thread_buffer()[number<1>{}];

        // An elementwise bias may carry -inf, like a mask.
        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking ||
                                      Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;
        // One max3 tree per row over both MSBs and the old max, then a single lane exchange.
        auto row_part = [&](auto half, float old_max) {
            array<float, 2 * Mapping::kElementsPerMsb + 1> values{};
            values(number<0>{}) = old_max;
            if constexpr(std::is_same_v<ColScale, NoColumnScale>)
                GatherRowPairs<decltype(half)::value, 0, Mapping::kNumSu, 1>(score, values);
            // D64 blockscale sink tiles stay at three workgroups per CU with fewer max partials.
            constexpr bool kChainedRowMax =
                Problem::BlockFmhaShape::kQKHeaddim == 64 && Problem::kHasSink &&
                !Problem::FmhaMask::IsMasking &&
                Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                Problem::QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE;
            const float lane_max = [&]() {
                if constexpr(!std::is_same_v<ColScale, NoColumnScale>)
                {
                    // Positive column scales commute with each stage's max.
                    array<float, Mapping::kNumSu + 1> scaled{};
                    scaled(number<0>{}) = old_max;
                    static_for<0, Mapping::kNumSu, 1>{}([&](auto su) {
                        array<float, 4 * Mapping::kPairsPerFragment> stage{};
                        GatherRowPairs<decltype(half)::value, su, su + 1, 0>(score, stage);
                        scaled(number<su + 1>{}) = TreeMax(stage) * col_scale[su];
                    });
                    return TreeMax(scaled);
                }
                else if constexpr(kChainedRowMax)
                    return ChainedMax<2>(values);
                else
                    return TreeMax(values);
            }();
            const float scale = BandScale<decltype(half)::value>(log2e_scale);
            return Softmax::template RunPart1<kValidateMax>(
                lane_max, Softmax::PermuteLaneX16(lane_max), old_max * scale, scale);
        };
        const auto p1_m0 = row_part(number<0>{}, old_max_m0);
        const auto p1_m1 = row_part(number<1>{}, old_max_m1);

        row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
        row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
        delta_m0                                 = p1_m0.delta;
        delta_m1                                 = p1_m1.delta;
    }

    // Lane max of QK stage Su's columns per row half, folded into part when Accumulate.
    template <index_t Su, bool Accumulate, typename ScoreTensor>
    CK_TILE_DEVICE static void StageLaneMax(const ScoreTensor& score, array<float, 2>& part)
    {
        using Mapping = typename ScoreSoftmax::Mapping;
        static_for<0, 2, 1>{}([&](auto half) {
            constexpr index_t kFirst = Accumulate ? 1 : 0;
            array<float, kFirst + 4 * Mapping::kPairsPerFragment> values{};
            if constexpr(Accumulate)
                values(number<0>{}) = part[half];
            GatherRowPairs<decltype(half)::value, Su, Su + 1, kFirst>(score, values);
            part(half) = TreeMax(values);
        });
    }

    // RunSplitSoftmaxPart01 for scores whose earlier QK stages are already in part.
    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01FromPartial(ScoreTensor& score,
                                                                RowTensor& row_max,
                                                                float log2e_scale,
                                                                float& delta_m0,
                                                                float& delta_m1,
                                                                const array<float, 2>& part)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;
        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                      !Problem::FmhaMask::IsMasking && !Problem::kHasLogitsSoftCap);
        static_assert(RowTensor::get_thread_buffer_size() == 2);
        constexpr index_t kSu = Mapping::kNumSu - 1;
        auto row_part         = [&](auto half, float old_max) {
            array<float, 2 + 4 * Mapping::kPairsPerFragment> values{};
            values(number<0>{}) = old_max;
            values(number<1>{}) = part[half];
            GatherRowPairs<decltype(half)::value, kSu, kSu + 1, 2>(score, values);
            const float lane_max = TreeMax(values);
            return Softmax::template RunPart1<false>(
                lane_max, Softmax::PermuteLaneX16(lane_max), old_max * log2e_scale, log2e_scale);
        };
        const auto p1_m0 = row_part(number<0>{}, row_max.get_thread_buffer()[number<0>{}]);
        const auto p1_m1 = row_part(number<1>{}, row_max.get_thread_buffer()[number<1>{}]);
        row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
        row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
        delta_m0                                 = p1_m0.delta;
        delta_m1                                 = p1_m1.delta;
    }

    // Row-sum add tree per MSB: nodes 0-7 add leaf pairs, 8-14 add nodes, 15 folds the pair.
    struct ExpInterleavePlan
    {
        static constexpr index_t kPairsPerMsb = 16;
        static constexpr index_t kNodesPerMsb = 16;
        static constexpr index_t kNumMsb      = 4;
        static constexpr index_t kPairs       = kPairsPerMsb * kNumMsb;
        static constexpr index_t kNodes       = kNodesPerMsb * kNumMsb;
        static constexpr index_t kLag         = 2;
        // Tree levels of one MSB: 8 leaf adds, then 4, 2 and the root, then the fold node.
        static constexpr index_t kLevel1   = kPairsPerMsb / 2;
        static constexpr index_t kLevel2   = kLevel1 + kLevel1 / 2;
        static constexpr index_t kRootNode = kLevel2 + kLevel1 / 4;
        static constexpr index_t kFoldNode = kRootNode + 1;
        static_assert(kFoldNode + 1 == kNodesPerMsb);

        index_t slot_node[kPairs] = {};
        index_t tail[kNodes]      = {};
        index_t num_tail          = 0;

        // First input of node k; adds read inputs a and a + 1, the fold reads node 14.
        static constexpr index_t InputA(index_t k)
        {
            if(k < kLevel1)
                return 2 * k;
            if(k < kLevel2)
                return 2 * (k - kLevel1);
            if(k < kRootNode)
                return kLevel1 + 2 * (k - kLevel2);
            return k == kRootNode ? kLevel2 : kRootNode;
        }
        static constexpr bool InputIsLeaf(index_t k) { return k < kLevel1; }

        // Greedily give each slot the first add whose inputs are kLag slots old.
        static constexpr ExpInterleavePlan Make()
        {
            ExpInterleavePlan plan{};
            index_t done[kNodes] = {};
            for(index_t n = 0; n < kNodes; ++n)
                done[n] = -1;
            auto input_slot = [&](index_t n, index_t which) {
                const index_t msb = n / kNodesPerMsb, k = n % kNodesPerMsb;
                const index_t a = InputA(k) + (k < kFoldNode ? which : 0);
                return InputIsLeaf(k) ? msb * kPairsPerMsb + a : done[msb * kNodesPerMsb + a];
            };
            auto ready_by = [&](index_t n, index_t slot) {
                const index_t x = input_slot(n, 0), y = input_slot(n, 1);
                return done[n] < 0 && x >= 0 && y >= 0 && x + kLag <= slot && y + kLag <= slot;
            };
            for(index_t q = 0; q < kPairs; ++q)
            {
                plan.slot_node[q] = -1;
                for(index_t n = 0; n < kNodes; ++n)
                    if(ready_by(n, q))
                    {
                        plan.slot_node[q] = n;
                        done[n]           = q;
                        break;
                    }
            }
            for(index_t n = 0; n < kNodes; ++n)
                if(done[n] < 0)
                {
                    plan.tail[plan.num_tail++] = n;
                    done[n]                    = kPairs;
                }
            return plan;
        }
    };

    // Fused output rescale: each slot issues exp, fma, exp, in-place mul, add, so no exp pairs up.
    template <bool SumRows,
              typename ScoreTensor,
              typename Fragments,
              typename LogScale,
              typename ColScale = NoColumnScale>
    CK_TILE_DEVICE static auto RunInterleavedLocalSums(ScoreTensor& score,
                                                       Fragments& output,
                                                       float exponent_max_m0,
                                                       float exponent_max_m1,
                                                       float delta_m0,
                                                       float delta_m1,
                                                       LogScale log2e_scale,
                                                       float& output_scale_m0,
                                                       float& output_scale_m1,
                                                       const ColScale& col_scale = {})
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;
        using Pair    = typename Softmax::Pair;
        using Plan    = ExpInterleavePlan;
        static_assert(Mapping::kNumMsb == Plan::kNumMsb &&
                      Mapping::kPairsPerMsb == Plan::kPairsPerMsb);
        constexpr index_t kOutPairsPerFragment = OutputFragments::kElementsPerFragment / 2;
        // Output vs score pairs: D64 rescales one per two slots, D256 two per slot.
        constexpr index_t kOutPairs = OutputFragments::kNumFragments * kOutPairsPerFragment;
        constexpr index_t kSlotsPerOutPair =
            kOutPairs < Plan::kPairs ? Plan::kPairs / kOutPairs : 1;
        constexpr index_t kOutPairsPerSlot =
            kOutPairs > Plan::kPairs ? kOutPairs / Plan::kPairs : 1;
        static_assert(kOutPairs * kSlotsPerOutPair == Plan::kPairs * kOutPairsPerSlot);
        constexpr Plan plan = Plan::Make();
        // SALU, WMMA, VMEM and LDS may cross; VALU and TRANS keep the source order.
        constexpr int32_t kValuFence = LLVMSchedGroupMask::SALU | LLVMSchedGroupMask::MFMA |
                                       LLVMSchedGroupMask::VMEM | LLVMSchedGroupMask::VMEM_READ |
                                       LLVMSchedGroupMask::VMEM_WRITE | LLVMSchedGroupMask::DS |
                                       LLVMSchedGroupMask::DS_READ | LLVMSchedGroupMask::DS_WRITE;

        array<Pair, Plan::kPairs> leaf{};
        array<Pair, Plan::kNodes> node{};
        array<float, Plan::kNumMsb> sums{};

        auto make_delta = [&](auto q) {
            constexpr index_t msb  = decltype(q)::value / Plan::kPairsPerMsb;
            constexpr index_t pair = decltype(q)::value % Plan::kPairsPerMsb;
            const float row_max    = msb < 2 ? exponent_max_m0 : exponent_max_m1;
            const float scale      = BandScale<(msb < 2 ? 0 : 1)>(log2e_scale);
            const float col        = [&]() {
                if constexpr(std::is_same_v<ColScale, NoColumnScale>)
                    return scale;
                else
                    return scale * col_scale[number<pair / Mapping::kPairsPerFragment>{}];
            }();
            leaf(q) = __builtin_elementwise_fma(Mapping::template LoadPair<msb, pair>(score),
                                                Softmax::Broadcast(col),
                                                -Softmax::Broadcast(row_max * scale));
        };
        auto exp_element = [&](auto q, auto e) {
            leaf(q)[decltype(e)::value] = Softmax::Exp2(leaf[q][decltype(e)::value]);
            if constexpr(decltype(e)::value == 1)
                Mapping::template StorePair<decltype(q)::value / Plan::kPairsPerMsb,
                                            decltype(q)::value % Plan::kPairsPerMsb>(score,
                                                                                     leaf[q]);
        };
        auto rescale_pair = [&](auto q) {
            constexpr index_t ordinal = decltype(q)::value / kOutPairsPerFragment;
            constexpr index_t e       = decltype(q)::value % kOutPairsPerFragment * 2;
            const float scale =
                ordinal / OutputFragments::kNumN < 2 ? output_scale_m0 : output_scale_m1;
            auto& fragment  = output.at(number<ordinal>{});
            const Pair v    = Pair{fragment[e], fragment[e + 1]} * Softmax::Broadcast(scale);
            fragment[e]     = v[0];
            fragment[e + 1] = v[1];
        };
        auto add_node = [&](auto n) {
            constexpr index_t msb  = decltype(n)::value / Plan::kNodesPerMsb;
            constexpr index_t k    = decltype(n)::value % Plan::kNodesPerMsb;
            constexpr index_t a    = Plan::InputA(k);
            constexpr index_t base = msb * Plan::kNodesPerMsb;
            if constexpr(k == Plan::kFoldNode)
                sums(number<msb>{}) =
                    Softmax::HorizontalAdd(node[number<base + Plan::kRootNode>{}]);
            else if constexpr(Plan::InputIsLeaf(k))
                node(n) = Softmax::AddPair(leaf[number<msb * Plan::kPairsPerMsb + a>{}],
                                           leaf[number<msb * Plan::kPairsPerMsb + a + 1>{}]);
            else
                node(n) = Softmax::AddPair(node[number<base + a>{}], node[number<base + a + 1>{}]);
        };

        output_scale_m0 = Softmax::Exp2(delta_m0);
        __builtin_amdgcn_sched_barrier(kValuFence);
        make_delta(number<0>{});
        __builtin_amdgcn_sched_barrier(kValuFence);
        output_scale_m1 = Softmax::Exp2(delta_m1);
        __builtin_amdgcn_sched_barrier(kValuFence);
        make_delta(number<1>{});
        __builtin_amdgcn_sched_barrier(kValuFence);
        static_for<0, Plan::kPairs, 1>{}([&](auto q) {
            exp_element(q, number<0>{});
            __builtin_amdgcn_sched_barrier(kValuFence);
            if constexpr(decltype(q)::value + 2 < Plan::kPairs)
            {
                make_delta(number<decltype(q)::value + 2>{});
                __builtin_amdgcn_sched_barrier(kValuFence);
            }
            exp_element(q, number<1>{});
            __builtin_amdgcn_sched_barrier(kValuFence);
            if constexpr(decltype(q)::value % kSlotsPerOutPair == 0)
            {
                static_for<0, kOutPairsPerSlot, 1>{}([&](auto j) {
                    rescale_pair(number<decltype(q)::value / kSlotsPerOutPair * kOutPairsPerSlot +
                                        decltype(j)::value>{});
                });
                __builtin_amdgcn_sched_barrier(kValuFence);
            }
            if constexpr(SumRows && plan.slot_node[decltype(q)::value] >= 0)
            {
                add_node(number<plan.slot_node[decltype(q)::value]>{});
                __builtin_amdgcn_sched_barrier(kValuFence);
            }
        });
        if constexpr(SumRows)
            static_for<0, plan.num_tail, 1>{}([&](auto i) { add_node(number<plan.tail[i]>{}); });
        return sums;
    }

    // SumRows false leaves row_sum to the caller, which sums the P it actually consumes.
    template <typename Problem,
              bool SumRows = true,
              typename ScoreTensor,
              typename RowTensor,
              typename Fragments,
              typename LogScale,
              typename ColScale = NoColumnScale>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart2AndGetScale(ScoreTensor& score,
                                                               const RowTensor& row_max,
                                                               RowTensor& row_sum,
                                                               LogScale log2e_scale,
                                                               float delta_m0,
                                                               float delta_m1,
                                                               float& output_scale_m0,
                                                               float& output_scale_m1,
                                                               Fragments& output,
                                                               const ColScale& col_scale = {})
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum != BlockAttentionBiasEnum::ALIBI);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        // An elementwise bias may carry -inf, like a mask.
        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking ||
                                      Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;

        const auto validated_max = [](float value) {
            if constexpr(kValidateMax)
                return value <= -numeric<float>::infinity() ? 0.0f : value;
            else
                return value;
        };

        const float row_max_m0                  = row_max.get_thread_buffer()[number<0>{}];
        const float row_max_m1                  = row_max.get_thread_buffer()[number<1>{}];
        [[maybe_unused]] const float old_sum_m0 = row_sum.get_thread_buffer()[number<0>{}];
        [[maybe_unused]] const float old_sum_m1 = row_sum.get_thread_buffer()[number<1>{}];

        const auto sums = RunInterleavedLocalSums<SumRows>(score,
                                                           output,
                                                           validated_max(row_max_m0),
                                                           validated_max(row_max_m1),
                                                           delta_m0,
                                                           delta_m1,
                                                           log2e_scale,
                                                           output_scale_m0,
                                                           output_scale_m1,
                                                           col_scale);

        if constexpr(SumRows)
        {
            row_sum.get_thread_buffer()[number<0>{}] =
                __builtin_fmaf(output_scale_m0,
                               old_sum_m0,
                               Softmax::MergeRowSum(sums[number<0>{}], sums[number<1>{}]));
            row_sum.get_thread_buffer()[number<1>{}] =
                __builtin_fmaf(output_scale_m1,
                               old_sum_m1,
                               Softmax::MergeRowSum(sums[number<2>{}], sums[number<3>{}]));
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsSupportedProblem()
    {
        using Shape = remove_cvref_t<typename Problem::BlockFmhaShape>;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        constexpr bool is_wave32 = get_warp_size() == Geometry::kWaveSize;
#else
        constexpr bool is_wave32 = true;
#endif
        return (Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                (Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS &&
                 Problem::QScaleEnum != BlockAttentionQuantScaleEnum::MX)) &&
               std::is_same_v<FmhaTdmSchedGeometryFor<Problem, Geometry::kQkSuColumns>, Geometry> &&
               Shape::Gemm1BlockWarps::at(number<0>{}) == Shape::Gemm0BlockWarps::at(number<0>{}) &&
               Shape::Gemm1BlockWarps::at(number<1>{}) == Shape::Gemm0BlockWarps::at(number<1>{}) &&
               Shape::Gemm1BlockWarps::at(number<2>{}) == Shape::Gemm0BlockWarps::at(number<2>{}) &&
               std::is_same_v<remove_cvref_t<typename Problem::SaccDataType>, float> &&
               std::is_same_v<remove_cvref_t<typename Problem::OaccDataType>, float> &&
               Shape::kK0 == Geometry::kHeadDimQK && Shape::kSubQKHeaddim == Geometry::kHeadDimQK &&
               Shape::IsVLayoutRowMajor && Shape::NumWarps == Geometry::kWaves && is_wave32;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK0() { return kLdsOffsetK0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetK1() { return kLdsOffsetK1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV0() { return kLdsOffsetV0; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsOffsetV1() { return kLdsOffsetV1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsArenaSize() { return kLdsArenaSize; }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_fp8_sched requires a full-head K TDM load");
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(kLdsRows % warp_num == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, kLdsRows / warp_num>, sequence<kKDramBoxWidth>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackK<Problem>();
        return make_naive_tensor_descriptor(make_tuple(number<kLdsRows>{}, number<kKValidWidth>{}),
                                            make_tuple(number<kKPhysicalStride>{}, number<1>{}),
                                            number<kKPack>{},
                                            number<1>{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        static_assert(LoadOnce, "qr_tdm_fp8_sched requires a full-head K LDS descriptor");
        return MakeKLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackV<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<kVLogicalWidth>{}),
            make_tuple(number<kVPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        static_assert(!Xor, "the V TDM writer requires an affine V LDS descriptor");
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        using BlockGemm = remove_cvref_t<decltype(BasePolicy::template GetPVBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t kMWarp        = Geometry::kWaves;
        constexpr index_t kNWarp        = 1;
        constexpr index_t kNIterPerWarp = Geometry::kPvNIter;
        constexpr index_t kKIterPerWarp = Geometry::kPvStageKIter;
        static_assert(kNIterPerWarp == kVLogicalWidth / (kNWarp * WarpGemm::kN) &&
                      kKIterPerWarp == Geometry::kPvStageK / WarpGemm::kK);

        constexpr auto outer_encoding = tile_distribution_encoding<
            sequence<kMWarp>,
            tuple<sequence<kNIterPerWarp, kNWarp>, sequence<kKIterPerWarp>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<2, 1>,
            sequence<0, 0>>{};
        constexpr auto block_encoding = detail::make_embed_tile_distribution_encoding(
            outer_encoding, typename WarpGemm::BWarpDstrEncoding{});
        using ReadEncoding =
            typename InputTileDistributionTraits<decltype(block_encoding),
                                                 typename Problem::VDataType>::TransposedDstrEncode;

        return make_static_tile_distribution(ReadEncoding{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_fp8_sched requires a full-head K LDS allocation");
        return kKFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        return kVFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        return GetLdsArenaSize();
    }

    // Without kPadQRows, Q reuses the qr_tdm LDS layout, which pads only 16-bit types.
    template <typename Problem>
    using QPadding = std::conditional_t<kPadQRows,
                                        KPadding,
                                        typename BasePolicy::template LdsPaddingConfigQ<Problem>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        static_assert(Problem::BlockFmhaShape::kSubQKHeaddim == kKValidWidth);
        constexpr auto desc = [] {
            if constexpr(kPadQRows)
                return detail::make_qr_tdm_row_major_lds_descriptor<typename Problem::QDataType,
                                                                    Problem::BlockFmhaShape::kM0,
                                                                    kKValidWidth,
                                                                    QPadding<Problem>,
                                                                    detail::kQrTdmLdsAccessBytes>();
            else
                return BasePolicy::template MakeQLdsBlockDescriptor<Problem>();
        }();
        static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                          kV1FootprintBytes,
                      "Q LDS descriptor overflows the V1 buffer");
        return desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        using Padding = detail::EncodedTdmPadding<QPadding<Problem>>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        using Padding = detail::EncodedTdmPadding<KPadding>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_fp8_sched policy geometry does not match the problem");
        using Padding = detail::EncodedTdmPadding<VPadding>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }
};

// Every quantization mode fits 512 VGPRs without mainloop spills, so all run two per CU.
inline constexpr index_t kFmhaTdmSchedFp8BlockPerCu = 2;
// The elementwise bias tile does not fit two workgroups' VGPRs, so it runs one per CU.
inline constexpr index_t kFmhaTdmSchedFp8BiasBlockPerCu = 1;
// D256 exceeds half of a CU's VGPRs and LDS, so it runs one workgroup per CU.
inline constexpr index_t kFmhaTdmSchedFp8D256BlockPerCu = 1;
// D192 MX scales (six bytes per Q/K row) spill at two per CU, so it runs one per CU.
inline constexpr index_t kFmhaTdmSchedFp8D192MxBlockPerCu = 1;
// D192 merged-GQA decode runs one per CU where two overflow the LDS (FP32 O) or VGPRs (sink).
inline constexpr index_t kFmhaTdmSchedFp8D192MergedBlockPerCu = 1;
template <typename Problem>
inline constexpr bool kFmhaTdmSchedFp8D192MergedOnePerCu =
    Problem::BlockFmhaShape::kQKHeaddim == 192 && Problem::kMergeNumHeadGroupsSeqLenQ &&
    (((Problem::QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD ||
       Problem::QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE) &&
      sizeof(typename Problem::ODataType) == 4) ||
     (Problem::kHasSink && Problem::FmhaMask::IsMasking));
// Target 2 per SIMD; D128/D192 are LDS-bound to 1 until the one-wavefront LDS footprint shrinks.
template <index_t HeadDim>
inline constexpr index_t kFmhaTdmSchedFp8OneWaveBlockPerCu = HeadDim == 256 ? 1 : 2;

// Dense four-wavefront D256 splits each K/V tile across LDS segments by wavefront parity.
template <typename Problem>
inline constexpr bool kFmhaTdmSchedFp8LdsSegLayout =
    Problem::BlockFmhaShape::kQKHeaddim == 256 && FmhaTdmSchedGeometryFor<Problem>::kWaves == 4 &&
    (Problem::QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ||
     Problem::QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR ||
     Problem::QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD) &&
    !Problem::kMergeNumHeadGroupsSeqLenQ && !Problem::kHasSink && !Problem::FmhaMask::IsMasking &&
    Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;

template <typename Problem>
using FmhaTdmSchedFp8TuningFor = FmhaTdmSchedDerivedTuning<
    FmhaTdmSchedGeometryFor<Problem>,
    Problem::BlockFmhaShape::kM0 == 32
        ? kFmhaTdmSchedFp8OneWaveBlockPerCu<Problem::BlockFmhaShape::kQKHeaddim>
    : Problem::BlockFmhaShape::kQKHeaddim == 256 ? kFmhaTdmSchedFp8D256BlockPerCu
    : Problem::BlockFmhaShape::kQKHeaddim == 192 &&
            Problem::QScaleEnum == BlockAttentionQuantScaleEnum::MX
        ? kFmhaTdmSchedFp8D192MxBlockPerCu
    : kFmhaTdmSchedFp8D192MergedOnePerCu<Problem>          ? kFmhaTdmSchedFp8D192MergedBlockPerCu
    : Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? kFmhaTdmSchedFp8BlockPerCu
                                                           : kFmhaTdmSchedFp8BiasBlockPerCu>;

template <typename Problem>
using FmhaTdmSchedFp8PolicyFor = BlockFmhaPipelineQRKSVSTdmSchedFp8Policy<
    FmhaTdmSchedGeometryFor<Problem>,
    std::conditional_t<kFmhaTdmSchedFp8LdsSegLayout<Problem>,
                       FmhaTdmSchedFp8LdsSegTuning<FmhaTdmSchedFp8TuningFor<Problem>>,
                       FmhaTdmSchedFp8TuningFor<Problem>>>;

} // namespace ck_tile
