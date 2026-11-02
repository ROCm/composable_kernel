// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_load.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_prepared_k_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_prepared_v_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_affine_k_read.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_affine_v_read.hpp"

namespace ck_tile {

template <typename Geometry_,
          typename Tuning_,
          typename TileSchedule_ = FmhaTdmSchedDefaultScheduleFor<Geometry_>>
struct BlockFmhaPipelineQRKSVSTdmSchedPolicy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    using BasePolicy = BlockFmhaPipelineQRKSVSTdmDefaultPolicy;
    using Geometry   = Geometry_;
    using Tuning     = Tuning_;
    using Schedule   = TileSchedule_;
    using Layout     = FmhaTdmSchedLayout<Geometry>;
    using KLoad      = FmhaTdmSchedLoad<Geometry::kKSuLoadCount>;
    using VLoad      = FmhaTdmSchedTransposeLoad<Geometry::kVStageLoadCount>;
    static_assert(Schedule::kNumQkStages == Geometry::kQkStages &&
                  Schedule::kNumPvStages == Geometry::kPvStages &&
                  Schedule::kQkWmmasPerStage == Geometry::kQkWmmasPerStage &&
                  Schedule::kPvWmmasPerStage == Geometry::kPvWmmasPerStage);
    static constexpr index_t kBlockPerCu = Tuning::kBlockPerCu;

    static constexpr index_t kLdsRows            = Geometry::kN;
    static constexpr index_t kKValidWidth        = Geometry::kHeadDimQK;
    static constexpr index_t kKPhysicalStride    = Layout::kKPhysicalStride;
    static constexpr index_t kVLogicalWidth      = Geometry::kDv;
    static constexpr index_t kVPhysicalStride    = Layout::kVPhysicalStride;
    static constexpr index_t kVPadAmount         = Layout::kVPadAmount;
    static constexpr index_t kVPadInterval       = Layout::kVPadInterval;
    static constexpr index_t kKFootprintElements = kLdsRows * kKPhysicalStride;
    static constexpr index_t kVFootprintElements = kLdsRows * kVPhysicalStride;
    static constexpr index_t kKFootprintBytes =
        kKFootprintElements * sizeof(typename Geometry::KDataType);
    static constexpr index_t kVFootprintBytes =
        kVFootprintElements * sizeof(typename Geometry::VDataType);
    // Named geometry knobs. Values frozen to the current D128/D192 formulas.
    static constexpr bool kEarlyOutputRescale = Geometry::kHeadDimQK == 128;
    static constexpr bool kLeadFirstLoad      = Geometry::kHeadDimQK == 128;
    static constexpr bool kUseUnsignedCausalMask =
        Geometry::kHeadDimQK == 128 || Geometry::kHeadDimQK == 192;
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
        s_wait_dscnt<GetQkTailDsCount<Stage>()>();
    }

    template <index_t Stage>
    CK_TILE_DEVICE static void WaitPvStageTail()
    {
        static_assert(Stage >= 0 && Stage < Geometry::kPvStages);
        s_wait_dscnt<GetPvTailDsCount<Stage>()>();
    }

    static constexpr index_t kLdsOffsetK0  = 0;
    static constexpr index_t kLdsOffsetK1  = kLdsOffsetK0 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV0  = kLdsOffsetK1 + kKFootprintBytes;
    static constexpr index_t kLdsOffsetV1  = kLdsOffsetV0 + kVFootprintBytes;
    static constexpr index_t kLdsArenaSize = kLdsOffsetV1 + kVFootprintBytes;

    using OutputFragments = FmhaTdmSchedOutputFragmentsFor<Geometry::kPvNIter / 2>;

    // The split-softmax score mapping and the PV output fragments are hand-written for one
    // decomposition; tie them to the derived geometry.
    static_assert(FmhaTdmSchedSupportedGeometry<Geometry>::value);
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

    // Q is used only during the prologue. Its descriptor may span both K buffers
    // when the KV tile is 64 rows; the LDS barrier precedes the K prologue.
    static_assert(Geometry::kM * Geometry::kHeadDimQK * sizeof(typename Geometry::QDataType) <=
                  2 * kKFootprintBytes);
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

    template <index_t WmmaOrdinal,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor>
    CK_TILE_DEVICE static void RunQkSuWmma(CBlockTensor& c_block_tensor,
                                           const ABlockTensor& a_block_tensor,
                                           const BBlockTensor& b_block_tensor)
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

        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
        c_block_tensor.set_y_sliced_thread_data(
            merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
            c_warp_tensor.get_thread_buffer());
    }

    template <index_t Stage,
              typename Mode,
              typename BlockGemm,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename NextBBlockTensor,
              typename BTileWindow>
    CK_TILE_DEVICE static void
    RunQkScheduledStage(const BlockGemm&,
                        CBlockTensor& c_block_tensor,
                        const ABlockTensor& a_block_tensor,
                        const BBlockTensor& b_block_tensor,
                        NextBBlockTensor& next_b_block_tensor,
                        const BTileWindow& b_lds_window,
                        const FmhaTdmPreparedKRead::Address* prepared_stage0 = nullptr)
    {
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumQkStages);
        static_assert((Stage < SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount) ||
                      (Stage == SelectedSchedule::kNumQkStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount));
        constexpr bool kUseAffineKRead = Stage < Geometry::kQkStages - 1;

        const auto affine_k0 = [&]() {
            if constexpr(kUseAffineKRead)
            {
                if constexpr(Stage == 0)
                {
                    if(prepared_stage0 != nullptr)
                        return *prepared_stage0;
                }
                return FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            }
            else
                return FmhaTdmPreparedKRead::Address{};
        }();
        const auto affine_v0 = [&]() {
            if constexpr(Stage == Geometry::kQkStages - 1)
                return FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            else
                return FmhaTdmPreparedKRead::Address{};
        }();
        constexpr bool kLeadFirstLoadThisStage = Mode::SelectedPolicy::kLeadFirstLoad &&
                                                 (Stage == 0 || Stage == Geometry::kQkStages - 1);
        if constexpr(kLeadFirstLoadThisStage)
        {
            SelectedSchedule::template RunQkReadPrelude<Mode, Stage>([&](auto event) {
                static_assert(decltype(event)::value == 0);
                if constexpr(Stage == 0)
                    FmhaTdmAffineKRead::Load<Stage, 0, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
                else
                    FmhaTdmAffineVRead::Load<0, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_v0);
            });
            __builtin_amdgcn_sched_barrier(0);
        }
        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
            {
                static_assert(Stage < Geometry::kQkStages - 1);
                if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
                {
                    // Access0 was issued before WMMA0; do not issue it twice.
                }
                else
                    FmhaTdmAffineKRead::Load<Stage, decltype(access)::value, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
            }
            else if constexpr(decltype(kind)::value == Kind::VRead)
            {
                static_assert(Stage == Geometry::kQkStages - 1);
                if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
                {
                    // Access0 was issued before WMMA0; do not issue it twice.
                }
                else
                    FmhaTdmAffineVRead::Load<decltype(access)::value, kVPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_v0);
            }
        };
        auto emit_point = [](auto, auto, auto point) {
            if constexpr(decltype(point)::value == FmhaTdmSchedSchedulePoint::AfterTokens)
                __builtin_amdgcn_sched_barrier(0);
            else
                __builtin_amdgcn_sched_barrier(0x0002 | 0x0400);
        };

        auto emit_wmma = [&](auto, auto wmma) {
            RunQkSuWmma<decltype(wmma)::value, BlockGemm>(
                c_block_tensor, a_block_tensor, b_block_tensor);
        };
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_token, emit_point);
        WaitQkStageTail<Stage>();
    }

    template <index_t Stage,
              typename Mode,
              typename NextBBlockTensor,
              typename BTileWindow,
              typename WmmaEmitter,
              typename RescaleEmitter>
    CK_TILE_DEVICE static void RunPvScheduledStageWithWmma(NextBBlockTensor& next_b_block_tensor,
                                                           const BTileWindow& b_lds_window,
                                                           WmmaEmitter& emit_wmma,
                                                           RescaleEmitter& emit_rescale)
    {
        using SelectedSchedule = typename Mode::SelectedPolicy::Schedule;
        using Executor         = FmhaTdmSchedPlacementExecutor<SelectedSchedule, Mode>;
        using Kind             = FmhaTdmSchedLoadKind;
        static_assert(Stage >= 0 && Stage < SelectedSchedule::kNumPvStages);
        static_assert((Stage < SelectedSchedule::kNumPvStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kVStageLoadCount) ||
                      (Stage == SelectedSchedule::kNumPvStages - 1 &&
                       BTileWindow::NumAccessPerCoord == Geometry::kKSuLoadCount));

        FmhaTdmPreparedKRead::Address affine_k0{};
        FmhaTdmPreparedKRead::Address prepared_v1{};
        constexpr bool kLeadFirstLoadThisStage = Mode::SelectedPolicy::kLeadFirstLoad &&
                                                 (Stage == 0 || Stage == Geometry::kPvStages - 1);
        if constexpr(kLeadFirstLoadThisStage)
        {
            SelectedSchedule::template RunPvReadPrelude<Mode, Stage>([&](auto event) {
                static_assert(decltype(event)::value == 0);
                if constexpr(Stage == 0)
                {
                    prepared_v1 = FmhaTdmPreparedKRead::Prepare<1>(b_lds_window);
                    __builtin_amdgcn_sched_barrier(0);
                    VLoad::template LoadAccess<0>(next_b_block_tensor, b_lds_window);
                }
                else
                {
                    affine_k0 = FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
                    FmhaTdmAffineKRead::Load<Stage, 0, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
                }
            });
            __builtin_amdgcn_sched_barrier(0);
        }
        auto emit_prepared_wmma = [&](auto stage, auto wmma) {
            if constexpr(!kLeadFirstLoadThisStage && Stage == 0 && decltype(wmma)::value == 0)
            {
                prepared_v1 = FmhaTdmPreparedKRead::Prepare<1>(b_lds_window);
                __builtin_amdgcn_sched_barrier(0);
            }
            if constexpr(!kLeadFirstLoadThisStage && Stage == Geometry::kPvStages - 1 &&
                         decltype(wmma)::value == 0)
            {
                affine_k0 = FmhaTdmPreparedKRead::Prepare<0>(b_lds_window);
            }
            emit_wmma(stage, wmma);
        };
        auto emit_token = [&](auto kind, auto access) {
            if constexpr(decltype(kind)::value == Kind::KRead)
            {
                static_assert(Stage == Geometry::kPvStages - 1);
                if constexpr(Geometry::kN == 64)
                    KLoad::template LoadInstruction<decltype(access)::value>(next_b_block_tensor,
                                                                             b_lds_window);
                else
                    FmhaTdmAffineKRead::Load<Stage, decltype(access)::value, kKPhysicalStride>(
                        next_b_block_tensor, b_lds_window, affine_k0);
            }
            else if constexpr(decltype(kind)::value == Kind::VRead)
            {
                static_assert(Stage < Geometry::kPvStages - 1);
                VLoad::template LoadAccess<decltype(access)::value>(next_b_block_tensor,
                                                                    b_lds_window);
            }
        };
        auto emit_point = [](auto, auto, auto point) {
            if constexpr(decltype(point)::value == FmhaTdmSchedSchedulePoint::AfterTokens)
                __builtin_amdgcn_sched_barrier(0);
            else
                __builtin_amdgcn_sched_barrier(0x0002 | 0x0400);
        };

        auto emit_prepared_token = [&](auto kind, auto access) {
            if constexpr(kLeadFirstLoadThisStage && decltype(access)::value == 0)
            {
                // Access0 was issued before WMMA0; do not issue it twice.
            }
            else if constexpr(Stage == 0 && decltype(kind)::value == Kind::VRead &&
                              decltype(access)::value == 1)
                FmhaTdmPreparedVRead::Load<1>(next_b_block_tensor, b_lds_window, prepared_v1);
            else
                emit_token(kind, access);
        };

        Executor::template ExecutePvStage<Stage>(
            emit_prepared_wmma, emit_prepared_token, emit_point, emit_rescale);
        WaitPvStageTail<Stage>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCountdownLoop()
    {
        return Geometry::kHeadDimQK == 128 && !Problem::FmhaMask::IsMasking && !Problem::kHasSink &&
               Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool DeferTensorReady()
    {
        // Defer the V tensor wait only for the supported geometry and problem cases.
        return (Geometry::kHeadDimQK == 128 ||
                (std::is_same_v<Geometry, FmhaTdmSchedBf16D192N64Geometry> &&
                 !Problem::FmhaMask::IsMasking)) &&
               !Problem::kHasSink && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{}) == 1;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool UseCompilerMax()
    {
        // Let the allocator place score tuples above the inline-asm low-VGPR range.
        // D64/D128 and D192 dense batch use compiler-scheduled maximum chains.
        return Geometry::kHeadDimQK <= 128 ||
               (Geometry::kHeadDimQK == 192 && !Problem::kIsGroupMode &&
                !Problem::FmhaMask::IsMasking);
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart01(
        ScoreTensor& score, RowTensor& row_max, float log2e_scale, float& delta_m0, float& delta_m1)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        const float old_max_m0 = row_max.get_thread_buffer()[number<0>{}];
        const float old_max_m1 = row_max.get_thread_buffer()[number<1>{}];

        constexpr bool kUseCompilerMax = UseCompilerMax<Problem>();
        const auto p0_m0 = Softmax::template RunPart0<0, ScoreTensor, kUseCompilerMax>(
            score, old_max_m0, log2e_scale);
        const auto p0_m1 = Softmax::template RunPart0<1, ScoreTensor, kUseCompilerMax>(
            score, old_max_m0, log2e_scale);
        const auto p0_m2 = Softmax::template RunPart0<2, ScoreTensor, kUseCompilerMax>(
            score, old_max_m1, log2e_scale);
        const auto p0_m3 = Softmax::template RunPart0<3, ScoreTensor, kUseCompilerMax>(
            score, old_max_m1, log2e_scale);

        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking;
        const auto p1_m0            = Softmax::template RunPart1<kValidateMax>(
            p0_m0.local_max, p0_m1.local_max, p0_m0.old_max_log2e, log2e_scale);
        const auto p1_m1 = Softmax::template RunPart1<kValidateMax>(
            p0_m2.local_max, p0_m3.local_max, p0_m2.old_max_log2e, log2e_scale);

        row_max.get_thread_buffer()[number<0>{}] = p1_m0.row_max;
        row_max.get_thread_buffer()[number<1>{}] = p1_m1.row_max;
        delta_m0                                 = p1_m0.delta;
        delta_m1                                 = p1_m1.delta;
    }

    template <typename Problem, typename ScoreTensor, typename RowTensor>
    CK_TILE_DEVICE static void RunSplitSoftmaxPart2AndGetScale(ScoreTensor& score,
                                                               const RowTensor& row_max,
                                                               RowTensor& row_sum,
                                                               float log2e_scale,
                                                               float delta_m0,
                                                               float delta_m1,
                                                               float& output_scale_m0,
                                                               float& output_scale_m1)
    {
        using Softmax = ScoreSoftmax;
        using Mapping = typename Softmax::Mapping;

        static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
        static_assert(!Problem::kHasLogitsSoftCap);
        static_assert(ScoreTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        static_assert(RowTensor::get_thread_buffer_size() == 2);

        constexpr bool kValidateMax = Problem::FmhaMask::IsMasking;

        const auto validated_max = [](float value) {
            if constexpr(kValidateMax)
            {
                return value <= -numeric<float>::infinity() ? 0.0f : value;
            }
            else
            {
                return value;
            }
        };

        const float row_max_m0 = row_max.get_thread_buffer()[number<0>{}];
        const float row_max_m1 = row_max.get_thread_buffer()[number<1>{}];
        const float old_sum_m0 = row_sum.get_thread_buffer()[number<0>{}];
        const float old_sum_m1 = row_sum.get_thread_buffer()[number<1>{}];

        const float sum_m0 =
            Softmax::template RunPart2LocalSum<0>(score, validated_max(row_max_m0), log2e_scale);
        const float sum_m1 =
            Softmax::template RunPart2LocalSum<1>(score, validated_max(row_max_m0), log2e_scale);
        const float sum_m2 =
            Softmax::template RunPart2LocalSum<2>(score, validated_max(row_max_m1), log2e_scale);
        const float sum_m3 =
            Softmax::template RunPart2LocalSum<3>(score, validated_max(row_max_m1), log2e_scale);

        row_sum.get_thread_buffer()[number<0>{}] =
            Softmax::UpdateRowSum(old_sum_m0, Softmax::MergeRowSum(sum_m0, sum_m1), delta_m0);
        row_sum.get_thread_buffer()[number<1>{}] =
            Softmax::UpdateRowSum(old_sum_m1, Softmax::MergeRowSum(sum_m2, sum_m3), delta_m1);

        output_scale_m0 = Softmax::Exp2(delta_m0);
        output_scale_m1 = Softmax::Exp2(delta_m1);
    }

    template <index_t Ordinal, typename Fragments>
    CK_TILE_DEVICE static void
    RescaleOutputFragment(Fragments& output, float scale_m0, float scale_m1)
    {
        static_assert(Ordinal >= 0 && Ordinal < OutputFragments::kNumFragments);
        constexpr index_t d_msb = Ordinal / OutputFragments::kNumN;
        const float scale       = d_msb < 2 ? scale_m0 : scale_m1;
        auto& fragment          = output.at(number<Ordinal>{});
        static_for<0, OutputFragments::kElementsPerFragment, 1>{}(
            [&](auto element) { fragment[decltype(element)::value] *= scale; });
    }

    template <typename Problem,
              bool WaitTensorAfterMax = false,
              typename ScoreTensor,
              typename RowTensor,
              typename Fragments>
    CK_TILE_DEVICE static void RunSplitSoftmaxFragments(ScoreTensor& score,
                                                        RowTensor& row_max,
                                                        RowTensor& row_sum,
                                                        Fragments& output,
                                                        float log2e_scale)
    {
        float delta_m0;
        float delta_m1;
        RunSplitSoftmaxPart01<Problem>(score, row_max, log2e_scale, delta_m0, delta_m1);

        if constexpr(WaitTensorAfterMax)
            s_wait_tensorcnt_barrier<kVPrefetchTensorCount>();

        const auto rescale_output = [&](float scale_m0, float scale_m1) {
            static_for<0, OutputFragments::kNumFragments, 1>{}([&](auto ordinal) {
                constexpr index_t d_msb = decltype(ordinal)::value / OutputFragments::kNumN;
                const float scale       = d_msb < 2 ? scale_m0 : scale_m1;
                auto& fragment          = output.at(ordinal);
                static_for<0, OutputFragments::kElementsPerFragment, 1>{}(
                    [&](auto element) { fragment[decltype(element)::value] *= scale; });
            });
        };
        if constexpr(kEarlyOutputRescale)
        {
            using Softmax = ScoreSoftmax;
            rescale_output(Softmax::Exp2(delta_m0), Softmax::Exp2(delta_m1));
        }

        float output_scale_m0;
        float output_scale_m1;
        RunSplitSoftmaxPart2AndGetScale<Problem>(score,
                                                 row_max,
                                                 row_sum,
                                                 log2e_scale,
                                                 delta_m0,
                                                 delta_m1,
                                                 output_scale_m0,
                                                 output_scale_m1);

        if constexpr(!kEarlyOutputRescale)
        {
            rescale_output(output_scale_m0, output_scale_m1);
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
        // The problem must describe exactly this policy's geometry (dtypes, block tile, wave
        // layout, WMMA tiles). Both GEMMs share one wave layout: P stays in the lanes that
        // produced S.
        return Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
               std::is_same_v<FmhaTdmSchedGeometryFor<Problem, Geometry::kQkSuColumns>, Geometry> &&
               Shape::Gemm1BlockWarps::at(number<0>{}) == Shape::Gemm0BlockWarps::at(number<0>{}) &&
               Shape::Gemm1BlockWarps::at(number<1>{}) == Shape::Gemm0BlockWarps::at(number<1>{}) &&
               Shape::Gemm1BlockWarps::at(number<2>{}) == Shape::Gemm0BlockWarps::at(number<2>{}) &&
               std::is_same_v<remove_cvref_t<typename Problem::SaccDataType>, float> &&
               std::is_same_v<remove_cvref_t<typename Problem::OaccDataType>, float> &&
               Shape::kK0 == Geometry::kQkWmmaK && Shape::kSubQKHeaddim == Geometry::kHeadDimQK &&
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
                      "qr_tdm_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K TDM load");
        constexpr index_t warp_num = Problem::kBlockSize / get_warp_size();
        static_assert(kLdsRows % warp_num == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<warp_num, kLdsRows / warp_num>, sequence<kKPhysicalStride>>,
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
                      "qr_tdm_sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackK<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<kKPhysicalStride>{}),
            make_tuple(number<kKPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem, bool LoadOnce = true>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K LDS descriptor");
        return MakeKLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        constexpr index_t kKPack = BasePolicy::template GetSmemKPackV<Problem>();
        return make_naive_tensor_descriptor(
            make_tuple(number<kLdsRows>{}, number<kVLogicalWidth>{}),
            make_tuple(number<kVPhysicalStride>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadBlockDescriptor()
    {
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        static_assert(!Xor, "D192/V128 TDM writer requires an affine V LDS descriptor");
        return MakeVLdsWriteBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
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
                      "qr_tdm_sched policy geometry does not match the problem");
        static_assert(LoadOnce, "qr_tdm_sched requires a full-head K LDS allocation");
        return kKFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return kVFootprintBytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return GetLdsArenaSize();
    }

    template <typename Problem>
    static constexpr bool kUsePaddedQ = Geometry::kHeadDimQK <= 128 && Geometry::kM == 128;

    template <typename Problem>
    using LdsPaddingConfigQ = std::conditional_t<
        kUsePaddedQ<Problem>,
        detail::
            LdsPaddingConfig<true, Geometry::kHeadDimQK * sizeof(typename Geometry::QDataType), 16>,
        typename BasePolicy::template LdsPaddingConfigQ<Problem>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        static_assert(IsSupportedProblem<Problem>(), "V128 policy received an invalid problem");
        if constexpr(kUsePaddedQ<Problem>)
        {
            using Padding = LdsPaddingConfigQ<Problem>;
            constexpr auto desc =
                detail::make_qr_tdm_row_major_lds_descriptor<typename Problem::QDataType,
                                                             Geometry::kM,
                                                             Geometry::kHeadDimQK,
                                                             Padding,
                                                             detail::kQrTdmLdsAccessBytes>();
            static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                          2 * kKFootprintBytes);
            return desc;
        }
        else
        {
            constexpr auto desc = BasePolicy::template MakeQLdsBlockDescriptor<Problem>();
            static_assert(desc.get_element_space_size() * sizeof(typename Problem::QDataType) <=
                              2 * kKFootprintBytes,
                          "Q LDS descriptor overlaps the V buffer");
            return desc;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigQ()
    {
        static_assert(IsSupportedProblem<Problem>(), "V128 policy received an invalid problem");
        using Padding = detail::EncodedTdmPadding<LdsPaddingConfigQ<Problem>>;
        return make_tuple(number<Padding::kEnabled>{},
                          number<Padding::kPadAmount>{},
                          number<Padding::kPadInterval>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigK()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return make_tuple(number<false>{}, number<0>{}, number<0>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfigV()
    {
        static_assert(IsSupportedProblem<Problem>(),
                      "qr_tdm_sched policy geometry does not match the problem");
        return make_tuple(number<true>{}, number<kVPadAmount>{}, number<kVPadInterval>{});
    }
};

template <typename Geometry>
struct FmhaTdmSchedTuningSelector
{
    static constexpr index_t kBlockPerCu =
        Geometry::kHeadDimQK == 64 ? 3
                                   : (Geometry::kHeadDimQK == 192 && Geometry::kN == 64 ? 2 : 1);
    using type = FmhaTdmSchedDerivedTuning<Geometry, kBlockPerCu>;
};

template <typename Geometry>
using FmhaTdmSchedDefaultTuning = typename FmhaTdmSchedTuningSelector<Geometry>::type;

template <typename Problem>
using FmhaTdmSchedPolicyFor = BlockFmhaPipelineQRKSVSTdmSchedPolicy<
    FmhaTdmSchedGeometryFor<Problem>,
    FmhaTdmSchedDefaultTuning<FmhaTdmSchedGeometryFor<Problem>>>;

} // namespace ck_tile
