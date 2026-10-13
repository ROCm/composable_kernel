// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_policy.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// Hand-scheduled gfx125 TDM pipeline for native 16-bit D64/V64 and D128/D192 x V128.
// Separate template instantiations keep finite virtual-sink normalization out of
// the generated no-sink kernel's register allocation.
template <typename Problem_,
          typename Policy_        = FmhaTdmSchedPolicyFor<Problem_>,
          bool EnableVirtualSink_ = true>
struct BlockFmhaPipelineQRKSVSTdmSched
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using SplitSoftmax          = FmhaTdmSchedSplitSoftmaxFor<Policy::Geometry::kQkStages>;
    using Schedule              = typename Policy::Schedule;
    using Mode                  = FmhaTdmSchedMode<Problem, Policy>;
    static_assert(Mode::kIsLegal,
                  "qr_tdm_sched schedule geometry or output-rescale placement is incompatible");

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);
    static constexpr bool kKLoadOnce = BlockFmhaShape::kM0 > 64;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;
    static constexpr index_t kNWarp        = BlockFmhaShape::Gemm0BlockWarps::at(I1);

    static constexpr bool kIsGroupMode       = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ        = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK        = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ       = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV       = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap  = Problem::kHasLogitsSoftCap;
    static constexpr bool kHasDropout        = Problem::kHasDropout;
    static constexpr auto BiasEnum           = Problem::BiasEnum;
    static constexpr bool kStoreLSE          = Problem::kStoreLSE;
    static constexpr bool kHasUnevenSplits   = true;
    static constexpr bool kHasSink           = Problem::kHasSink;
    static constexpr bool kEnableVirtualSink = EnableVirtualSink_;
    static constexpr auto QScaleEnum         = Problem::QScaleEnum;

    static_assert(CK_TILE_FMHA_FWD_FAST_EXP2,
                  "qr_tdm_sched: every softmax site calls exp2, so log2(e) must be folded into "
                  "scale_s");
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "qr_tdm_sched does not implement attention bias");
    static_assert(!kHasLogitsSoftCap, "qr_tdm_sched does not implement logits soft cap");
    static_assert(!kHasDropout, "qr_tdm_sched does not support dropout");
    static_assert(QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE,
                  "qr_tdm_sched does not implement quantized attention yet");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ    = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK    = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV    = Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentO    = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentOacc = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static_assert(Policy::template IsSupportedProblem<Problem>(),
                  "qr_tdm_sched requires an exact supported FP16/BF16 geometry");

    // IsSupportedProblem above pins Geometry == FmhaTdmSchedGeometryFor<Problem>.
    using Geometry = typename Policy::Geometry;
    static_assert(Geometry::kM == kM0 && Geometry::kN == kN0 && Geometry::kDv == kN1);

    static constexpr const char* name = "qr_tdm_sched";

    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;
    static constexpr bool kUsesFixedSegmentedLdsArena  = true;
    static constexpr index_t kBlockPerCu               = Policy::kBlockPerCu;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    using QKBlockGemm = remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
    using PVBlockGemm = remove_cvref_t<decltype(Policy::template GetPVBlockGemm<Problem>())>;
    using ScoreTile   = decltype(QKBlockGemm::MakeCBlockTile());
    static_assert(ScoreTile::get_thread_buffer_size() == SplitSoftmax::Mapping::kThreadBufferSize);
    // Both WarpGemm implementations expose the physical instruction shape here.
    using QkMmaShape = typename QKBlockGemm::WarpGemm::WarpGemmAttribute::Impl;
    using PvMmaShape = typename PVBlockGemm::WarpGemm::WarpGemmAttribute::Impl;
    static_assert(QkMmaShape::kM == Geometry::QkWmma::kM &&
                      QkMmaShape::kN == Geometry::QkWmma::kN &&
                      QkMmaShape::kK == Geometry::QkWmma::kK,
                  "gemm_0 dispatched a different physical WMMA shape");
    static_assert(PvMmaShape::kM == Geometry::PvWmma::kM &&
                      PvMmaShape::kN == Geometry::PvWmma::kN &&
                      PvMmaShape::kK == Geometry::PvWmma::kK,
                  "gemm_1 dispatched a different physical WMMA shape");
#endif

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(Policy::template GetSmemSize<Problem>() == Policy::GetLdsArenaSize());
        return Policy::GetLdsArenaSize();
    }

    // Re-pack gemm_0 C into gemm_1 A: C is M-outer (MIter,KIter), A is K-outer.
    // Lanes already align (NWarp==1), so this is an in-thread block reorder;
    // identity when MIterPerWarp==1.
    template <typename Gemm1, typename PComputeTensor>
    CK_TILE_DEVICE static auto MakePForGemm1(const PComputeTensor& p_compute)
    {
        static_assert(Geometry::PRepackLayout::kInThread &&
                          Gemm1::MIterPerWarp == Geometry::kPvMIter,
                      "P re-pack requires gemm_0 C^T and gemm_1 A to share lanes");
        auto p_tile = make_static_distributed_tensor<PDataType>(
            Policy::template MakePRegTileDistribution<Problem>());
        const auto p_src        = cast_tile<PDataType>(p_compute);
        constexpr index_t kPBuf = decltype(p_src)::get_thread_buffer_size();
        constexpr index_t kPMI  = Gemm1::MIterPerWarp;
        constexpr index_t kPKI  = kN0 / Gemm1::WarpGemm::kK;
        constexpr index_t kPSub = kPBuf / (kPMI * kPKI);
        using p_bulk_t          = array<PDataType, kPSub>;
        static_for<0, kPMI, 1>{}([&](auto mi) {
            static_for<0, kPKI, 1>{}([&](auto ki) {
                p_tile.get_thread_buffer().template set_as<p_bulk_t>(
                    number<ki * kPMI + mi>{},
                    p_src.get_thread_buffer().template get_as<p_bulk_t>(number<mi * kPKI + ki>{}));
            });
        });
        return p_tile;
    }

    // Prefill, double lds
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    run(const QDramBlockWindowTmp& __restrict__ q_dram_block_window_tmp,       // M0*K0 tile
        const KDramBlockWindowTmp& __restrict__ k_dram_block_window_tmp,       // N0*K0 tile
        const VDramBlockWindowTmp& __restrict__ v_dram_block_window_tmp,       // N1*K1 tile
        const BiasDramBlockWindowTmp& __restrict__ bias_dram_block_window_tmp, // M0*N0 tile
        LSEaccDramBlockWindowTmp& __restrict__ lse_acc_dram_window_tmp,        // M0*1 tile
        FmhaMask mask,
        PositionEncoding position_encoding,
        float scale_s,
        void* __restrict__ smem_ptrk0,
        void* __restrict__ smem_ptrk1,
        void* __restrict__ smem_ptrv0,
        void* __restrict__ smem_ptrv1,
        float sink_v) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Q, K, and V use TDM global-to-LDS transfers; V is consumed through
        // transposed LDS loads.
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");
        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max        = [](auto e0, auto e1) { return max(e0, e1); };
        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        using PvBlockGemm   = remove_cvref_t<decltype(gemm_1)>;
        using PvWarpGemm    = typename PvBlockGemm::WarpGemm;
        using PvCWarpTensor = typename PvWarpGemm::CWarpTensor;
        static_assert(PvBlockGemm::KIterPerWarp == 1 && PvBlockGemm::MIterPerWarp == 2 &&
                      PvBlockGemm::NIterPerWarp == Geometry::kPvNIter);
        static_assert(std::is_same_v<OaccDataType, float>);
        static_assert(std::is_same_v<remove_cvref_t<typename PvCWarpTensor::DataType>, float>);
        static_assert(std::is_same_v<typename Policy::OutputFragments::Fragment, fp32x8_t>);
        static_assert(PvCWarpTensor::get_thread_buffer_size() ==
                      Policy::OutputFragments::kElementsPerFragment);
        static_assert(Policy::OutputFragments::kNumDmsb == 2 * PvBlockGemm::MIterPerWarp);
        static_assert(Policy::OutputFragments::kNumN == PvBlockGemm::NIterPerWarp / 2);
        static_assert(OaccBlockTileType::get_thread_buffer_size() ==
                      Policy::OutputFragments::Mapping::kThreadBufferSize);
        if constexpr(Geometry::kN == 128)
            static_assert(std::is_same_v<typename SaccBlockTileType::StaticTileDistribution,
                                         typename OaccBlockTileType::StaticTileDistribution>,
                          "N128 score/output distributions must preserve the fragment "
                          "reconstruction map");

        auto o_acc = Policy::OutputFragments::MakeZero();

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(OaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // The shared kernel can supply a sink logit independently of kHasSink,
        // which controls sink tokens in the mask. Preserve this runtime input.
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::NO_BIAS)
        {
            // A virtual sink has V=0. Merge it only at normalization so its
            // logit cannot change the BF16 rounding of the real-token P operand.
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }
        else if(__builtin_isinf_sign(sink_v) >= 0)
        {
            set_tile(m, sink_v * C_LOG2E);
            set_tile(l, SMPLComputeDataType{1.0f});
        }
        else
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        const auto tile_range_result = [&]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
            else
            {
                auto [start, end] =
                    mask.GetTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end           = tile_range_result.get(ck_tile::number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) +
                num_sink_loop;
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse_acc =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    if(__builtin_isinf_sign(sink_v) >= 0)
                    {
                        set_tile(lse_acc, SMPLComputeDataType{sink_v * scale_s});
                    }
                    else
                    {
                        set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
                    }

                    store_tile(lse_acc_dram_window_tmp, lse_acc);
                }

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
            }
        }

        // ---------------------------------------------------------------------
        // TDM configs for Q / K / V
        // pad_enable + pad_amount + pad_interval are compile-time, sourced from
        // the policy. workgroup_mask defaults to 0 (no cluster multicast).
        // V uses load_tile_tdm (single-box plain layout) -- same TDM machinery
        // as Q / K, with V dram dist switched to trivial tile-major and the
        // V LDS read view kept plain row-major (matches the write view).
        // ---------------------------------------------------------------------
        TDMConfig tdm_config_q;
        TDMConfig tdm_config_k;
        TDMConfig tdm_config_v;
        {
            constexpr auto LdsPaddingConfigQ     = Policy::template GetLdsPaddingConfigQ<Problem>();
            tdm_config_q.pad_enable              = LdsPaddingConfigQ[I0];
            tdm_config_q.pad_config.pad_amount   = LdsPaddingConfigQ[I1];
            tdm_config_q.pad_config.pad_interval = LdsPaddingConfigQ[number<2>{}];

            constexpr auto LdsPaddingConfigK     = Policy::template GetLdsPaddingConfigK<Problem>();
            tdm_config_k.pad_enable              = LdsPaddingConfigK[I0];
            tdm_config_k.pad_config.pad_amount   = LdsPaddingConfigK[I1];
            tdm_config_k.pad_config.pad_interval = LdsPaddingConfigK[number<2>{}];

            constexpr auto LdsPaddingConfigV     = Policy::template GetLdsPaddingConfigV<Problem>();
            tdm_config_v.pad_enable              = LdsPaddingConfigV[I0];
            tdm_config_v.pad_config.pad_amount   = LdsPaddingConfigV[I1];
            tdm_config_v.pad_config.pad_interval = LdsPaddingConfigV[number<2>{}];
        }

        // Q tile in LDS
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        auto q_lds_write_view = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptrk0),
            Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_read_view = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptrk0),
            Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window =
            make_tile_window(q_lds_write_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_read_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
        s_wait_tensorcnt_barrier<0>();
        auto q_tile = load_tile(q_lds_read_window);

        // K tile in LDS (sink-aware start)
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;
        const index_t physical_seqlen_k_start = logical_seqlen_k_start;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end;

        // Bias tile window (prefill path)
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             gemm_0.MakeCBlockTile().get_tile_distribution());

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());

        auto k_lds_write_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType* __restrict__>(smem_ptrk0),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType* __restrict__>(smem_ptrk0),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_write_window = make_tile_window(
            k_lds_write_view,
            Policy::template MakeKLdsBlockDescriptor<Problem, true>().get_lengths(),
            {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_read_view,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());

        // V tile in LDS (sink-aware start)
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds_write_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(smem_ptrv0)),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_read_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(smem_ptrv0)),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds_write_view,
                             Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_read_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

        constexpr bool kUseCountdownLoop = Policy::template UseCountdownLoop<Problem>();
        bool even_buffer                 = true;
        index_t i_total_loops            = 0;
        index_t remaining_loops          = num_total_loop;
        constexpr index_t k1_loops       = kN0 / kK1;

        static_assert(1 <= k1_loops);
        block_sync_lds<0>();
        // Fixed TDM prologue: current K/V and the first look-ahead K buffer.
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
        move_tile_window(k_dram_window, {kN0, 0});
        // K1 is consumed at iteration 1. A one-tile prefix must jump before K1 loads.
        if constexpr(kHasSink)
        {
            if(num_sink_loop == 1 && num_total_loop > 1)
                move_tile_window(k_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
        }
        k_lds_write_window.set_bottom_tensor_view_data_ptr(
            static_cast<KDataType* __restrict__>(smem_ptrk1));
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        constexpr index_t k_lds_insts = k_lds_read_window.get_num_of_access();
        constexpr index_t v_lds_insts = v_lds_read_window.get_num_of_access();
        static_assert(k_lds_insts <= Geometry::kKSuLoadCount &&
                      v_lds_insts == Geometry::kVStageLoadCount);

        s_wait_tensorcnt_barrier<Policy::kKPrefetchTensorCount>();
        auto initial_k_lds_su_read_window =
            make_tile_window(k_lds_read_view,
                             make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                             {0, 0},
                             Policy::template MakeKSuRegTileDistribution<Problem>());
        initial_k_lds_su_read_window.set_bottom_tensor_view_data_ptr(
            static_cast<KDataType* __restrict__>(smem_ptrk0));
        auto k_su0_tile = load_tile(initial_k_lds_su_read_window);

        __builtin_amdgcn_sched_barrier(0);

        auto mainloop = [&](KDataType* __restrict__ k_lds_write_ptr,
                            KDataType* __restrict__ k_lds_read_ptr,
                            KDataType* __restrict__ v_lds_write_ptr,
                            KDataType* __restrict__ v_lds_read_ptr) {
            FmhaTdmPreparedKRead::Address qk_stage0_base{};

            decltype(load_tile_transpose(v_lds_read_window)) v_tile;
            [[maybe_unused]] float pv_output_scale_m0 = 1.0f;
            [[maybe_unused]] float pv_output_scale_m1 = 1.0f;
            v_lds_read_window.set_bottom_tensor_view_data_ptr(v_lds_read_ptr);

            auto run_fragment_pv_stage = [&](auto stage,
                                             const auto& a_block_tensor,
                                             const auto& b_block_tensor,
                                             auto& next_b_block_tensor,
                                             const auto& b_lds_window) {
                using BlockGemm   = remove_cvref_t<decltype(gemm_1)>;
                using WarpGemm    = typename BlockGemm::WarpGemm;
                using AWarpDstr   = typename WarpGemm::AWarpDstr;
                using BWarpDstr   = typename WarpGemm::BWarpDstr;
                using AWarpTensor = typename WarpGemm::AWarpTensor;
                using BWarpTensor = typename WarpGemm::BWarpTensor;
                using CWarpTensor = typename WarpGemm::CWarpTensor;
                constexpr auto a_warp_y_lengths =
                    to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                constexpr auto b_warp_y_lengths =
                    to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
                constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};

                auto rescale_fragment = [&](auto ordinal) {
                    Policy::template RescaleOutputFragment<decltype(ordinal)::value>(
                        o_acc, pv_output_scale_m0, pv_output_scale_m1);
                };
                Schedule::template RunPvStagePrelude<Mode, decltype(stage)::value>(
                    [&](auto ordinal) { rescale_fragment(ordinal); });

                auto emit_wmma = [&](auto, auto wmma) {
                    constexpr index_t ordinal     = decltype(wmma)::value;
                    constexpr index_t d_msb       = ordinal / Policy::OutputFragments::kNumN;
                    constexpr index_t n           = ordinal % Policy::OutputFragments::kNumN;
                    constexpr index_t m_iter      = d_msb / 2;
                    constexpr index_t v_msb       = d_msb % 2;
                    constexpr index_t full_n_iter = n * 2 + v_msb;

                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, m_iter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, full_n_iter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer().template set_as<fp32x8_t>(number<0>{},
                                                                                o_acc.at(wmma));
                    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    o_acc.at(wmma) =
                        c_warp_tensor.get_thread_buffer().template get_as<fp32x8_t>(number<0>{});
                };

                auto emit_rescale = [&](auto ordinal) {
                    __builtin_amdgcn_sched_barrier(0);
                    rescale_fragment(ordinal);
                };

                Policy::template RunPvScheduledStageWithWmma<decltype(stage)::value, Mode>(
                    next_b_block_tensor, b_lds_window, emit_wmma, emit_rescale);
            };

            auto run_pv = [&](const auto& p_tile) {
                if constexpr(1 < k1_loops)
                {
                    static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                        move_tile_window(v_lds_read_window, {kK1, 0});
                        decltype(v_tile) v_tile_switch;
                        auto p_tile_stage = get_slice_tile(
                            p_tile, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{});
                        run_fragment_pv_stage(
                            i_k1, p_tile_stage, v_tile, v_tile_switch, v_lds_read_window);
                        v_tile = v_tile_switch;
                    });
                    move_tile_window(v_lds_read_window, {-kK1 * (k1_loops - 1), 0});
                }

                auto p_tile_last = get_slice_tile(
                    p_tile, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, k1_loops * kK1>{});
                auto k_lds_su0_read_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                    {0, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                k_lds_su0_read_window.set_bottom_tensor_view_data_ptr(k_lds_read_ptr);
                run_fragment_pv_stage(number<Geometry::kPvStages - 1>{},
                                      p_tile_last,
                                      v_tile,
                                      k_su0_tile,
                                      k_lds_su0_read_window);
            };

            auto begin_iteration = [&]() {
                // Fixed boundary: start the current V TDM transfer.
                auto next_su_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                    {Geometry::kQkSuColumns, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                next_su_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                qk_stage0_base = FmhaTdmPreparedKRead::Prepare<0>(next_su_window);
                __builtin_amdgcn_sched_barrier(0);

                // Reuse the V LDS buffer only after the preceding reads have completed.
                block_sync_lds<k_lds_insts>();
                // V is prefetched one iteration before its consumer.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1 && i_total_loops + 1 < num_total_loop)
                        move_tile_window(v_dram_window,
                                         {physical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(v_dram_window, {kN0, 0});
                v_lds_write_window.set_bottom_tensor_view_data_ptr(v_lds_write_ptr);
                if constexpr(kUseCountdownLoop && kN0 == 128 && kN1 == 128 &&
                             std::is_same_v<VDataType, bf16_t>)
                {
                    v_dram_window.tdm_load_to_lds(
                        tdm_config_v,
                        v_lds_write_window,
                        make_null_tile_window(v_dram_window.get_window_lengths()),
                        number<-1>{},
                        bool_constant<false>{},
                        bool_constant<true>{});
                }
                else
                {
                    load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
                }
            };

            auto run_qk = [&] {
                // STAGE 1, QK gemm
                clear_tile(s_acc); // initialize C
                constexpr auto gemm_0_su  = Policy::template GetQKBlockGemmSu<Problem>();
                auto k_lds_su_read_window = make_tile_window(
                    k_lds_read_view,
                    make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                    {0, 0},
                    Policy::template MakeKSuRegTileDistribution<Problem>());
                k_lds_su_read_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                auto s_acc_su  = gemm_0_su.MakeCBlockTile();
                auto k_su_tile = k_su0_tile;
                static_for<0, Geometry::kQkStages, 1>{}([&](auto i_su) {
                    clear_tile(s_acc_su);
                    if constexpr(decltype(i_su)::value < Geometry::kQkStages - 1)
                    {
                        move_tile_window(k_lds_su_read_window, {Geometry::kQkSuColumns, 0});
                        decltype(k_su_tile) k_su_tile_next;

                        Policy::template RunQkScheduledStage<decltype(i_su)::value, Mode>(
                            gemm_0_su,
                            s_acc_su,
                            q_tile,
                            k_su_tile,
                            k_su_tile_next,
                            k_lds_su_read_window,
                            &qk_stage0_base);

                        k_su_tile = k_su_tile_next;
                    }
                    else
                    {
                        Policy::template RunQkScheduledStage<Geometry::kQkStages - 1, Mode>(
                            gemm_0_su, s_acc_su, q_tile, k_su_tile, v_tile, v_lds_read_window);
                    }
                    set_slice_tile(
                        s_acc,
                        s_acc_su,
                        sequence<0, decltype(i_su)::value * Geometry::kQkSuColumns>{},
                        sequence<kM0, (decltype(i_su)::value + 1) * Geometry::kQkSuColumns>{});
                });
            };

            auto process_score = [&]() {
                // QK scores are ready for bias, masking, and softmax.
                // STAGE 2: scale_s, add bias (prefill path, mirrors baseline)
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                    const auto bias_tile = load_tile(bias_dram_window);
                    tile_elementwise_inout(
                        [](auto& x, const auto& y) {
                            x += log2e_v<SaccDataType> * type_convert<SaccDataType>(y);
                        },
                        s_acc,
                        bias_tile);
                }
                else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    const auto current_k_origin = [&]() {
                        const bool in_sink = (num_sink_loop > i_total_loops);
                        if(in_sink)
                            return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                        else
                            return make_tuple(
                                kN0 * (i_total_loops - num_sink_loop) + physical_seqlen_k_start, 0);
                    }();
                    constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                    sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                            const auto tile_idx = get_x_indices_from_distributed_indices(
                                s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = current_k_origin.at(I0) + tile_idx.at(number<1>{});
                            constexpr auto i_j_idx = make_tuple(idx0, idx1);
                            s_acc(i_j_idx) *= scale_s;
                            position_encoding.update(s_acc(i_j_idx), row, col);
                        });
                    });
                }

                constexpr bool kDeferTensorReady = Mode::kDeferTensorReady;
                if constexpr(!kDeferTensorReady)
                    s_wait_tensorcnt_barrier<Policy::kVPrefetchTensorCount>();

                // Sink-aware k_origin (prefill path)
                const auto k_origin = [&]() {
                    if constexpr(kUseCountdownLoop)
                    {
                        // V's window already points one prefetched tile ahead.
                        return make_tuple(v_dram_window.get_window_origin().at(I0) - kN0, 0);
                    }
                    const bool in_sink_phase = (num_sink_loop > i_total_loops);
                    if(in_sink_phase)
                        return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                    else
                        return make_tuple(
                            kN0 * (i_total_loops - num_sink_loop) + physical_seqlen_k_start, 0);
                }();
                if constexpr(kUseCountdownLoop)
                {
                    bool need_split_check = false;
                    if constexpr(kHasUnevenSplits)
                    {
                        const bool needs_tail_predicate =
                            k_origin.at(I0) + kN0 > physical_seqlen_k_end;
                        need_split_check = remaining_loops == 1 && needs_tail_predicate;
                    }
                    bool need_mask_check = false;
                    if constexpr(kPadSeqLenK)
                    {
                        need_mask_check = mask.IsEdgeTile(
                            q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                    }

                    if(__builtin_expect(need_split_check || need_mask_check, false))
                    {
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                return (need_split_check && physical_seqlen_k_end <= col) ||
                                       (need_mask_check && mask.IsOutOfBound(row, col));
                            });
                    }
                }
                else
                {
                    if constexpr(kHasUnevenSplits)
                    {
                        const bool needs_tail_predicate =
                            k_origin.at(I0) + kN0 > physical_seqlen_k_end;
                        const bool is_last_loop = kUseCountdownLoop
                                                      ? remaining_loops == 1
                                                      : i_total_loops == (num_total_loop - 1);
                        if(is_last_loop && needs_tail_predicate)
                        {
                            set_tile_if(
                                s_acc,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&, physical_seqlen_k_end_ = physical_seqlen_k_end](auto tile_idx) {
                                    const auto col = k_origin.at(I0) + tile_idx.at(I1);

                                    {
                                        return physical_seqlen_k_end_ <= col;
                                    }
                                });
                        }
                    }

                    if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                    {
                        bool need_perpixel_check = mask.IsEdgeTile(
                            q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                        if(need_perpixel_check)
                        {
                            set_tile_if(
                                s_acc,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&](auto tile_idx) {
                                    const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                    const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                    if constexpr(kHasSink)
                                        return mask.IsOutOfSinkBound(row, col);
                                    else if constexpr(Policy::kUseUnsignedCausalMask &&
                                                      std::is_same_v<
                                                          FmhaMask,
                                                          SimplifiedGenericAttentionMask<true>>)
                                        return mask.IsOutOfBoundUnsigned(row, col);
                                    else
                                        return mask.IsOutOfBound(row, col);
                                });
                        }
                    }
                }

                // Sink->normal window jump (prefill path)
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1)
                    {
                        move_tile_window(bias_dram_window,
                                         {0, physical_seqlen_k_start - sink_seq_end});
                    }
                }

                // move bias window (prefill path)
                move_tile_window(bias_dram_window, {0, kN0});
            };

            auto run_softmax = [&] {
                // Gemm1
                auto s_new = cast_tile<SMPLComputeDataType>(s_acc); // S{j}

                auto p_tile = [&]() {
                    static_assert(kNWarp == 1);
                    static_assert(std::is_same_v<SaccDataType, float> &&
                                  std::is_same_v<SMPLComputeDataType, float>);
                    float delta_m0;
                    float delta_m1;
                    Schedule::template RunSplitSoftmax<Mode>(
                        [&]() {
                            Policy::template RunSplitSoftmaxPart01<Problem>(
                                s_new, m, scale_s, delta_m0, delta_m1);
                        },
                        [&]() { s_wait_tensorcnt_barrier<Policy::kVPrefetchTensorCount>(); },
                        [&](auto ordinal) {
                            const float scale_m0 = Mode::kEarlyOutputRescale
                                                       ? SplitSoftmax::Exp2(delta_m0)
                                                       : pv_output_scale_m0;
                            const float scale_m1 = Mode::kEarlyOutputRescale
                                                       ? SplitSoftmax::Exp2(delta_m1)
                                                       : pv_output_scale_m1;
                            Policy::template RescaleOutputFragment<decltype(ordinal)::value>(
                                o_acc, scale_m0, scale_m1);
                        },
                        [&](auto) {
                            Policy::template RunSplitSoftmaxPart2AndGetScale<Problem>(
                                s_new,
                                m,
                                l,
                                scale_s,
                                delta_m0,
                                delta_m1,
                                pv_output_scale_m0,
                                pv_output_scale_m1);
                        });
                    return MakePForGemm1<decltype(gemm_1)>(s_new);
                }();
                return p_tile;
            };

            auto load_next_k = [&]() {
                // Fixed boundary: the next K TDM transfer follows softmax.
                block_sync_lds<v_lds_insts>();
                // The K transfer here feeds iteration i+2, beyond the already prefetched K1.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 2 && i_total_loops + 2 < num_total_loop)
                        move_tile_window(k_dram_window,
                                         {physical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(k_dram_window, {kN0, 0});
                k_lds_write_window.set_bottom_tensor_view_data_ptr(k_lds_write_ptr);
                load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            };

            auto advance_iteration = [&]() {
                // Fixed boundary: wait for the next K tile before advancing.
                s_wait_tensorcnt_barrier<Policy::kKPrefetchTensorCount>();
            };

            auto tile_ops = MakeFmhaTdmSchedTileOps(begin_iteration,
                                                    run_qk,
                                                    process_score,
                                                    run_softmax,
                                                    load_next_k,
                                                    run_pv,
                                                    advance_iteration);
            Schedule::Run(tile_ops);
        }; // mainloop

        const index_t num_pipeline_loop = num_total_loop;
        do
        {
            bool is_even_loop    = kUseCountdownLoop ? even_buffer : i_total_loops % 2 == 0;
            auto k_lds_write_ptr = is_even_loop ? static_cast<KDataType* __restrict__>(smem_ptrk0)
                                                : static_cast<KDataType* __restrict__>(smem_ptrk1);
            auto k_lds_read_ptr  = is_even_loop ? static_cast<KDataType* __restrict__>(smem_ptrk1)
                                                : static_cast<KDataType* __restrict__>(smem_ptrk0);
            auto v_lds_write_ptr = is_even_loop ? static_cast<VDataType* __restrict__>(smem_ptrv1)
                                                : static_cast<VDataType* __restrict__>(smem_ptrv0);
            auto v_lds_read_ptr  = is_even_loop ? static_cast<VDataType* __restrict__>(smem_ptrv0)
                                                : static_cast<VDataType* __restrict__>(smem_ptrv1);
            mainloop(k_lds_write_ptr, k_lds_read_ptr, v_lds_write_ptr, v_lds_read_ptr);
            if constexpr(kUseCountdownLoop)
            {
                even_buffer = !even_buffer;
                --remaining_loops;
            }
            else
            {
                i_total_loops++;
            }
        } while(kUseCountdownLoop ? remaining_loops > 0 : i_total_loops < num_pipeline_loop);

        // Reconstruct the CK tensor only at the final normalization boundary.
        auto output = Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
        constexpr auto o_spans = decltype(output)::get_distributed_spans();

        auto finalize_real_tokens = [&]() {
            if constexpr(kStoreLSE)
            {
                // store lse acc
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
                sweep_tile_span(lse_acc_spans[I0], [&, m_ = m, l_ = l](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse_acc(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                    }
                });

                store_tile(lse_acc_dram_window_tmp, lse_acc);
            }

            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp       = [&]() {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 FmhaMask::IsMasking)
                    {
                        return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                    }
                    else
                        return 1 / l[i_idx];
                }();
                sweep_tile_span(o_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    output(i_j_idx) *= tmp;
                });
            });
        };

        if constexpr(kEnableVirtualSink && BiasEnum == BlockAttentionBiasEnum::NO_BIAS)
        {
            if(__builtin_isinf_sign(sink_v) != -1)
            {
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
                const float sink_natural = sink_v * scale_s;
                const float sink_log2    = sink_natural * C_LOG2E;
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    float output_scale;
                    float merged_lse;
                    if(__builtin_isnan(sink_natural))
                    {
                        output_scale = sink_natural;
                        merged_lse   = sink_natural;
                    }
                    else if(l[i_idx] == 0.f || __builtin_isinf_sign(sink_natural) > 0)
                    {
                        output_scale = 0.f;
                        merged_lse   = sink_natural;
                    }
                    else
                    {
                        const float real_log2   = m[i_idx] * scale_s;
                        const float merged_max  = max(real_log2, sink_log2);
                        const float real_weight = SplitSoftmax::Exp2(real_log2 - merged_max);
                        const float sink_weight = SplitSoftmax::Exp2(sink_log2 - merged_max);
                        const float merged_sum = __builtin_fmaf(l[i_idx], real_weight, sink_weight);
                        output_scale           = real_weight / merged_sum;
                        merged_lse             = merged_max / C_LOG2E + log(merged_sum);
                    }
                    if constexpr(kStoreLSE)
                        lse_acc(i_idx) = merged_lse;
                    sweep_tile_span(o_spans[I1], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        // Explicit zeros also cover empty rows and an infinite sink.
                        output(i_j_idx) =
                            output_scale == 0.f ? 0.f : output(i_j_idx) * output_scale;
                    });
                });
                if constexpr(kStoreLSE)
                    store_tile(lse_acc_dram_window_tmp, lse_acc);
            }
            else
            {
                finalize_real_tokens();
            }
        }
        else
        {
            finalize_real_tokens();
        }

        Schedule::template RunEpilogue<Mode>(
            [&]() {
                // A partial prefetch wait can leave the terminal look-ahead TDM in flight.
                s_wait_tensorcnt<0>();
            },
            [&]() {
                // The final PV stage's next-K reads have no consumer in the last iteration.
                s_wait_dscnt<0>();
            });

        return output;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        float sink_v,
                                        void* smem_ptrk0,
                                        void* smem_ptrk1,
                                        void* smem_ptrv0,
                                        void* smem_ptrv1) const
    {
        return run(q_dram_block_window_tmp,
                   k_dram_block_window_tmp,
                   v_dram_block_window_tmp,
                   bias_dram_block_window_tmp,
                   lse_acc_dram_window_tmp,
                   mask,
                   position_encoding,
                   scale_s,
                   smem_ptrk0,
                   smem_ptrk1,
                   smem_ptrv0,
                   smem_ptrv1,
                   sink_v);
    }
};

} // namespace ck_tile
