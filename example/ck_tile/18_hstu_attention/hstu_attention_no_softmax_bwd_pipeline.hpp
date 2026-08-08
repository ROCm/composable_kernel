// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// HSTU attention backward MAIN pipeline — SiLU path (kUseSoftmax = false).
//
// Structural blueprint: FMHA BlockFmhaBwdDQDKDVPipelineKRKTRVR (kr_ktr_vr).
// Reuses the FMHA default policy verbatim (5 GEMM shapes + all LDS / shuffle
// descriptors), so the KV-resident register layout, the Q/dO/dS LDS staging, and
// the shuffle/early-exit machinery are identical to FMHA. The HSTU differences
// (DESIGN §2.2/§2.3) are confined to:
//   STAGE2: s_acc *= alpha;  p = silu(s_acc)*scale_p (→dV);  g = scale_p*dsilu(s_acc) (→dS)
//           (no LSE/softmax; no D load)
//   STAGE5: ds = dp_acc * g    (g already carries scale_p)
//   final : dq_acc *= alpha, dk_acc *= alpha   (dv_acc unscaled)   [FMHA raw_scale slot]
//
// The LDS pointer-offset arithmetic is kept byte-identical to FMHA (it still
// reserves the LSE/D/bias regions in GetSmemSize) so the proven layout is reused
// unchanged; those regions are simply never written/read on the SiLU path. Trimming
// them to recover LDS is a post-M1 perf item (DESIGN §8.2-R6).
//
// M1 scope: no-mask only. The mask object is still consulted for GetTileRangeAlongY
// (called unconditionally, P1-D); GenericAttentionMask<false> returns (0, seqlen_q)
// and IsMasking=false compiles out the edge-tile masking. The 5-factor mask
// (set_tile_if on p,g) is M2.
template <typename Problem_, typename Policy_ = BlockFmhaBwdPipelineDefaultPolicy>
struct HstuAttentionBwdDQDKDVPipelineKRKTRVR
{
    using Problem  = remove_cvref_t<Problem_>;
    using Policy   = remove_cvref_t<Policy_>;

    using QDataType        = remove_cvref_t<typename Problem::QDataType>;
    using KDataType        = remove_cvref_t<typename Problem::KDataType>;
    using VDataType        = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType     = remove_cvref_t<typename Problem::GemmDataType>;
    using AccDataType      = remove_cvref_t<typename Problem::AccDataType>;
    using OGradDataType    = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType    = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType    = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType    = remove_cvref_t<typename Problem::VGradDataType>;
    // dummy typedefs kept so default policy compiles (P1-A); SiLU path unused.
    using BiasDataType     = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType      = remove_cvref_t<typename Problem::LSEDataType>;
    using DDataType        = remove_cvref_t<typename Problem::DDataType>;
    using FmhaMask         = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    // computation type for the non-linear SiLU/dsilu (matches HSTU fwd CompDataType)
    using CompDataType = AccDataType;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    // ck_tile bwd: headdim pad is index_t (0/8/1); seqlen is never padded (OOB via buffer_load)
    static constexpr index_t kPadHeadDimQ  = Problem::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum         = Problem::BiasEnum;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;
    static constexpr bool kUseTrLoad       = Problem::kUseTrLoad;
    static_assert(!kUseTrLoad, "HSTU SiLU bwd M1 uses the non-trload kr_ktr_vr path");

    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentQGrad = 1;
    static constexpr index_t kAlignmentKGrad =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentKGrad<Problem>();
    static constexpr index_t kAlignmentVGrad =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentVGrad<Problem>();

    static constexpr const char* name = "hstu_silu_kr_ktr_vr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
                                        const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
                                        FmhaMask mask,
                                        float alpha,
                                        float scale_p,
                                        void* smem_ptr) const
    {
        // Block GEMM (identical to FMHA)
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
        auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};

        // K, HBM -> LDS -> Reg
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             k_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());

        const auto k_origin = k_dram_window.get_window_origin();
        const auto [seqlen_q_start, seqlen_q_end] =
            mask.GetTileRangeAlongY(k_origin.at(number<0>{}), number<kN0>{}, number<kM0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0);

        if constexpr(FmhaMask::IsMasking)
        {
            if(num_total_loop <= 0)
            {
                return make_tuple(dk_acc, dv_acc);
            }
        }
        KDataType* k_lds_ptr =
            static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());

        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKRegBlockDescriptor<Problem>());

        auto k_reg_tensor = make_static_distributed_tensor<KDataType>(
            Policy::template MakeKRegBlockDescriptor<Problem>());

        // V, HBM -> LDS -> Reg
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             v_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeVDramTileDistribution<Problem>());

        VDataType* v_lds_ptr =
            static_cast<VDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kK2>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegBlockDescriptor<Problem>());

        // KT, Reg -> LDS -> Reg
        auto shuffled_k_block_tile = make_static_distributed_tensor<KDataType>(
            Policy::template MakeShuffledKRegWriteBlockDescriptor<Problem>());

        KDataType* kt_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));

        auto shuffled_k_lds_write = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeShuffledKLdsWriteBlockDescriptor<Problem>());

        auto shuffled_k_lds_write_window = make_tile_window(
            shuffled_k_lds_write, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto kt_lds_read = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeKTLdsReadBlockDescriptor<Problem>());

        auto kt_lds_read_window =
            make_tile_window(kt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

        // Pre-Load KV into Registers
        auto k_block_tile = load_tile(k_dram_window);
        auto v_block_tile = load_tile(v_dram_window);

        store_tile(k_lds_write_window, k_block_tile);
        shuffle_tile(shuffled_k_block_tile, k_block_tile);
        store_tile(shuffled_k_lds_write_window, shuffled_k_block_tile);

        block_sync_lds();
        k_reg_tensor = load_tile(k_lds_read_window);
        block_sync_lds();

        auto kt_reg_tensor = load_tile(kt_lds_read_window);

        store_tile(v_lds_write_window, v_block_tile);

        block_sync_lds();

        auto v_reg_tensor = load_tile(v_lds_read_window);
        block_sync_lds();

        // Q: HBM -> Reg -> LDS
        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());

        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>()));

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakePTRegSliceBlockDescriptor<Problem>());
        // QT: Reg -> Reg -> LDS
        auto shuffled_q_block_tile = make_static_distributed_tensor<QDataType>(
            Policy::template MakeShuffledQRegWriteBlockDescriptor<Problem>());

        QDataType* qt_lds_ptr =
            static_cast<QDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto shuffled_q_lds_write = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeShuffledQLdsWriteBlockDescriptor<Problem>());

        auto shuffled_q_lds_write_window = make_tile_window(
            shuffled_q_lds_write, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto qt_lds_read = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeQTLdsReadBlockDescriptor<Problem>());

        auto qt_lds_read_window =
            make_tile_window(qt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeQTRegSliceBlockDescriptor<Problem>());

        // dO: HBM -> Reg -> LDS
        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>()));

        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());

        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto do_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());
        // dOT: Reg -> Reg -> LDS
        auto shuffled_do_block_tile = make_static_distributed_tensor<OGradDataType>(
            Policy::template MakeShuffledOGradRegWriteBlockDescriptor<Problem>());

        OGradDataType* dot_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>()));

        auto shuffled_do_lds_write = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeShuffledOGradLdsWriteBlockDescriptor<Problem>());

        auto shuffled_do_lds_write_window = make_tile_window(
            shuffled_do_lds_write, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto dot_read_lds = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsReadBlockDescriptor<Problem>());

        auto dot_lds_read_window =
            make_tile_window(dot_read_lds,
                             make_tuple(number<kVHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeOGradTRegSliceBlockDescriptor<Problem>());

        // dS: Reg -> Reg -> LDS  (offset kept identical to FMHA, incl. reserved LSE/D regions)
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto ds_lds = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());

        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto ds_lds_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK4>{}),
                             ds_lds_window.get_window_origin(),
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());

        auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());

        // dQ write out (atomic_add into float dq_acc; window provided by kernel)
        auto dq_dram_window = make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dq_dram_block_window_tmp.get_window_lengths(),
                                               {seqlen_q_start, 0});

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType  = decltype(gemm_4.MakeCBlockTile());

        index_t i_total_loops = 0;
        index_t seqlen_q_step = seqlen_q_start;
        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kM0 == kK1, "kM0 should equal to kK1");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        static_assert(kM0 == kK3, "kM0 should equal to kK3");
        constexpr index_t k4_loops = kN0 / kK4;

        clear_tile(dv_acc);
        clear_tile(dk_acc);

        // sigmoid/silu/dsilu (fp32 compute, matches HSTU fwd f_silu)
        const auto f_sigmoid = [](CompDataType x) {
            const auto one = type_convert<CompDataType>(1.0f);
            if constexpr(std::is_same_v<CompDataType, float>)
                return one * __builtin_amdgcn_rcpf(one + __expf(-x));
            else
                return one / (one + exp(-x));
        };

        __builtin_amdgcn_sched_barrier(0);
        // Hot loop over Q tiles
        while(i_total_loops < num_total_loop)
        {
            auto q_block_tile = load_tile(q_dram_window);
            move_tile_window(q_dram_window, {kM0, 0});

            store_tile(q_lds_window, q_block_tile);
            shuffle_tile(shuffled_q_block_tile, q_block_tile);
            store_tile(shuffled_q_lds_write_window, shuffled_q_block_tile);

            block_sync_lds();

            auto q_reg_tensor = load_tile(q_lds_read_window);

            block_sync_lds();

            // STAGE 1, Q@K Gemm0 (unscaled)
            auto s_acc = SPBlockTileType{};
            s_acc      = gemm_0(q_reg_tensor, k_reg_tensor);

            // STAGE 2, HSTU SiLU: s = alpha*S; p = silu(s)*scale_p; g = scale_p*dsilu(s)
            auto p                 = SPBlockTileType{};
            auto g                 = SPBlockTileType{};
            constexpr auto p_spans = decltype(p)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    const CompDataType s   = alpha * type_convert<CompDataType>(s_acc[i_j_idx]);
                    const CompDataType sig = f_sigmoid(s);
                    const CompDataType silu = s * sig;
                    const CompDataType dsilu =
                        sig * (type_convert<CompDataType>(1.0f) +
                               s * (type_convert<CompDataType>(1.0f) - sig));
                    p(i_j_idx) = silu * scale_p;
                    g(i_j_idx) = scale_p * dsilu;
                });
            });

            // STAGE 2 (M2): masked-out positions must be EXPLICITLY zeroed on SiLU path
            // (silu(0)*scale_p=0 but dsilu(0)=0.5 != 0; -inf is forbidden -> NaN). On edge
            // tiles, clear p & g where !IsTokenPairInsideMask. ds=dp*g then auto-zeros (g=0),
            // so dV/dK/dQ get no contribution from masked pairs (matches reference dS=0).
            //
            // P1-1 fix: gate on the RUNTIME mask.IsEdgeTile, NOT compile-time
            // FmhaMask::IsMasking. NoLocal sets IsMasking=kUseCausal, so with causal=0 the
            // old `if constexpr(IsMasking)` deleted this block entirely -- yet num_target>0
            // still requires masking the target region (max_uih_len = seqlen - num_target <
            // seqlen), and IsFullTileInsideMask already flags those tiles as edge for both
            // causal branches. The reference applies IsTokenPairInsideMask unconditionally
            // (reference_hstu_attention_bwd.hpp:671), and the fwd pipeline likewise checks
            // at runtime; this aligns bwd with both. Pure no-mask (causal=0, no factors)
            // stays cheap: IsEdgeTile is false for fully-inside tiles -> no per-pixel sweep
            // (tile-divisible no-mask flags zero edge tiles).
            if(mask.IsEdgeTile(
                   seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{}))
            {
                auto is_masked_out = [&](auto tile_idx) {
                    const int row = seqlen_q_step + tile_idx.at(number<0>{});
                    const int col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    return !mask.IsTokenPairInsideMask(row, col);
                };
                set_tile_if(p, type_convert<AccDataType>(0.0f), is_masked_out);
                set_tile_if(g, type_convert<AccDataType>(0.0f), is_masked_out);
            }

            const auto p_gemm = cast_tile<GemmDataType>(p);

            // STAGE 3, P^T @ dO^T  Gemm1 -> dV
            auto do_block_tile = load_tile(do_dram_window);
            move_tile_window(do_dram_window, {kM0, 0});

            store_tile(do_lds_window, do_block_tile);
            shuffle_tile(shuffled_do_block_tile, do_block_tile);
            store_tile(shuffled_do_lds_write_window, shuffled_do_block_tile);

            block_sync_lds();

            auto dot_reg_tensor = load_tile(dot_lds_read_window);

            block_sync_lds();

            Policy::template PTFromGemm0CToGemm1A<Problem,
                                                  decltype(pt_reg_tensor),
                                                  decltype(p_gemm)>(pt_reg_tensor, p_gemm);
            gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);

            // STAGE 4, dO @ V  Gemm2 -> dP
            auto do_reg_tensor = load_tile(do_lds_read_window);
            block_sync_lds();

            auto dp_acc = SPGradBlockTileType{};
            dp_acc      = gemm_2(do_reg_tensor, v_reg_tensor);

            // STAGE 5, dS = dP * g   (g already carries scale_p)
            auto ds                 = SPGradBlockTileType{};
            constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
            sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ds(i_j_idx)            = dp_acc[i_j_idx] * g[i_j_idx];
                });
            });

            // STAGE 6, dS^T @ Q^T  Gemm3 -> dK
            auto qt_reg_tensor = load_tile(qt_lds_read_window);
            block_sync_lds();

            const auto ds_gemm = cast_tile<GemmDataType>(ds);

            Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                      decltype(dst_reg_tensor),
                                                      decltype(ds_gemm)>(dst_reg_tensor, ds_gemm);

            gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);

            store_tile(ds_lds_window, ds_gemm);

            block_sync_lds();

            auto ds_reg_tensor      = load_tile(ds_lds_read_window);
            auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
            move_tile_window(ds_lds_read_window, {0, kK4});

            // STAGE 7, dS @ K^T  Gemm4 -> dQ
            auto dq_acc = QGradBlockTileType{};
            clear_tile(dq_acc);

            static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor_next = load_tile(ds_lds_read_window);
                    move_tile_window(ds_lds_read_window, {0, kK4});
                }
                auto kt_reg_tensor_slice = get_slice_tile(kt_reg_tensor,
                                                          sequence<0, i_k4 * kK4>{},
                                                          sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);

                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
                }
            });
            move_tile_window(ds_lds_read_window, {0, -kN0});

            // dQ scale by alpha (FMHA raw_scale slot)
            tile_elementwise_inout([&alpha](auto& x) { x = x * alpha; }, dq_acc);

            if constexpr(kIsDeterministic)
            {
                store_tile(dq_dram_window, dq_acc);
            }
            else
            {
                update_tile(dq_dram_window, dq_acc); // atomic_add into float dq_acc
            }
            move_tile_window(dq_dram_window, {kM0, 0});

            i_total_loops += 1;
            seqlen_q_step += kM0;
        }

        // dK scale by alpha; dV unscaled
        tile_elementwise_inout([&alpha](auto& x) { x = x * alpha; }, dk_acc);

        return make_tuple(dk_acc, dv_acc);
    }
};

} // namespace ck_tile
