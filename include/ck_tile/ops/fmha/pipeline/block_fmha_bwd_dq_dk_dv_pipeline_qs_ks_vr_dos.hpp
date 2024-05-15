// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_qs_ks_vr_dos_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdDQDKDVPipelineQSKSVROGradSDefaultPolicy>
struct BlockFmhaBwdDQDKDVPipelineQSKSVROGradS
{
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType          = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType             = remove_cvref_t<typename Problem::DDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType         = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType         = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType         = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType         = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType      = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

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

    static constexpr bool kQLoadOnce      = true;
    static constexpr bool kQTLoadOnce     = false;
    static constexpr bool kKLoadOnce      = true;
    static constexpr bool kKTLoadOnce     = false;
    static constexpr bool kVLoadOnce      = true;
    static constexpr bool kOGradLoadOnce  = true;
    static constexpr bool kOGradTLoadOnce = false;

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum     = Problem::BiasEnum;
    static constexpr bool kHasBiasGrad = Problem::kHasBiasGrad;
    static constexpr bool kHasDropout  = Problem::kHasDropout;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentQGrad =
        kPadHeadDimQ ? 2 : Policy::template GetAlignmentQGrad<Problem>();
    static constexpr index_t kAlignmentKGrad =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentKGrad<Problem>();
    static constexpr index_t kAlignmentVGrad =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentVGrad<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetTransposedAlignmentBias<Problem>();

    static constexpr const char* name = "qs_ks_vr_dos";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename QTDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename KTDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename OGradTDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp,
              typename BiasGradDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const QTDramBlockWindowTmp& /*qt_dram_block_window_tmp*/,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const KTDramBlockWindowTmp& /*kt_dram_block_window_tmp*/,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
               const OGradTDramBlockWindowTmp& /*dot_dram_block_window_tmp*/,
               const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
               const DDramBlockWindowTmp& d_dram_block_window_tmp,
               const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float raw_scale,
#if CK_TILE_FMHA_FWD_FAST_EXP2
               float scale,
#endif
               float rp_undrop,
               float scale_rp_undrop,
               void* smem_ptr,
               BlockDropout& dropout) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<LSEDataType,
                               remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QGradDataType,
                               remove_cvref_t<typename QGradDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // Q tile in LDS
        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));
        auto q_lds           = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        // QT tile in LDS
        auto qt_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptorAsQT<Problem>());
        auto qt_lds_window =
            make_tile_window(qt_lds, make_tuple(number<kQKHeaddim>{}, number<kM0>{}), {0, 0});

        // K tile in LDS
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<KDataType*>(smem_ptr),
            Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        // KT tile in LDS
        auto kt_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<KDataType*>(smem_ptr),
            Policy::template MakeKLdsBlockDescriptorAsKT<Problem>());
        auto kt_lds_window =
            make_tile_window(kt_lds, make_tuple(number<kQKHeaddim>{}, number<kN0>{}), {0, 0});

        // OGrad tile in LDS
        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeQ<Problem>()));
        auto do_lds               = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());
        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        // OGradT tile in LDS
        auto dot_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptorAsOGradT<Problem>());
        auto dot_lds_window =
            make_tile_window(dot_lds, make_tuple(number<kVHeaddim>{}, number<kM0>{}), {0, 0});

        // SGrad tile in LDS
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>()));
        auto ds_lds              = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());
        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        // BiasT/BiasGradT tile in LDS, use the same size and layout
        BiasDataType* biast_lds_ptr = static_cast<BiasDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>()));
        auto biast_lds              = make_tensor_view<address_space_enum::lds>(
            biast_lds_ptr, Policy::template MakeBiasTLdsBlockDescriptor<Problem>());
        auto biast_lds_shuffle_window =
            make_tile_window(biast_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});
        auto dbiast_lds_shuffle_window =
            make_tile_window(biast_lds,
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeShuffledBiasTileDistribution<Problem>());

        static_assert(std::is_same_v<BiasDataType, BiasGradDataType>,
                      "BiasDataType and BiasGradDataType should be the same!");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        auto v_dram_window = make_tile_window(
            v_dram_block_window_tmp.get_bottom_tensor_view(),
            v_dram_block_window_tmp.get_window_lengths(),
            v_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeVInRegDramTileDistribution<Problem, decltype(gemm_2)>());

        auto v = load_tile(v_dram_window); // persistent V register tile

        using SPTBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradTBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType   = decltype(gemm_4.MakeCBlockTile());

        // init VGrad & KGrad
        auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
        auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};

        clear_tile(dv_acc);
        clear_tile(dk_acc);

        auto k_dram_window = make_tile_window(
            k_dram_block_window_tmp.get_bottom_tensor_view(),
            k_dram_block_window_tmp.get_window_lengths(),
            k_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                    // load

        __builtin_amdgcn_sched_barrier(0);
        const auto k_origin = k_dram_window.get_window_origin();
        const auto [seqlen_q_start, seqlen_q_end] =
            mask.GetTileRangeAlongY(k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0);

        // check early exit if masked and no work to do.
        if constexpr(FmhaMask::IsMasking)
        {
            if(num_total_loop <= 0)
            {
                // Note: here dk_acc&dv_acc are all cleard, return it
                // Note: v loaded but no fence, ignore it.
                return ck_tile::make_tuple(dk_acc, dv_acc);
            }
        }

        auto k_block_tile = load_tile(k_dram_window);

        store_tile(k_lds_window, k_block_tile); // // persistent K in LDS

        auto q_dram_block_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0});

        auto do_dram_block_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0});

        auto dq_dram_block_window =
            make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                             dq_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0});

        auto lse_dram_block_window =
            make_tile_window(lse_dram_block_window_tmp.get_bottom_tensor_view(),
                             lse_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start});

        auto d_dram_block_window =
            make_tile_window(d_dram_block_window_tmp.get_bottom_tensor_view(),
                             d_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start});

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_block_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, bias_origin.at(number<1>{})}); // M/N

        const auto dbias_origin = dbias_dram_block_window_tmp.get_window_origin();
        auto dbias_dram_block_window =
            make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                             dbias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, dbias_origin.at(number<1>{})}); // M/N

        auto lse_dram_window = make_tile_window(
            lse_dram_block_window.get_bottom_tensor_view(),
            lse_dram_block_window.get_window_lengths(),
            lse_dram_block_window.get_window_origin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        auto d_dram_window = make_tile_window(
            d_dram_block_window.get_bottom_tensor_view(),
            d_dram_block_window.get_window_lengths(),
            d_dram_block_window.get_window_origin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        auto bias_dram_window =
            make_tile_window(bias_dram_block_window.get_bottom_tensor_view(),
                             bias_dram_block_window.get_window_lengths(),
                             bias_dram_block_window.get_window_origin(),
                             Policy::template MakeBiasTileDistribution<Problem>());

        auto biast_lds_window =
            make_tile_window(biast_lds_shuffle_window.get_bottom_tensor_view(),
                             biast_lds_shuffle_window.get_window_lengths(),
                             biast_lds_shuffle_window.get_window_origin(),
                             Policy::template MakeBiasTTileDistribution<decltype(gemm_0)>());

        auto randval_dram_window = dropout.MakeRandvalDramWindow<decltype(gemm_0), false>(
            randval_dram_block_window_tmp, seqlen_q_start);

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kM0 / kK1;
        constexpr index_t k2_loops = kVHeaddim / kK2;
        constexpr index_t k3_loops = kM0 / kK3;
        constexpr index_t k4_loops = kN0 / kK4;
        do
        {
            auto q_dram_window = make_tile_window(
                q_dram_block_window.get_bottom_tensor_view(),
                q_dram_block_window.get_window_lengths(),
                q_dram_block_window.get_window_origin(),
                Policy::template MakeQDramTileDistribution<Problem>()); // Q DRAM tile window for
                                                                        // load

            auto do_dram_window = make_tile_window(
                do_dram_block_window.get_bottom_tensor_view(),
                do_dram_block_window.get_window_lengths(),
                do_dram_block_window.get_window_origin(),
                Policy::template MakeOGradDramTileDistribution<Problem>()); // OGrad DRAM tile
                                                                            // window for load

            // STAGE 1, Q@K Gemm0
            auto st_acc = SPTBlockTileType{};

            auto q_block_tile = load_tile(q_dram_window);
            clear_tile(st_acc);                     // Initialize S^T
            store_tile(q_lds_window, q_block_tile); // LDS write

            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }
            const auto bias_tile = load_tile(bias_dram_window); // load bias tile
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }

            if constexpr(k0_loops > 1)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    block_sync_lds();
                    gemm_0(st_acc,
                           get_slice_tile(q_lds_window,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kM0, (i_k0 + 1) * kK0>{}),
                           get_slice_tile(k_lds_window,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kN0, (i_k0 + 1) * kK0>{}));
                    block_sync_lds();
                });
            }

            auto do_block_tile = load_tile(do_dram_window); // prefetch load OGrad tile
            {                                               // tail
                block_sync_lds();
                gemm_0(st_acc,
                       get_slice_tile(q_lds_window,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       get_slice_tile(k_lds_window,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kN0, k0_loops * kK0>{}));
                block_sync_lds();
            }

            // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                block_sync_lds();
                auto bias_shuffle_tmp = make_static_distributed_tensor<BiasDataType>(
                    Policy::template MakeShuffledBiasTileDistribution<Problem>());
                shuffle_tile(bias_shuffle_tmp, bias_tile);
                store_tile(biast_lds_shuffle_window, bias_shuffle_tmp);
                block_sync_lds();
                auto biast_tile = load_tile(biast_lds_window);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        x = raw_scale * x + type_convert<AccDataType>(y);
#else
                        x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
#endif
                    },
                    st_acc,
                    biast_tile);
                move_tile_window(bias_dram_window, {kM0, 0});
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto q_origin     = q_dram_block_window.get_window_origin();
                constexpr auto st_spans = decltype(st_acc)::get_distributed_spans();
                sweep_tile_span(st_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(st_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            st_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        st_acc(i_j_idx) *= raw_scale;
#else
                        st_acc(i_j_idx) *= scale;
#endif
                        position_encoding.update(st_acc(i_j_idx), row, col);
                    });
                });
            }
            else
            {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, st_acc);
#endif
            }

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto q_origin      = q_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(st_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }

            const auto lse = load_tile(lse_dram_window);

            static const auto get_validated_lse = [](LSEDataType raw_lse) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_lse == -numeric<LSEDataType>::infinity()
                               ? type_convert<LSEDataType>(0.f)
                               : raw_lse;
                }
                else
                {
                    return raw_lse;
                }
            };

            auto pt                 = SPTBlockTileType{};
            constexpr auto pt_spans = decltype(pt)::get_distributed_spans();
            sweep_tile_span(pt_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                auto row_lse = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);
#endif
                sweep_tile_span(pt_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        pt(i_j_idx) = exp2(st_acc[i_j_idx] - row_lse);
                    }
                    else
                    {
                        pt(i_j_idx) = exp2(scale * st_acc[i_j_idx] - row_lse);
                    }
#else
                    pt(i_j_idx) = exp(st_acc[i_j_idx] - get_validated_lse(lse[i_idx]));
#endif
                });
            });

            if constexpr(kHasDropout)
            {
                dropout.Run<decltype(gemm_0), RandValOutputDataType>(
                    seqlen_q_start + i_total_loops * kM0, pt, randval_dram_window);
            }

            // STAGE 3, P^T@OGrad^T Gemm1
            block_sync_lds();
            store_tile(do_lds_window, do_block_tile); // store the prefetch

            const auto pt_gemm = [&]() {
                if constexpr(kHasDropout)
                {
                    return tile_elementwise_in(
                        [](const auto& x) { return type_convert<GemmDataType>(x > 0.f ? x : 0.f); },
                        pt);
                }
                else
                {
                    return cast_tile<GemmDataType>(pt);
                }
            }();

            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                block_sync_lds();
                gemm_1(dv_acc,
                       get_slice_tile(
                           pt_gemm, sequence<i_k1 * kK1, 0>{}, sequence<(i_k1 + 1) * kK1, kN0>{}),
                       get_slice_tile(dot_lds_window,
                                      sequence<0, i_k1 * kK1>{},
                                      sequence<kVHeaddim, (i_k1 + 1) * kK1>{}));
                block_sync_lds();
            });

            // STAGE 4, OGrad@V Gemm2
            auto dpt_acc = SPGradTBlockTileType{};
            clear_tile(dpt_acc); // Initialize PGrad^T

            static_for<0, k2_loops, 1>{}([&](auto i_k2) {
                block_sync_lds();
                gemm_2(dpt_acc,
                       get_slice_tile(do_lds_window,
                                      sequence<0, i_k2 * kK2>{},
                                      sequence<kM0, (i_k2 + 1) * kK2>{}),
                       get_slice_tile(
                           v, sequence<0, i_k2 * kK2>{}, sequence<kN0, (i_k2 + 1) * kK2>{}));
                block_sync_lds();
            });

            // STAGE 5, P^T(PGrad^T - D)
            const auto d = load_tile(d_dram_window);

            auto dst                 = SPGradTBlockTileType{};
            constexpr auto dst_spans = decltype(dst)::get_distributed_spans();
            sweep_tile_span(dst_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(dst_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    bool undrop_flag       = pt[i_j_idx] >= 0;
                    dst(i_j_idx) =
                        pt[i_j_idx] *
                        (!kHasDropout || undrop_flag ? (dpt_acc[i_j_idx] - d[i_idx]) : d[i_idx]);
                });
            });

            if constexpr(kHasBiasGrad)
            {
                const auto dbiast = [&]() {
                    if constexpr(kHasDropout)
                    {
                        return tile_elementwise_in(
                            [&rp_undrop](const auto& x) {
                                return type_convert<BiasGradDataType>(x * rp_undrop);
                            },
                            dst);
                    }
                    else
                    {
                        return cast_tile<BiasGradDataType>(dst);
                    }
                }();
                store_tile(biast_lds_shuffle_window, dbiast);
                block_sync_lds();
                auto dbiast_tile        = load_tile(dbiast_lds_shuffle_window);
                auto dbiast_shuffle_tmp = make_static_distributed_tensor<BiasGradDataType>(
                    Policy::template MakeBiasTileDistribution<Problem>());
                shuffle_tile(dbiast_shuffle_tmp, dbiast_tile);
                store_tile(dbias_dram_block_window, dbiast_shuffle_tmp);
                move_tile_window(dbias_dram_block_window, {kM0, 0});
            }

            // STAGE 6, SGrad^T@Q^T Gemm3
            block_sync_lds();
            const auto dst_gemm = cast_tile<GemmDataType>(dst);

            static_for<0, k3_loops, 1>{}([&](auto i_k3) {
                block_sync_lds();
                gemm_3(dk_acc,
                       get_slice_tile(
                           dst_gemm, sequence<i_k3 * kK3, 0>{}, sequence<(i_k3 + 1) * kK3, kN0>{}),
                       get_slice_tile(qt_lds_window,
                                      sequence<0, i_k3 * kK3>{},
                                      sequence<kQKHeaddim, (i_k3 + 1) * kK3>{}));
                block_sync_lds();
            });

            // STAGE 7, SGrad@K^T Gemm4
            store_tile(ds_lds_window, dst_gemm);

            auto dq_acc = QGradBlockTileType{};
            clear_tile(dq_acc); // Initialize QGrad

            block_sync_lds();

            static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                gemm_4(dq_acc,
                       get_slice_tile(ds_lds_window,
                                      sequence<0, i_k4 * kK4>{},
                                      sequence<kM0, (i_k4 + 1) * kK4>{}),
                       get_slice_tile(kt_lds_window,
                                      sequence<0, i_k4 * kK4>{},
                                      sequence<kQKHeaddim, (i_k4 + 1) * kK4>{}));
            });

            // QGrad Scale
            if constexpr(kHasDropout)
            {
                tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                       dq_acc);
            }
            else
            {
                tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            }
            const auto dq = cast_tile<QGradDataType>(dq_acc);
            update_tile(dq_dram_block_window, dq);

            // move tile windows
            move_tile_window(q_dram_block_window, {kM0, 0});
            move_tile_window(dq_dram_block_window, {kM0, 0});
            move_tile_window(do_dram_block_window, {kM0, 0});
            move_tile_window(lse_dram_window, {kM0});
            move_tile_window(d_dram_window, {kM0});
        } while(++i_total_loops < num_total_loop);

        // KGrad Scale
        if constexpr(kHasDropout)
        {
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dk_acc);
        }
        else
        {
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
        }
        // VGrad Scale
        if constexpr(kHasDropout)
        {
            tile_elementwise_inout([&rp_undrop](auto& x) { x = x * rp_undrop; }, dv_acc);
        }

        return ck_tile::make_tuple(dk_acc, dv_acc);
    }
};

} // namespace ck_tile
