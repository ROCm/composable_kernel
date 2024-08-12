// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP
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
    using FmhaDropout           = remove_cvref_t<typename Problem::FmhaDropout>;
    using HotLoopScheduler      = typename Policy::template HotLoopScheduler<Problem>;

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

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ      = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK      = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ     = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV     = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum         = Problem::BiasEnum;
    static constexpr bool kHasBiasGrad     = Problem::kHasBiasGrad;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
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
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetTransposedAlignmentBias<Problem>();

    static constexpr const char* name = "kr_ktr_vr_iglp";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp,
              typename BiasGradDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
               const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
               const DDramBlockWindowTmp& d_dram_block_window_tmp,
               const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float raw_scale,
               float scale,
               float rp_undrop,
               float scale_rp_undrop,
               void* smem_ptr,
               FmhaDropout& dropout) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<LSEDataType,
                               remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
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

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        // init VGrad & KGrad
        auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
        auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};

        // K, HBM ->LDS ->Reg
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             k_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());

        const auto k_origin = k_dram_window.get_window_origin();
        // Early termination
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
                return make_tuple(dk_acc, dv_acc);
            }
        }
        KDataType* k_lds_ptr =
            static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());

        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKRegSliceBlockDescriptor<Problem>());

        auto k_reg_tensor = make_static_distributed_tensor<KDataType>(
            Policy::template MakeKRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // V, HBM ->LDS ->Reg
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
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kK2>{}), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kK2>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegSliceBlockDescriptor<Problem>());

        auto v_reg_tensor = make_static_distributed_tensor<VDataType>(
            Policy::template MakeVRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // KT, Reg ->LDS ->Reg
        auto kt_block_tile = make_static_distributed_tensor<KDataType>(
            Policy::template MakeKTRegWriteBlockDescriptor<Problem>());

        KDataType* kt_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));

        auto kt_lds_write = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeKTLdsWriteBlockDescriptor<Problem>());

        auto kt_lds_write_window =
            make_tile_window(kt_lds_write, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        auto kt_lds_read = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeKTLdsReadBlockDescriptor<Problem>());

        auto kt_lds_read_window =
            make_tile_window(kt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // Pre-Load KV into Registers
        auto k_block_tile = load_tile(k_dram_window);
        auto v_block_tile = load_tile(v_dram_window);

        store_tile(k_lds_write_window, k_block_tile);
        shuffle_tile(kt_block_tile, k_block_tile);
        store_tile(kt_lds_write_window, kt_block_tile);

        block_sync_lds();
        k_reg_tensor = load_tile(k_lds_read_window);
        block_sync_lds();

        auto kt_reg_tensor = load_tile(kt_lds_read_window);

        store_tile(v_lds_write_window, v_block_tile);

        block_sync_lds();

        v_reg_tensor = load_tile(v_lds_read_window);
        //---------------------------- Loop Load in ----------------------------//
        // Q: HBM ->Reg ->LDS
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
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kK0>{}), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakePTRegSliceBlockDescriptor<Problem>());
        // QT: Reg -> Reg-> LDS
        auto qt_block_tile = make_static_distributed_tensor<QDataType>(
            Policy::template MakeQTRegWriteBlockDescriptor<Problem>());

        QDataType* qt_lds_ptr =
            static_cast<QDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto qt_lds_write = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeQTLdsWriteBlockDescriptor<Problem>());

        auto qt_lds_write_window =
            make_tile_window(qt_lds_write, make_tuple(number<kM0>{}, number<kK0>{}), {0, 0});

        auto qt_lds_read = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeQTLdsReadBlockDescriptor<Problem>());

        auto qt_lds_read_window =
            make_tile_window(qt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeQTRegSliceBlockDescriptor<Problem>());

        // dO: HBM ->Reg ->LDS
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
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kK2>{}), {0, 0});

        auto do_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());
        // dOT: Reg ->Reg ->LDS
        auto dot_block_tile = make_static_distributed_tensor<OGradDataType>(
            Policy::template MakeOGradTRegWriteBlockDescriptor<Problem>());

        OGradDataType* dot_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>()));

        auto dot_write_lds = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsWriteBlockDescriptor<Problem>());

        auto dot_lds_write_window =
            make_tile_window(dot_write_lds, make_tuple(number<kM0>{}, number<kK2>{}), {0, 0});

        auto dot_read_lds = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsReadBlockDescriptor<Problem>());

        auto dot_lds_read_window =
            make_tile_window(dot_read_lds,
                             make_tuple(number<kVHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeOGradTRegSliceBlockDescriptor<Problem>());

        // dS: Reg -> Reg -> LDS
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
        // Bias: HBM ->Reg ->Reg ->LDS
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();

        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, bias_origin.at(number<1>{})},
                             Policy::template MakeBiasTileDistribution<Problem>());

        BiasDataType* biast_lds_ptr = static_cast<BiasDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto biast_lds = make_tensor_view<address_space_enum::lds>(
            biast_lds_ptr, Policy::template MakeBiasTLdsBlockDescriptor<Problem>());

        auto biast_lds_shuffle_window =
            make_tile_window(biast_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto biast_lds_window =
            make_tile_window(biast_lds_shuffle_window.get_bottom_tensor_view(),
                             biast_lds_shuffle_window.get_window_lengths(),
                             biast_lds_shuffle_window.get_window_origin(),
                             Policy::template MakeBiasTTileDistribution<decltype(gemm_0)>());

        static_assert(std::is_same_v<BiasDataType, BiasGradDataType>,
                      "BiasDataType and BiasGradDataType should be the same!");

        // LSE: HBM -> LDS ->Reg
        auto lse_dram_window = make_tile_window(
            lse_dram_block_window_tmp.get_bottom_tensor_view(),
            lse_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_start},
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        LSEDataType* lse_lds_ptr = static_cast<LSEDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>()));

        auto lse_lds = make_tensor_view<address_space_enum::lds>(
            lse_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto lse_lds_write_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0});

        auto lse_lds_read_window = make_tile_window(
            lse_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        // D: HBM ->Reg
        auto d_dram_window = make_tile_window(
            d_dram_block_window_tmp.get_bottom_tensor_view(),
            d_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_start},
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        DDataType* d_lds_ptr = static_cast<DDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>()));

        auto d_lds = make_tensor_view<address_space_enum::lds>(
            d_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto d_lds_write_window = make_tile_window(d_lds, make_tuple(number<kM0>{}), {0});

        auto d_lds_read_window = make_tile_window(
            d_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        // RandVal: HBM ->Reg
        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0), false>(
            randval_dram_block_window_tmp, seqlen_q_start);

        // BiasGrad
        // Reg ->LDS ->Reg ->HBM
        const auto dbias_origin = dbias_dram_block_window_tmp.get_window_origin();

        auto dbias_dram_window =
            make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                             dbias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, dbias_origin.at(number<1>{})}); // M/N

        auto dbiast_lds_shuffle_window =
            make_tile_window(biast_lds,
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeShuffledBiasTileDistribution<Problem>());

        // ----------------------------Loop write out------------------------------//
        auto dq_dram_window = make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dq_dram_block_window_tmp.get_window_lengths(),
                                               {seqlen_q_start, 0});

        using SPTBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradTBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType   = decltype(gemm_4.MakeCBlockTile());

        index_t i_total_loops = 0;
        index_t seqlen_q_step = seqlen_q_start;
        static_assert(kQKHeaddim == kK0, "kQKHeaddim should equal to kK0");
        static_assert(kM0 == kK1, "kM0 should equal to kK1");
        static_assert(kVHeaddim == kK2, "kVHeaddim should equal to kK2");
        static_assert(kM0 == kK3, "kM0 should equal to kK3");
        constexpr index_t k4_loops = kN0 / kK4;

        /*
         * Prefetch Q, LSE, dO, D
         */
        auto q_block_tile = load_tile(q_dram_window);
        move_tile_window(q_dram_window, {kM0, 0});
        auto lse_block_tile = load_tile(lse_dram_window);
        move_tile_window(lse_dram_window, {kM0});

        auto do_block_tile = load_tile(do_dram_window);
        move_tile_window(do_dram_window, {kM0, 0});

        auto d_block_tile = load_tile(d_dram_window);
        move_tile_window(d_dram_window, {kM0});

        /*
         * Store prefetched data into LDS
         */
        block_sync_lds();
        store_tile(q_lds_window, q_block_tile);
        shuffle_tile(qt_block_tile, q_block_tile);
        store_tile(qt_lds_write_window, qt_block_tile);

        store_tile(lse_lds_write_window, lse_block_tile);

        store_tile(do_lds_window, do_block_tile);
        shuffle_tile(dot_block_tile, do_block_tile);
        store_tile(dot_lds_write_window, dot_block_tile);

        store_tile(d_lds_write_window, d_block_tile);
        block_sync_lds();

        /*
         * Prefetch LDS data into Reg to Asynchronous Data Movement and MFMA pipeline
         */

        auto q_reg_tensor  = load_tile(q_lds_read_window);
        auto lse           = load_tile(lse_lds_read_window);
        auto do_reg_tensor = load_tile(do_lds_read_window);
        auto d             = load_tile(d_lds_read_window);

        clear_tile(dv_acc);
        clear_tile(dk_acc);

        __builtin_amdgcn_sched_barrier(0);
        // Hot loop
        while(i_total_loops < (num_total_loop - 1))
        {
            // STAGE 1, Q@K Gemm0
            auto st_acc = SPTBlockTileType{};

            q_block_tile = load_tile(q_dram_window);
            move_tile_window(q_dram_window, {kM0, 0});

            lse_block_tile = load_tile(lse_dram_window);
            move_tile_window(lse_dram_window, {kM0});

            do_block_tile = load_tile(do_dram_window);
            move_tile_window(do_dram_window, {kM0, 0});

            d_block_tile = load_tile(d_dram_window);
            move_tile_window(d_dram_window, {kM0});

            st_acc = gemm_0(q_reg_tensor, k_reg_tensor);

            auto dot_reg_tensor = load_tile(dot_lds_read_window);

            HotLoopScheduler::template GemmStagedScheduler<0>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const auto bias_tile  = load_tile(bias_dram_window);
                auto bias_shuffle_tmp = make_static_distributed_tensor<BiasDataType>(
                    Policy::template MakeShuffledBiasTileDistribution<Problem>());
                shuffle_tile(bias_shuffle_tmp, bias_tile);
                store_tile(biast_lds_shuffle_window, bias_shuffle_tmp);
                block_sync_lds();
                auto biast_tile = load_tile(biast_lds_window);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
                        x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                    },
                    st_acc,
                    biast_tile);
                move_tile_window(bias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                constexpr auto st_spans = decltype(st_acc)::get_distributed_spans();
                sweep_tile_span(st_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(st_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            st_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        st_acc(i_j_idx) *= scale;
                        position_encoding.update(st_acc(i_j_idx), row, col);
                    });
                });
            }

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(st_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }

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
                auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

                sweep_tile_span(pt_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        pt(i_j_idx) = exp2(st_acc[i_j_idx] - row_lse);
                    }
                    else
                    {
                        pt(i_j_idx) = exp2(scale * st_acc[i_j_idx] - row_lse);
                    }
                });
            });

            if constexpr(FmhaDropout::IsDropout)
            {
                dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                    seqlen_q_step, k_origin.at(number<0>{}), pt, randval_dram_window);
            }
            const auto pt_gemm = [&]() {
                if constexpr(FmhaDropout::IsDropout)
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

            // STAGE 3, P^T@OGrad^T Gemm1
            Policy::template PTFromGemm0CToGemm1A<Problem,
                                                  decltype(pt_reg_tensor),
                                                  decltype(pt_gemm)>(pt_reg_tensor, pt_gemm);
            gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);

            auto qt_reg_tensor = load_tile(qt_lds_read_window);

            HotLoopScheduler::template GemmStagedScheduler<1>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 4, OGrad@V Gemm2
            auto dpt_acc = SPGradTBlockTileType{};

            dpt_acc = gemm_2(do_reg_tensor, v_reg_tensor);

            block_sync_lds();

            store_tile(q_lds_window, q_block_tile);
            shuffle_tile(qt_block_tile, q_block_tile);
            store_tile(qt_lds_write_window, qt_block_tile);

            store_tile(lse_lds_write_window, lse_block_tile);

            store_tile(do_lds_window, do_block_tile);
            shuffle_tile(dot_block_tile, do_block_tile);
            store_tile(dot_lds_write_window, dot_block_tile);

            store_tile(d_lds_write_window, d_block_tile);

            HotLoopScheduler::template GemmStagedScheduler<2>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE 5, P^T(PGrad^T - D)
            auto dst                 = SPGradTBlockTileType{};
            constexpr auto dst_spans = decltype(dst)::get_distributed_spans();
            sweep_tile_span(dst_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(dst_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    bool undrop_flag       = pt[i_j_idx] >= 0;
                    dst(i_j_idx)           = pt[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                                ? (dpt_acc[i_j_idx] - d[i_idx])
                                                                : d[i_idx]);
                });
            });

            if constexpr(kHasBiasGrad)
            {
                const auto dbiast = [&]() {
                    if constexpr(FmhaDropout::IsDropout)
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
                store_tile(dbias_dram_window, dbiast_shuffle_tmp);
                move_tile_window(dbias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }

            // STAGE 6, SGrad^T@Q^T Gemm3
            const auto dst_gemm = cast_tile<GemmDataType>(dst);

            Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                      decltype(dst_reg_tensor),
                                                      decltype(dst_gemm)>(dst_reg_tensor, dst_gemm);

            gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);

            store_tile(ds_lds_window, dst_gemm);

            block_sync_lds();

            auto ds_reg_tensor      = load_tile(ds_lds_read_window);
            auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
            move_tile_window(ds_lds_read_window, {0, kK4});
            q_reg_tensor = load_tile(q_lds_read_window);
            lse          = load_tile(lse_lds_read_window);

            HotLoopScheduler::template GemmStagedScheduler<3>();
            __builtin_amdgcn_sched_barrier(0);
            // STAGE7 SGrad@K^T Gemm4
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

            do_reg_tensor = load_tile(do_lds_read_window);
            d             = load_tile(d_lds_read_window);

            HotLoopScheduler::template GemmStagedScheduler<4>();

            // QGrad Scale
            if constexpr(FmhaDropout::IsDropout)
            {
                tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                       dq_acc);
            }
            else
            {
                tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            }
            if constexpr(kIsDeterministic)
            {
                store_tile(dq_dram_window, dq_acc);
            }
            else
            {
                update_tile(dq_dram_window, dq_acc);
            }
            move_tile_window(dq_dram_window, {kM0, 0});

            i_total_loops += 1;
            seqlen_q_step += kM0;
        }
        __builtin_amdgcn_sched_barrier(0);

        // Tail
        auto st_acc = SPTBlockTileType{};

        // STAGE 1, Q@K Gemm0
        st_acc = gemm_0(q_reg_tensor, k_reg_tensor);

        // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            const auto bias_tile  = load_tile(bias_dram_window);
            auto bias_shuffle_tmp = make_static_distributed_tensor<BiasDataType>(
                Policy::template MakeShuffledBiasTileDistribution<Problem>());
            shuffle_tile(bias_shuffle_tmp, bias_tile);
            store_tile(biast_lds_shuffle_window, bias_shuffle_tmp);
            block_sync_lds();
            auto biast_tile = load_tile(biast_lds_window);
            tile_elementwise_inout(
                [&](auto& x, const auto& y) {
                    x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                },
                st_acc,
                biast_tile);
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            constexpr auto st_spans = decltype(st_acc)::get_distributed_spans();
            sweep_tile_span(st_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(st_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        st_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                    const auto row         = seqlen_q_step + tile_idx.at(number<0>{});
                    const auto col         = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    st_acc(i_j_idx) *= scale;
                    position_encoding.update(st_acc(i_j_idx), row, col);
                });
            });
        }

        if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
        {
            bool need_perpixel_check = mask.IsEdgeTile(
                seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
            if(need_perpixel_check)
            {
                set_tile_if(st_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                    const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                    const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    return mask.IsOutOfBound(row, col);
                });
            }
        }

        static const auto get_validated_lse = [](LSEDataType raw_lse) {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         FmhaMask::IsMasking)
            {
                return raw_lse == -numeric<LSEDataType>::infinity() ? type_convert<LSEDataType>(0.f)
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
            auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

            sweep_tile_span(pt_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    pt(i_j_idx) = exp2(st_acc[i_j_idx] - row_lse);
                }
                else
                {
                    pt(i_j_idx) = exp2(scale * st_acc[i_j_idx] - row_lse);
                }
            });
        });

        if constexpr(FmhaDropout::IsDropout)
        {
            dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                seqlen_q_step, k_origin.at(number<0>{}), pt, randval_dram_window);
        }

        // STAGE 3, P^T@OGrad^T Gemm1
        const auto pt_gemm = [&]() {
            if constexpr(FmhaDropout::IsDropout)
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

        Policy::template PTFromGemm0CToGemm1A<Problem, decltype(pt_reg_tensor), decltype(pt_gemm)>(
            pt_reg_tensor, pt_gemm);
        auto dot_reg_tensor = load_tile(dot_lds_read_window);
        gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);

        HotLoopScheduler::template GemmStagedScheduler<1>();

        // STAGE 4, OGrad@V Gemm2
        auto dpt_acc = SPGradTBlockTileType{};

        auto qt_reg_tensor = load_tile(qt_lds_read_window);

        dpt_acc = gemm_2(do_reg_tensor, v_reg_tensor);

        HotLoopScheduler::template GemmStagedScheduler<2>();

        // STAGE 5, P^T(PGrad^T - D)
        auto dst                 = SPGradTBlockTileType{};
        constexpr auto dst_spans = decltype(dst)::get_distributed_spans();
        sweep_tile_span(dst_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(dst_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                bool undrop_flag       = pt[i_j_idx] >= 0;
                dst(i_j_idx)           = pt[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                            ? (dpt_acc[i_j_idx] - d[i_idx])
                                                            : d[i_idx]);
            });
        });

        if constexpr(kHasBiasGrad)
        {
            const auto dbiast = [&]() {
                if constexpr(FmhaDropout::IsDropout)
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
            store_tile(dbias_dram_window, dbiast_shuffle_tmp);
        }

        // STAGE 6, SGrad^T@Q^T Gemm3
        const auto dst_gemm = cast_tile<GemmDataType>(dst);

        Policy::template SGradTFromGemm2CToGemm3A<Problem,
                                                  decltype(dst_reg_tensor),
                                                  decltype(dst_gemm)>(dst_reg_tensor, dst_gemm);

        gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);
        store_tile(ds_lds_window, dst_gemm);

        block_sync_lds();

        auto ds_reg_tensor      = load_tile(ds_lds_read_window);
        auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
        move_tile_window(ds_lds_read_window, {0, kK4});

        HotLoopScheduler::template GemmStagedScheduler<3>();
        // STAGE 7, SGrad@K^T Gemm4
        auto dq_acc = QGradBlockTileType{};
        clear_tile(dq_acc);

        static_for<0, k4_loops, 1>{}([&](auto i_k4) {
            if constexpr(i_k4 < k4_loops - 1)
            {
                ds_reg_tensor_next = load_tile(ds_lds_read_window);
                move_tile_window(ds_lds_read_window, {0, kK4});
            }
            auto kt_reg_tensor_slice = get_slice_tile(
                kt_reg_tensor, sequence<0, i_k4 * kK4>{}, sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});

            gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);
            if constexpr(i_k4 < k4_loops - 1)
            {
                ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
            }
        });

        HotLoopScheduler::template GemmStagedScheduler<4>();

        // Results Scale
        if constexpr(FmhaDropout::IsDropout)
        {
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dq_acc);
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dk_acc);
            tile_elementwise_inout([&rp_undrop](auto& x) { x = x * rp_undrop; }, dv_acc);
        }
        else
        {
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
        }

        if constexpr(kIsDeterministic)
        {
            store_tile(dq_dram_window, dq_acc);
        }
        else
        {
            update_tile(dq_dram_window, dq_acc);
        }

        return make_tuple(dk_acc, dv_acc);
    }
};

} // namespace ck_tile
