// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_decode_pipeline_qr_ks_vs_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
template <typename Problem_, typename Policy_ = BlockFmhaFwdDecodePipelineQRKSVSDefaultPolicy>
struct BlockFmhaFwdDecodePipelineQRKSVS
{
    using Problem             = remove_cvref_t<Problem_>;
    using Policy              = remove_cvref_t<Policy_>;
    using QDataType           = remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType        = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType         = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType           = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant    = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask            = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;
    static constexpr index_t kNWarp        = BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});
    static constexpr index_t kNXdl         = BlockFmhaShape::Gemm0WarpTile::at(number<1>{});

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    // static_assert(Problem::kPadSeqLenQ == true && Problem::kPadHeadDimQ == true &&
    //               Problem::kPadHeadDimV == true);

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ =
        Problem::kPadHeadDimQ; // support multiple of vector(like 8x)
    static constexpr bool kPadHeadDimV =
        Problem::kPadHeadDimV; // support multiple of vector(like 8x)

    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kIsPagedKV        = Problem::kIsPagedKV;
    static constexpr bool kHasUnevenSplits  = Problem::kHasUnevenSplits;

    static_assert((CK_TILE_FMHA_FWD_FAST_EXP2 &&
                   (kHasLogitsSoftCap && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                    !kHasLogitsSoftCap)) ||
                  (!CK_TILE_FMHA_FWD_FAST_EXP2 && !kHasLogitsSoftCap));

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kAlignmentOacc = Policy::template GetAlignmentO<Problem>();

    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim <= 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            }
        }
    }();

    static constexpr const char* name = "decode_qr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowLengths,
              typename KPageBlockNavigator,
              typename VDramBlockWindowLengths,
              typename VPageBlockNavigator,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,         // M0*K0 tile
               const KDramBlockWindowLengths& k_dram_block_window_lengths, // N0*K0 tile
               const KPageBlockNavigator& k_page_block_navigator,
               const VDramBlockWindowLengths& v_dram_block_window_lengths, // N1*K1 tile
               const VPageBlockNavigator& v_page_block_navigator,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,        // M0*1 tile
               index_t num_splits,
               index_t i_split,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               index_t kv_l2p_offset, // logical-to-physical offset of seqlen_k coordinate
               void* smem_ptr) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KPageBlockNavigator::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VPageBlockNavigator::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kSubQKHeaddim ==
                              QDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN0 == KDramBlockWindowLengths{}[number<0>{}] &&
                          kK0 == KDramBlockWindowLengths{}[number<1>{}] &&
                          kN1 == VDramBlockWindowLengths{}[number<0>{}] &&
                          kK1 == VDramBlockWindowLengths{}[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        auto o_acc = OaccBlockTileType{};

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(o_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // init M, L
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        clear_tile(o_acc);
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();
        const auto [logical_seqlen_k_start, logical_seqlen_k_end] = mask.GetTileRangeAlongX(
            q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{}, num_splits, i_split);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0);
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse_acc =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());

                    if(get_thread_local_1d_id() < kM0)
                    {
                        store_tile(lse_acc_dram_window_tmp, lse_acc);
                    }
                }

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return o_acc;
            }
        }

        // Q tile in LDS
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptr), Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window = make_tile_window(
            q_lds, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        async_load_tile(q_lds_store_window, q_dram_window);

        // K tile in LDS
        const index_t physical_seqlen_k_start = logical_seqlen_k_start + kv_l2p_offset;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end + kv_l2p_offset;
        // make sure the first tile is completely located in page-block (page-block size should be
        // divisible by kN0)
        // relationship between each *_start variables: aligned_physical_seqlen_k_start <=
        // physical_seqlen_k_start, logical_seqlen_k_start <= physical_seqlen_k_start
        const index_t aligned_physical_seqlen_k_start =
            [&, physical_seqlen_k_start_ = physical_seqlen_k_start] {
                if constexpr(kIsPagedKV)
                {
                    return kN0 * integer_divide_floor(physical_seqlen_k_start_, kN0);
                }
                else
                {
                    return physical_seqlen_k_start_;
                }
            }();

        auto [i_page_block_k, k_dram_block_window] = k_page_block_navigator.make_tile_window(
            k_dram_block_window_lengths, {aligned_physical_seqlen_k_start, 0});

        auto k_dram_window = make_tile_window(
            k_dram_block_window, Policy::template MakeKDramTileDistribution<Problem>());

        auto k_lds = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType*>(smem_ptr), Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});
        auto k_lds_read_window =
            make_tile_window(k_lds,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());

        // S tile in LDS
        auto s_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SaccDataType*>(reinterpret_cast<char*>(smem_ptr) +
                                            max(Policy::template GetSmemSizeQ<Problem>(),
                                                Policy::template GetSmemSizeK<Problem>())),
            Policy::template MakeSLdsBlockDescriptor<Problem>());
        auto s_write_lds_window = make_tile_window(
            s_lds, Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto s_read_lds_window =
            make_tile_window(s_lds,
                             Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeSRegTileDistribution<Problem>());

        // V tile in LDS
        auto [i_page_block_v, v_dram_block_window] = v_page_block_navigator.make_tile_window(
            v_dram_block_window_lengths, {0, aligned_physical_seqlen_k_start});

        auto v_dram_window = make_tile_window(
            v_dram_block_window, Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_ptr) +
                                         max(Policy::template GetSmemSizeQ<Problem>(),
                                             Policy::template GetSmemSizeK<Problem>()) +
                                         Policy::template GetSmemSizeS<Problem>()),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_write_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds,
                             Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        // Bias tile in LDS
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}),
                              logical_seqlen_k_start - (physical_seqlen_k_start -
                                                        aligned_physical_seqlen_k_start)}, // M/N
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        block_sync_lds_direct_load<0>();
        auto q_tile = load_tile(q_lds_read_window);

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - aligned_physical_seqlen_k_start, kN0);

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 == k0_loops);
        static_assert(1 == k1_loops);

        async_load_tile(k_lds_write_window, k_dram_window);
        // move K tile windows
        i_page_block_k =
            k_page_block_navigator.move_tile_window(i_page_block_k, k_dram_block_window, {kN0, 0});

        k_dram_window = make_tile_window(k_dram_block_window,
                                         Policy::template MakeKDramTileDistribution<Problem>());

        constexpr index_t k_vmem_insts = k_dram_window.get_num_of_access();
        constexpr index_t v_vmem_insts = v_dram_window.get_num_of_access();

        do
        {
            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C

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

            async_load_tile(v_lds_write_window, v_dram_window); // prefetch load v tile
            // auto v_temp = load_tile(v_dram_window);
            // printf("Tid: %02d, v_temp: %04x %04x %04x %04x %04x %04x %04x %04x| %04x %04x %04x
            // %04x %04x %04x %04x %04x| %04x %04x %04x %04x %04x %04x %04x %04x| %04x %04x %04x
            // %04x %04x %04x %04x %04x|\n",
            //        get_thread_local_1d_id(),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<7>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<8+0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<8+1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<8+2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<8+3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<8+4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<8+5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<8+6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<8+7>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<16+0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<16+1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<16+2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<16+3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<16+4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<16+5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<16+6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<16+7>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<24+0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<24+1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<24+2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<24+3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<24+4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<24+5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<24+6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_temp.get_thread_buffer()(number<24+7>{})))));

            // move V tile windows
            i_page_block_v =
                v_page_block_navigator.move_tile_window(i_page_block_v, v_dram_window, {kK1, 0});

            // CK_PRINT<decltype(v_dram_window.get_num_of_access())>();
            block_sync_lds_direct_load<v_vmem_insts>();
            auto k_tile = load_tile(k_lds_read_window);

            gemm_0(
                s_acc,
                get_slice_tile(
                    q_tile, sequence<0, (k0_loops - 1) * kK0>{}, sequence<kM0, k0_loops * kK0>{}),
                get_slice_tile(
                    k_tile, sequence<0, (k0_loops - 1) * kK0>{}, sequence<kN0, k0_loops * kK0>{}));

            // printf("Tid: %02d, SAcc: %.0lf %.0lf %.0lf %.0lf %.0lf %.0lf %.0lf %.0lf\n",
            //        get_thread_local_1d_id(),
            //        s_acc.get_thread_buffer()(number<0>{}),
            //        s_acc.get_thread_buffer()(number<1>{}),
            //        s_acc.get_thread_buffer()(number<2>{}),
            //        s_acc.get_thread_buffer()(number<3>{}),
            //        s_acc.get_thread_buffer()(number<4>{}),
            //        s_acc.get_thread_buffer()(number<5>{}),
            //        s_acc.get_thread_buffer()(number<6>{}),
            //        s_acc.get_thread_buffer()(number<7>{}));

            // STAGE 2, scale_s, add bias, mask, softmax
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        x += type_convert<SaccDataType>(bias_element_func(y));
#else
                        x += log2e_v<SaccDataType> *
                             type_convert<SaccDataType>(bias_element_func(y));
#endif
                    },
                    s_acc,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto k_origin = k_page_block_navigator.to_global_window_origin(
                    i_page_block_k, k_dram_block_window.get_window_origin());
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale_s;
                        // position_encoding accept only logical coordinates, do conversion here
                        position_encoding.update(s_acc(i_j_idx), row, col - kv_l2p_offset);
                    });
                });
            }
            else
            {
                if constexpr(kHasLogitsSoftCap)
                {
                    auto apply_logits_transform =
                        [&variant, &variant_params, &block_indices](auto& x) {
                            x = variant.LogitsTransform(variant_params,
                                                        variant.QueryTransform(variant_params, x),
                                                        block_indices.batch_idx,
                                                        block_indices.qo_head_idx,
                                                        block_indices.kv_head_idx);
                        };
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    for(index_t i = 0; i < s_acc.thread_buf_.size(); ++i)
                    {
                        apply_logits_transform(s_acc.thread_buf_[i]);
                    }
#else
                    for(index_t i = 0; i < s_acc.thread_buf_.size(); ++i)
                    {
                        apply_logits_transform(s_acc.thread_buf_[i]);
                    }
#endif
                }
                else
                {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
#endif
                }
            }
            move_tile_window(bias_dram_window, {0, kN0});

            /// TODO: only check in first/last iteration without increasing code size
            if constexpr(kHasUnevenSplits)
            {
                const auto k_origin = k_page_block_navigator.to_global_window_origin(
                    i_page_block_k, k_dram_block_window.get_window_origin());
                set_tile_if(
                    s_acc,
                    -numeric<SMPLComputeDataType>::infinity(),
                    [&,
                     physical_seqlen_k_start_ = physical_seqlen_k_start,
                     physical_seqlen_k_end_   = physical_seqlen_k_end](auto tile_idx) {
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        if constexpr(kIsPagedKV)
                        {
                            return col < physical_seqlen_k_start_ || physical_seqlen_k_end_ <= col;
                        }
                        else
                        {
                            return physical_seqlen_k_end_ <= col;
                        }
                    });
            }

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin = k_page_block_navigator.to_global_window_origin(
                    i_page_block_k, k_dram_block_window.get_window_origin());
                // mask accept only logical coordinates, do conversion here
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}) - kv_l2p_offset,
                                                           number<kM0>{},
                                                           number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                            return mask.IsOutOfBound(row, col - kv_l2p_offset);
                        });
                }
            }

            async_load_tile(k_lds_write_window, k_dram_window);
            i_page_block_k = k_page_block_navigator.move_tile_window(
                i_page_block_k, k_dram_block_window, {kN0, 0});

            k_dram_window = make_tile_window(k_dram_block_window,
                                             Policy::template MakeKDramTileDistribution<Problem>());

            // In Nwarp=1 and NXdl=32, GEMM0 output naturally fit the input of GEMM1
            // Otherwise shuffle through LDS so that the tile layout is consistent with required by
            // Gemm1
            auto s_new = [&]() {
                if constexpr(kNWarp > 1)
                {
                    auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}

                    store_tile(s_write_lds_window, s);
                    block_sync_lds();
                    return load_tile(s_read_lds_window);
                }
                else
                {
                    return cast_tile<SMPLComputeDataType>(s_acc); // S{j}
                }
            }();

            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s_new,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s_new.get_tile_distribution()); // Pcompute{j}

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                /// NOTICE: bias might be materialized mask including -inf values, need
                /// consideration
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                auto row_max = scale_s * get_validated_m(m[i_idx]);
#endif
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p_compute(i_j_idx) = exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            p_compute(i_j_idx) = exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s_new[i_j_idx] - row_max);
                        }
                    }
#else
                    p_compute(i_j_idx)     = exp(s_new[i_j_idx] - get_validated_m(m[i_idx]));
#endif
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

            auto p_tile = make_static_distributed_tensor<PDataType>(
                Policy::template MakePRegTileDistribution<Problem>());
            p_tile.get_thread_buffer() = cast_tile<PDataType>(p_compute).get_thread_buffer();

            // printf("Tid: %02d, PCompute: %.0lf %.0lf %.0lf %.0lf %.0lf %.0lf %.0lf %.0lf\n",
            //        get_thread_local_1d_id(),
            //        p_compute.get_thread_buffer()(number<0>{}),
            //        p_compute.get_thread_buffer()(number<1>{}),
            //        p_compute.get_thread_buffer()(number<2>{}),
            //        p_compute.get_thread_buffer()(number<3>{}),
            //        p_compute.get_thread_buffer()(number<4>{}),
            //        p_compute.get_thread_buffer()(number<5>{}),
            //        p_compute.get_thread_buffer()(number<6>{}),
            //        p_compute.get_thread_buffer()(number<7>{}));

            // printf("Tid: %02d, p_tile: %04x %04x %04x %04x %04x %04x %04x %04x\n",
            //        get_thread_local_1d_id(),
            //        *(reinterpret_cast<uint16_t*>(&(p_tile.get_thread_buffer()(number<0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(p_tile.get_thread_buffer()(number<1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(p_tile.get_thread_buffer()(number<2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(p_tile.get_thread_buffer()(number<3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(p_tile.get_thread_buffer()(number<4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(p_tile.get_thread_buffer()(number<5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(p_tile.get_thread_buffer()(number<6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(p_tile.get_thread_buffer()(number<7>{})))));

            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                const auto tmp = [&]() {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            auto row_max = scale_s * get_validated_m(m[i_idx]);
                            return exp2(scale_s * m_old[i_idx] - row_max);
                        }
                    }
                }();
#else
                const auto tmp       = exp(m_old[i_idx] - get_validated_m(m[i_idx]));
#endif
                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // FIXME: this use different equation from FA v2 paper,
                    // but produce correc result.
                    // Is the equation wrong?
                    o_acc(i_j_idx) *= tmp;
                });
            });

            block_sync_lds_direct_load<k_vmem_insts>();
            auto v_tile = load_tile_transpose(v_lds_read_window);

            // printf("Tid: %02d, v_tile: %04x %04x %04x %04x %04x %04x %04x %04x| %04x %04x %04x
            // %04x %04x %04x %04x %04x| %04x %04x %04x %04x %04x %04x %04x %04x| %04x %04x %04x
            // %04x %04x %04x %04x %04x|\n",
            //        get_thread_local_1d_id(),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<7>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<8+0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<8+1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<8+2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<8+3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<8+4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<8+5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<8+6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<8+7>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<16+0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<16+1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<16+2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<16+3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<16+4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<16+5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<16+6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<16+7>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<24+0>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<24+1>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<24+2>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<24+3>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<24+4>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<24+5>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<24+6>{})))),
            //        *(reinterpret_cast<uint16_t*>(&(v_tile.get_thread_buffer()(number<24+7>{})))));

            // block_sync_lds();
            // CK_PRINT<decltype(p_tile)>();
            // ck_tile::tile_distribution_encoding<
            //     ck_tile::sequence<1>,
            //     ck_tile::tuple<ck_tile::sequence<1, 1, 16>, ck_tile::sequence<1, 2, 4, 4>>,
            //     ck_tile::tuple<ck_tile::sequence<1, 0>, ck_tile::sequence<2, 1>>,
            //     ck_tile::tuple<ck_tile::sequence<1, 0>, ck_tile::sequence<2, 2>>,
            //     ck_tile::sequence<1, 2, 2, 2>,
            //     ck_tile::sequence<0, 0, 1, 3>>;
            // CK_PRINT<decltype(v_tile)>();
            // ck_tile::tile_distribution_encoding<
            //     ck_tile::sequence<1>,
            //     ck_tile::tuple<ck_tile::sequence<4, 1, 16>, ck_tile::sequence<1, 2, 4, 4>>,
            //     ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<2, 1>>,
            //     ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<2, 2>>,
            //     ck_tile::sequence<1, 2, 2, 2>,
            //     ck_tile::sequence<0, 0, 1, 3>>;

            gemm_1(
                o_acc,
                get_slice_tile(
                    p_tile, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, k1_loops * kK1>{}),
                get_slice_tile(
                    v_tile, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kN1, k1_loops * kK1>{}));

        } while(++i_total_loops < num_total_loop);

        if constexpr(kStoreLSE)
        {
            // store lse acc
            auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
            sweep_tile_span(lse_acc_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                }
                else
                {
                    if constexpr(kHasLogitsSoftCap)
                    {
                        lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse_acc(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                    }
                }
#else
                    lse_acc(i_idx) = m_[i_idx] + log(l_[i_idx]);
#endif
            });

            if(get_thread_local_1d_id() < kM0)
            {
                store_tile(lse_acc_dram_window_tmp, lse_acc);
            }
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
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
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        return o_acc;
    }
};

} // namespace ck_tile
