// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BlockFmhaFwdAppendKVPipelineDefaultPolicy>
struct BlockFmhaFwdAppendKVPipeline
{
    using Problem   = remove_cvref_t<Problem_>;
    using Policy    = remove_cvref_t<Policy_>;
    using QDataType = typename Problem::QDataType;
    using KDataType = typename Problem::KDataType;
    using VDataType = typename Problem::VDataType;

    using VLayout = typename Problem::VLayout;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kTileSizeS  = Problem::kTileSizeS;
    static constexpr index_t kTileSizeSk = Problem::kTileSizeSk;
    static constexpr index_t kTileSizeD  = Problem::kTileSizeD;
    static constexpr index_t kTileSizeDv = Problem::kTileSizeDv;

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;
    static constexpr bool kApplyRoPE   = Problem::kApplyRoPE;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kTileSizeD <= 32)
            {
                return 2;
            }
            else if constexpr(kTileSizeD <= 64)
            {
                return 3;
            }
            else if constexpr(kTileSizeD <= 128)
            {
                return 2;
            }
            else if constexpr(kTileSizeD <= 256)
            {
                return 1;
            }
        }
    }();

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename KnewDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename VnewDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename KnewElementFunction,
              typename VElementFunction,
              typename VnewElementFunction,
              typename RotaryCosBlockWindowTemp,
              typename RotarySinBlockWindowTemp>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& k_element_func,
               const KnewDramBlockWindowTmp& knew_dram_block_window_tmp, // N0*K0 tile
               const KnewElementFunction& knew_element_func,
               VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const VnewDramBlockWindowTmp& vnew_dram_block_window_tmp, // N1*K1 tile
               const VnewElementFunction& vnew_element_func,
               const RotaryCosBlockWindowTemp rotary_cos_block_window_tmp,
               const RotarySinBlockWindowTemp rotary_sin_block_window_tmp,
               void* smem_ptr,
               index_t rotary_dim         = 0,
               bool is_rotary_interleaved = false) const
    {
        (void)q_dram_block_window_tmp;
        (void)q_element_func;
        (void)k_dram_block_window_tmp;
        (void)k_element_func;
        (void)knew_dram_block_window_tmp;
        (void)knew_element_func;
        (void)v_dram_block_window_tmp;
        (void)v_element_func;
        (void)vnew_dram_block_window_tmp;
        (void)vnew_element_func;
        (void)rotary_cos_block_window_tmp;
        (void)rotary_sin_block_window_tmp;
        (void)smem_ptr;
        (void)rotary_dim;
        (void)is_rotary_interleaved;

        auto knew_dram_block_window =
            make_tile_window(knew_dram_block_window_tmp.get_bottom_tensor_view(),
                             knew_dram_block_window_tmp.get_window_lengths(),
                             {0, 0});

        auto knew_dram_window =
            make_tile_window(knew_dram_block_window.get_bottom_tensor_view(),
                             knew_dram_block_window.get_window_lengths(),
                             knew_dram_block_window.get_window_origin(),
                             Policy::template MakeKnewDramTileDistribution<Problem>());

        auto knew_tile = load_tile(knew_dram_window);
        /// TODO: apply RoPE on knew_tile here
        store_tile(k_dram_block_window_tmp, knew_tile);

        auto vnew_dram_block_window =
            make_tile_window(vnew_dram_block_window_tmp.get_bottom_tensor_view(),
                             vnew_dram_block_window_tmp.get_window_lengths(),
                             {0, 0});

        auto vnew_dram_window =
            make_tile_window(vnew_dram_block_window.get_bottom_tensor_view(),
                             vnew_dram_block_window.get_window_lengths(),
                             vnew_dram_block_window.get_window_origin(),
                             Policy::template MakeVnewDramTileDistribution<Problem>());

        auto vnew_tile = load_tile(vnew_dram_window);

#if defined(ENABLE_PIPELINE_DEBUG_PRINT)
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == TID)
        {
            printf("[POYENC][DEVICE] tid: %d\n", TID);
            constexpr auto spans = decltype(vnew_tile)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        vnew_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                    const auto row         = tile_idx.at(number<0>{});
                    const auto col         = tile_idx.at(number<1>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    printf("[POYENC][DEVICE] vnew_tile(%2d,%2d): %11.7f\n",
                           row,
                           col,
                           type_convert<float>(vnew_tile(i_j_idx)));
                });
            });
        }
#endif
        store_tile(v_dram_block_window_tmp, vnew_tile);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename KnewDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename VnewDramBlockWindowTmp,
              typename RotaryCosBlockWindowTemp,
              typename RotarySinBlockWindowTemp>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const KnewDramBlockWindowTmp& knew_dram_block_window_tmp,
                                        VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const VnewDramBlockWindowTmp& vnew_dram_block_window_tmp,
                                        const RotaryCosBlockWindowTemp& rotary_cos_block_window_tmp,
                                        const RotarySinBlockWindowTemp& rotary_sin_block_window_tmp,
                                        void* smem_ptr,
                                        index_t rotary_dim         = 0,
                                        bool is_rotary_interleaved = false) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          knew_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          vnew_dram_block_window_tmp,
                          identity{},
                          rotary_cos_block_window_tmp,
                          rotary_sin_block_window_tmp,
                          smem_ptr,
                          rotary_dim,
                          is_rotary_interleaved);
    }
};

} // namespace ck_tile
