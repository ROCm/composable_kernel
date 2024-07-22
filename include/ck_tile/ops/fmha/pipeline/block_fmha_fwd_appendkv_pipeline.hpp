// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding_enum.hpp"
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
    static constexpr auto RotaryEnum   = Problem::RotaryEnum;

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
               index_t rotary_dim = 0) const
    {

        auto* const ksmem = reinterpret_cast<KDataType*>(smem_ptr);
        if(threadIdx.x == 0)
        {
            printf("\n");
        }

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

        if constexpr(RotaryEnum != BlockRotaryEmbeddingEnum::NONE)
        {
            auto rotary_cos_window = make_tile_window(
                rotary_cos_block_window_tmp.get_bottom_tensor_view(),
                rotary_cos_block_window_tmp.get_window_lengths(),
                rotary_cos_block_window_tmp.get_window_origin(),
                Policy::template MakeRotaryCosSinInterleaveDramTileDistribution<Problem>());

            auto rotary_sin_window = make_tile_window(
                rotary_sin_block_window_tmp.get_bottom_tensor_view(),
                rotary_sin_block_window_tmp.get_window_lengths(),
                rotary_sin_block_window_tmp.get_window_origin(),
                Policy::template MakeRotaryCosSinInterleaveDramTileDistribution<Problem>());

            if constexpr(RotaryEnum == BlockRotaryEmbeddingEnum::INTERLEAVED)
            {
                auto rotary_cos_tile = load_tile(rotary_cos_window);
                auto rotary_sin_tile = load_tile(rotary_sin_window);

                constexpr index_t KPerThread = 16 / sizeof(KDataType);
                static_assert(kTileSizeD % KPerThread == 0);
                constexpr index_t KThreadPerBlock = kTileSizeD / KPerThread;
                index_t start_x                   = (threadIdx.x % KThreadPerBlock) * KPerThread;

                if((start_x + KPerThread) <= rotary_dim)
                {
                    constexpr index_t thread_buffer_size = decltype(knew_tile.thread_buf_)::size();
                    static_assert(thread_buffer_size % KPerThread == 0);
                    static_for<0, thread_buffer_size, 2>{}([&](auto idx) {
                        const auto left  = type_convert<float>(knew_tile.thread_buf_[idx]);
                        const auto right = type_convert<float>(knew_tile.thread_buf_[idx + 1]);

                        const auto cos = type_convert<float>(rotary_cos_tile.thread_buf_[idx / 2]);
                        const auto sin = type_convert<float>(rotary_sin_tile.thread_buf_[idx / 2]);

                        knew_tile.thread_buf_[idx]     = left * cos - right * sin;
                        knew_tile.thread_buf_[idx + 1] = right * cos + left * sin;
                    });
                }
            }
            else // RotaryEnum == BlockRotaryEmbeddingEnum::HALF_ROTATED
            {
                constexpr index_t KPerThread = 16 / sizeof(KDataType);
                static_assert(kTileSizeD % KPerThread == 0);
                constexpr index_t KThreadPerBlock = kTileSizeD / KPerThread;
                index_t start_x                   = (threadIdx.x % KThreadPerBlock) * KPerThread;

                bool is_left = (start_x + KPerThread) <= (rotary_dim / 2);

                if((start_x + KPerThread) <= rotary_dim)
                {
                    auto knew_other_dram_window = knew_dram_window;
                    DEVICE_DEBUG_STMTS
                    {
                        auto origin = knew_other_dram_window.get_window_origin();
                        printf("after move window, origin = (%3d, %3d)\n",
                               origin.at(number<0>{}),
                               origin.at(number<1>{}));
                    }
                    move_tile_window(knew_other_dram_window,
                                     {0, is_left ? rotary_dim / 2 : -(rotary_dim / 2)});
                    DEVICE_DEBUG_STMTS
                    {
                        auto origin = knew_other_dram_window.get_window_origin();
                        printf("after move window, origin = (%3d, %3d)\n",
                               origin.at(number<0>{}),
                               origin.at(number<1>{}));
                    }
                    auto knew_other_tile = load_tile(knew_other_dram_window);
                }
            }

#if defined(ENABLE_DEVICE_DEBUG_STMTS)
            {
                constexpr auto spans = decltype(knew_tile)::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            knew_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row         = tile_idx.at(number<0>{});
                        const auto col         = tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        ksmem[row * kTileSizeD + col] = knew_tile(i_j_idx);
                    });
                });
            }

            block_sync_lds();

            DEVICE_DEBUG_STMTS
            {
                for(int row = 0; row < 7; ++row)
                {
                    printf("[DEVICE] knew_tile[%3d] = ", row);

                    for(int col = 0; col < kTileSizeD; ++col)
                    {
                        printf("%11.7f", type_convert<float>(ksmem[row * kTileSizeD + col]));
                    }
                    printf("\n");
                }
            }
#endif
        }
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
                                        index_t rotary_dim = 0) const
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
                          rotary_dim);
    }
};

} // namespace ck_tile
