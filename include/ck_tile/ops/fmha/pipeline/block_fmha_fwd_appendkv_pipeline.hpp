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

    template <typename QDramBlockWindow,
              typename KDramBlockWindow,
              typename KnewDramBlockWindow,
              typename VDramBlockWindow,
              typename VnewDramBlockWindow,
              typename QElementFunction,
              typename KnewElementFunction,
              typename VnewElementFunction,
              typename RotaryCosDramBlockWindow,
              typename RotarySinDramBlockWindow>
    CK_TILE_HOST_DEVICE auto
    operator()(QDramBlockWindow& q_dram_block_window, // M0*K0 tile
               const QElementFunction& q_element_func,
               KDramBlockWindow& k_dram_block_window,             // N0*K0 tile
               const KnewDramBlockWindow& knew_dram_block_window, // N0*K0 tile
               const KnewElementFunction& knew_element_func,
               VDramBlockWindow& v_dram_block_window,             // N1*K1 tile
               const VnewDramBlockWindow& vnew_dram_block_window, // N1*K1 tile
               const VnewElementFunction& vnew_element_func,
               const RotaryCosDramBlockWindow rotary_cos_dram_block_window,
               const RotarySinDramBlockWindow rotary_sin_dram_block_window,
               void* smem_ptr,
               index_t rotary_dim = 0) const
    {
#if defined(ENABLE_DEVICE_DEBUG_STMTS)
        auto* const ksmem = reinterpret_cast<KDataType*>(smem_ptr);
        if(threadIdx.x == 0)
        {
            printf("\n");
        }
#endif

        auto print_tile = [&](const auto& tile, index_t num_display_rows = -1) {
            (void)tile;
#if defined(ENABLE_DEVICE_DEBUG_STMTS)
            using Dstr                 = decltype(tile.get_tile_distribution());
            constexpr index_t num_rows = Dstr::get_lengths()[number<0>{}];
            constexpr index_t num_cols = Dstr::get_lengths()[number<1>{}];
            {
                constexpr auto spans = std::decay_t<decltype(tile)>::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            tile.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row         = tile_idx.at(number<0>{});
                        const auto col         = tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        ksmem[row * num_cols + col] = tile[i_j_idx];
                    });
                });
            }

            block_sync_lds();

            DEVICE_DEBUG_STMTS
            {
                for(int row = 0;
                    row < (0 < num_display_rows ? std::min(num_display_rows, num_rows) : num_rows);
                    ++row)
                {
                    printf("[DEVICE] tile[%3d] = ", row);

                    for(int col = 0; col < num_cols; ++col)
                    {
                        if(0 < col && col % 8 == 0)
                        {
                            printf("|");
                        }
                        printf("%11.7f", type_convert<float>(ksmem[row * num_cols + col]));
                    }
                    printf("\n");
                }
            }
#endif
        };

        auto knew_window =
            make_tile_window(knew_dram_block_window.get_bottom_tensor_view(),
                             knew_dram_block_window.get_window_lengths(),
                             knew_dram_block_window.get_window_origin(),
                             Policy::template MakeKnewDramTileDistribution<Problem>());

        auto knew_tile = [&]() {
            auto knew = load_tile(knew_window);
            return tile_elementwise_in(knew_element_func, knew);
        }();

        // optionally apply rotary embedding to Knew
        if constexpr(RotaryEnum != BlockRotaryEmbeddingEnum::NONE)
        {
            auto rotary_cos_window =
                make_tile_window(rotary_cos_dram_block_window.get_bottom_tensor_view(),
                                 rotary_cos_dram_block_window.get_window_lengths(),
                                 rotary_cos_dram_block_window.get_window_origin(),
                                 Policy::template MakeRotaryCosSinTileDistribution<Problem>());

            auto rotary_sin_window =
                make_tile_window(rotary_sin_dram_block_window.get_bottom_tensor_view(),
                                 rotary_sin_dram_block_window.get_window_lengths(),
                                 rotary_sin_dram_block_window.get_window_origin(),
                                 Policy::template MakeRotaryCosSinTileDistribution<Problem>());

            // We assume that each thread owns contiguous elements on head dimention. And we will
            // use the distribution to enable/disable threads in order to override knew_tile content
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

                        knew_tile.thread_buf_[idx] =
                            type_convert<KDataType>(left * cos - right * sin);
                        knew_tile.thread_buf_[idx + 1] =
                            type_convert<KDataType>(right * cos + left * sin);
                    });
                }
            }
            else // RotaryEnum == BlockRotaryEmbeddingEnum::HALF_ROTATED
            {
                constexpr index_t KPerThread = 8 / sizeof(KDataType);
                static_assert(kTileSizeD % KPerThread == 0);
                constexpr index_t KThreadPerBlock = kTileSizeD / KPerThread;
                index_t start_x                   = (threadIdx.x % KThreadPerBlock) * KPerThread;

                if((start_x + KPerThread) <= rotary_dim)
                {
                    const bool is_left = (start_x + KPerThread) <= (rotary_dim / 2);

                    auto knew_other_window = knew_window;
                    move_tile_window(knew_other_window,
                                     {0, is_left ? rotary_dim / 2 : -(rotary_dim / 2)});
                    auto knew_other_tile = load_tile(knew_other_window);

                    move_tile_window(rotary_cos_window, {0, is_left ? 0 : -(rotary_dim / 2)});
                    auto rotary_cos_tile = load_tile(rotary_cos_window);

                    move_tile_window(rotary_sin_window, {0, is_left ? 0 : -(rotary_dim / 2)});
                    auto rotary_sin_tile = load_tile(rotary_sin_window);

                    constexpr index_t thread_buffer_size = decltype(knew_tile.thread_buf_)::size();
                    static_assert(thread_buffer_size % KPerThread == 0);
                    static_for<0, thread_buffer_size, 1>{}([&](auto idx) {
                        const auto curr  = type_convert<float>(knew_tile.thread_buf_[idx]);
                        const auto other = type_convert<float>(knew_other_tile.thread_buf_[idx]);

                        const auto cos = type_convert<float>(rotary_cos_tile.thread_buf_[idx]);
                        const auto sin = type_convert<float>(rotary_sin_tile.thread_buf_[idx]);

                        knew_tile.thread_buf_[idx] =
                            type_convert<KDataType>(curr * cos + other * (is_left ? -sin : sin));
                    });
                }
            }
        }
        // print_tile(knew_tile, 7);
        store_tile(k_dram_block_window, knew_tile);

        auto vnew_window =
            make_tile_window(vnew_dram_block_window.get_bottom_tensor_view(),
                             vnew_dram_block_window.get_window_lengths(),
                             vnew_dram_block_window.get_window_origin(),
                             Policy::template MakeVnewDramTileDistribution<Problem>());

        auto vnew_tile = [&]() {
            auto vnew = load_tile(vnew_window);
            return tile_elementwise_in(vnew_element_func, vnew);
        }();
        store_tile(v_dram_block_window, vnew_tile);

        // optionally apply rotary embedding to Q
        if constexpr(RotaryEnum != BlockRotaryEmbeddingEnum::NONE)
        {
            auto q_window = make_tile_window(q_dram_block_window.get_bottom_tensor_view(),
                                             q_dram_block_window.get_window_lengths(),
                                             q_dram_block_window.get_window_origin(),
                                             Policy::template MakeQDramTileDistribution<Problem>());

            auto q_tile = [&]() {
                auto q = load_tile(q_window);
                return tile_elementwise_in(q_element_func, q);
            }();
            print_tile(q_tile, 8);
            /// TODO: add rotary_cos/rotary_sin windows for Q (tile size: M0xK0)
            // We assume that each thread owns contiguous elements on head dimention. And we will
            // use the distribution to enable/disable threads in order to override knew_tile content
            if constexpr(RotaryEnum == BlockRotaryEmbeddingEnum::INTERLEAVED) {}
            else // RotaryEnum == BlockRotaryEmbeddingEnum::HALF_ROTATED
            {
            }

            store_tile(q_dram_block_window, q_tile);
        }
    }

    template <typename QDramBlockWindow,
              typename KDramBlockWindow,
              typename KnewDramBlockWindow,
              typename VDramBlockWindow,
              typename VnewDramBlockWindow,
              typename RotaryCosDramBlockWindow,
              typename RotarySinDramBlockWindow>
    CK_TILE_HOST_DEVICE auto
    operator()(QDramBlockWindow& q_dram_block_window,
               KDramBlockWindow& k_dram_block_window,
               const KnewDramBlockWindow& knew_dram_block_window,
               VDramBlockWindow& v_dram_block_window,
               const VnewDramBlockWindow& vnew_dram_block_window,
               const RotaryCosDramBlockWindow& rotary_cos_dram_block_window,
               const RotarySinDramBlockWindow& rotary_sin_dram_block_window,
               void* smem_ptr,
               index_t rotary_dim = 0) const
    {
        return operator()(q_dram_block_window,
                          identity{},
                          k_dram_block_window,
                          knew_dram_block_window,
                          identity{},
                          v_dram_block_window,
                          vnew_dram_block_window,
                          identity{},
                          rotary_cos_dram_block_window,
                          rotary_sin_dram_block_window,
                          smem_ptr,
                          rotary_dim);
    }
};

} // namespace ck_tile
