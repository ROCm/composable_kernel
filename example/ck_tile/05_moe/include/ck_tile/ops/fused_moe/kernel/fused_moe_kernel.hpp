// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include <string>
#include <type_traits>

// clang-format off
// [indexing implementation-1]
// using M_a as constexpr block_size to partition all tokens into different slices
// each slice map to one expert, and one expert can have multiple slices
// e.g. num_experts = 6, top_k=3, M_a = 4, input_tokens = 5
// before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
//                            tok-0      tok-1      tok-2      tok-3      tok-4
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float number)
//
// token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 5, 5]]
//  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
// weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
//
// max_tokens_post_padded : top_k * input_tokens + num_experts * (M_a - 1)
// * this could be larger than actual, since actual tokens are on GPU
//
// sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
//                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
// sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
//
// * length is max_tokens_post_padded, actual size is num_tokens_post_padded_ptr
//
// sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 4, 5]
// * length is (max_tokens_post_padded + block_size - 1) / block_size
//
// num_tokens_post_padded_ptr : [28]
// num_sorted_tiles_ptr : [7]
//
// * different from vLLM
//   1) token_id stored in sorted_token_ids_ptr is actual token_id, not token_id*top_K expanded id
//   2）need sorted_weight_ptr
//   3) use num_sorted_tiles_ptr, already divided by M_a
//
// * below used for indexing
//  1) sorted_token_ids_ptr
//  2) sorted_weight_ptr
//  3) sorted_expert_ids_ptr
//  4）num_tokens_post_padded_ptr/num_sorted_tiles_ptr (select one)
//
//
// [indexing implementation-2]
// before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
//                            tok-0      tok-1      tok-2      tok-3      tok-4
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float number)
//
// we generate original rol/col id as
//              topk_rc_ids : [[0, 5, A], [1, 6, B], [2, 7, C], [3, 8, D], [4, 9, E]]
// let x be one element of above, we can get:
//          tpok_row_id(token_id) = x % num_tokens(5)
//         tpok_col_id(expert_Id) = x / num_tokens
// topk_row_id/col_id can be used to access original topk_ids/topk_weight
//
// token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 5, 5]]
//  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
// weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
//
// we can get permuted_rc_ids:
//                          [[0], [2, 3, 4], [1, 8], [5, 6, 7, D, 9], [], [A, B, C, E]]
//
//
// clang-format on
//
namespace ck_tile {

// This is scatter/gather b2b group-gemm
template <typename TilePartitioner_, typename FusedMoePipeline_, typename EpiloguePipeline_>
struct FusedMoeKernel
{
    using TilePartitioner                = remove_cvref_t<TilePartitioner_>;
    using FusedMoePipeline               = remove_cvref_t<FusedMoePipeline_>;
    using EpiloguePipeline               = remove_cvref_t<EpiloguePipeline_>; // TODO: not used
    static constexpr index_t kBlockSize  = FusedMoePipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = FusedMoePipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr index_t kBlockPerCuInput = FusedMoePipeline::Problem::kBlockPerCu;

    using ADataType         = remove_cvref_t<typename FusedMoePipeline::ADataType>;
    using GDataType         = remove_cvref_t<typename FusedMoePipeline::GDataType>;
    using UDataType         = remove_cvref_t<typename FusedMoePipeline::UDataType>;
    using DDataType         = remove_cvref_t<typename FusedMoePipeline::DDataType>;
    using ODataType         = remove_cvref_t<typename FusedMoePipeline::ODataType>;
    using AccDataType       = remove_cvref_t<typename FusedMoePipeline::AccDataType>;
    using ScaleDataType     = remove_cvref_t<typename FusedMoePipeline::ScaleDataType>;
    using FusedMoeTileShape = remove_cvref_t<typename FusedMoePipeline::FusedMoeTileShape>;

    static constexpr bool kPadDimSize    = FusedMoePipeline::kPadDimSize;
    static constexpr bool kPadHiddenSize = FusedMoePipeline::kPadHiddenSize;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<bf8_t> { static constexpr const char * name = "bf8"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off

        // clang-format on
    }

    template <index_t I> // to avoid duplicated base class prblem, introduce an template
                         // arg
    struct FusedMoeEmptyKargs
    {
    };

    // tensors:
    // 1. act  (A): input feature map
    // 2. gate (G): B matrix for first gemm, output will do activation(Silu)
    // 3. up   (U): B matrix for first gemm
    // 4. down (D): B matrix for second gemm
    struct FusedMoeCommonKargs
    {
        const void* a_ptr;
        const void* g_ptr;
        const void* u_ptr;
        const void* d_ptr;
        // const void* w_ptr;  //topk-weight
        void* o_ptr;

        const void* sorted_token_ids_ptr;
        const void* sorted_weight_ptr;
        const void* sorted_expert_ids_ptr;
        // const void* num_tokens_post_padded_ptr;
        const void* num_sorted_tiles_ptr;

        index_t dim_size;
        index_t hidden_size;
        index_t num_tokens;  // input number of tokens for current iteration
        index_t num_experts; // number of groups
        // index_t top_k;      // need this?

        index_t stride_a;
        index_t stride_gu; // assume g/u have same stride
        // index_t stride_u;
        index_t stride_d;
        index_t stride_o;

        index_t stride_expert_gu; // assume g/u have same stride
        index_t stride_expert_d;
    };

    struct FusedMoeMatrixCoreShuffleKargs : public FusedMoeCommonKargs
    {
        // batch*nr_0*kr_0*waveflattern, now stride_kr is the stride in above
        index_t stride_gu_nr;
        index_t stride_d_nr;
    };

    // TODO: switch karg based on
    using Kargs = FusedMoeMatrixCoreShuffleKargs;

    // host args are used inside host API
    // and should be POD data structure
    struct FusedMoeCommonHargs
    {
        const void* a_ptr;
        const void* g_ptr;
        const void* u_ptr;
        const void* d_ptr;
        // const void* w_ptr;  //topk-weight
        void* o_ptr;

        const void* sorted_token_ids_ptr;
        const void* sorted_weight_ptr;
        const void* sorted_expert_ids_ptr;
        // const void* num_tokens_post_padded_ptr;
        const void* num_sorted_tiles_ptr;

        index_t dim_size;
        index_t hidden_size;
        index_t num_tokens;  // input number of tokens for current iteration
        index_t num_experts; // number of groups
        // index_t top_k;      // need this?

        index_t stride_a;
        index_t stride_g;
        index_t stride_u;
        index_t stride_d;
        index_t stride_o;

        index_t stride_expert_gu;
        index_t stride_expert_d;
    };
    using Hargs = FusedMoeCommonHargs;

    CK_TILE_HOST static constexpr auto ToKargs(const Hargs hargs) { return hargs; }

    CK_TILE_HOST static constexpr auto GridSize(index_t num_cu, index_t blocks_per_cu)
    {
        return TilePartitioner::GridSize(num_cu, blocks_per_cu);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(FusedMoePipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        // __shared__ char smem_ptr[GetSmemSize()];
        index_t num_sorted_tiles = __builtin_amdgcn_readfirstlane(
            *reinterpret_cast<const index_t*>(kargs.num_sorted_tiles_ptr));

        index_t nr_0 = kargs.hidden_size / FusedMoePipeline::kBlockNr_0;
        index_t kr_0 = kargs.dim_size / FusedMoePipeline::kBlockKr_0;
        index_t nr_1 = kargs.dim_size / FusedMoePipeline::kBlockNr_1;
        index_t kr_1 = kargs.hidden_size / FusedMoePipeline::kBlockKr_1;

        __shared__ CK_TILE_LDS_ADDR ADataType smem_0[FusedMoePipeline::GetSmemSizeSingleBuffer()];
        __shared__ CK_TILE_LDS_ADDR ADataType smem_1[FusedMoePipeline::GetSmemSizeSingleBuffer()];

        // persistent loop
        // while(true)
        {
            const auto [sorted_tile_id, hidden_tile_id] =
                TilePartitioner{}(num_sorted_tiles, kargs.hidden_size);
            if(sorted_tile_id >= num_sorted_tiles)
                return;

            index_t expert_id = __builtin_amdgcn_readfirstlane(
                reinterpret_cast<const index_t*>(kargs.sorted_expert_ids_ptr)[sorted_tile_id]);

            // index along hidden_size
            index_t hidden_id =
                __builtin_amdgcn_readfirstlane(hidden_tile_id * FusedMoeTileShape::kBlockN_0);
            index_t hidden_id_nr = __builtin_amdgcn_readfirstlane(hidden_tile_id * block_nr);

            const auto a_coord = FusedMoePipeline::GetAIndex(); // 2d thread offset, [i_row, i_col]
            const auto sorted_token_id =
                a_coord[number<0>{}] + sorted_tile_id * FusedMoeTileShape::kBlockM_0;

            index_t token_id =
                reinterpret_cast<const index_t*>(kargs.sorted_token_ids_ptr)[sorted_token_id];
            ScaleDataType scale =
                reinterpret_cast<const ScaleDataType*>(kargs.sorted_weight_ptr)[sorted_token_id];

            const auto a_gtile_window = [&]() {
                // A is already pre-padded in previous kernel
                const ADataType* a_ptr = reinterpret_cast<const ADataType*>(kargs.a_ptr);
                const auto a_view_     = make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.num_tokens, kargs.dim_size),
                    make_tuple(kargs.stride_a, 1),
                    number<FusedMoePipeline::kAlignmentA>{},
                    number<1>{});

                // gather is here
                const auto a_gather_view_ = transform_tensor_view(
                    a_view_,
                    make_tuple(make_indexing_transform(kargs.num_tokens, token_id),
                               make_pass_through_transform(kargs.dim_size)),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                const auto a_gtile_window_ =
                    make_tile_window(a_gather_view_,
                                     make_tuple(number<FusedMoeTileShape::kBlockM_0>{},
                                                number<FusedMoePipeline::kBlockK_0>{}),
                                     {0, 0});
                return a_gtile_window_;
            }();

            // TODO: gtile using NSub to have less register pressure
            const auto g_gtile_window = [&]() {
                const GDataType* g_ptr =
                    reinterpret_cast<const GDataType*>(kargs.g_ptr) +
                    static_cast<long_index_t>(expert_id) * kargs.stride_expert_gu +
                    hidden_id_nr * kargs.stride_gu_nr;
                const auto g_view_ = make_naive_tensor_view<address_space_enum::global>(
                    g_ptr,
                    make_tuple(nr_0, kr_0, number<FusedMoePipeline::kBlockWaveFlatten>{}),
                    make_tuple(stride_gu_nr, number<FusedMoePipeline::kBlockWaveFlatten>{}, 1),
                    number<FusedMoePipeline::kAlignmentG>{},
                    number<1>{});
                const auto g_view_1_ =
                    pad_tensor_view(g_view_,
                                    make_tuple(number<FusedMoePipeline::kBlockNr_0>{},
                                               number<FusedMoePipeline::kBlockKr_0>{},
                                               number<FusedMoePipeline::kBlockWaveFlatten>{}),
                                    sequence<kPadHiddenSize, kPadDimSize, 0>{});

                const auto g_gtile_window_ =
                    make_tile_window(g_view_1_,
                                     make_tuple(number<FusedMoeTileShape::kBlockNr_0>{},
                                                number<FusedMoePipeline::kBlockKr_0>{},
                                                number<FusedMoePipeline::kBlockWaveFlatten>{}),
                                     {0, 0, 0});
                return g_gtile_window_;
            }();

            const auto u_gtile_window = [&]() {
                const UDataType* u_ptr =
                    reinterpret_cast<const UDataType*>(kargs.u_ptr) +
                    static_cast<long_index_t>(expert_id) * kargs.stride_expert_gu +
                    hidden_id_nr * kargs.stride_gu_nr;
                const auto u_view_ = make_naive_tensor_view<address_space_enum::global>(
                    u_ptr,
                    make_tuple(nr_0, kr_0, number<FusedMoePipeline::kBlockWaveFlatten>{}),
                    make_tuple(stride_gu_nr, number<FusedMoePipeline::kBlockWaveFlatten>{}, 1),
                    number<FusedMoePipeline::kAlignmentU>{},
                    number<1>{});
                const auto u_view_1_ =
                    pad_tensor_view(u_view_,
                                    make_tuple(number<FusedMoePipeline::kBlockNr_0>{},
                                               number<FusedMoePipeline::kBlockKr_0>{},
                                               number<FusedMoePipeline::kBlockWaveFlatten>{}),
                                    sequence<kPadHiddenSize, kPadDimSize, 0>{});
                const auto u_gtile_window_ =
                    make_tile_window(u_view_1_,
                                     make_tuple(number<FusedMoeTileShape::kBlockNr_0>{},
                                                number<FusedMoePipeline::kBlockKr_0>{},
                                                number<FusedMoePipeline::kBlockWaveFlatten>{}),
                                     {0, 0, 0});
                return u_gtile_window_;
            }();

            const auto d_gtile_window = [&]() {
                const DDataType* d_ptr = [&]() {
                    reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                        static_cast<long_index_t>(expert_id) * kargs.stride_expert_d +
                        hidden_id_nr* kargs.stride_d_nr;
                }();

                const auto d_view_ = make_naive_tensor_view<address_space_enum::global>(
                    d_ptr,
                    make_tuple(nr_1, kr_1, FusedMoePipeline::kBlockWaveFlatten),
                    make_tuple(kargs.stride_d_nr, FusedMoePipeline::kBlockWaveFlatten, 1),
                    number<FusedMoePipeline::kAlignmentD>{},
                    number<1>{});
                const auto d_view_1_ =
                    pad_tensor_view(d_view_,
                                    make_tuple(number<FusedMoePipeline::kBlockNr_1>{},
                                               number<FusedMoePipeline::kBlockKr_1>{},
                                               number<FusedMoePipeline::kBlockWaveFlatten>{}),
                                    sequence<kPadDimSize, kPadHiddenSize, 0>{});

                const auto d_gtile_window_ =
                    make_tile_window(d_view_1_,
                                     make_tuple(number<FusedMoePipeline::kBlockNr_1>{},
                                                number<FusedMoePipeline::kBlockKr_1>{},
                                                number<FusedMoePipeline::kBlockWaveFlatten>{}),
                                     {0, 0, 0});
                return d_gtile_window_;
            }();

            auto o_gtile_window = [&]() {
                const ODataType* o_ptr = reinterpret_cast<const ODataType*>(kargs.o_ptr);
                const auto o_view_     = make_naive_tensor_view<address_space_enum::global,
                                                            memory_operation_enum::atomic_add>(
                    o_ptr,
                    make_tuple(kargs.num_tokens, kargs.dim_size),
                    make_tuple(kargs.stride_o, 1),
                    number<FusedMoePipeline::kAlignmentO>{},
                    number<1>{});

                // gather is here
                const auto o_scatter_view_ = transform_tensor_view(
                    o_view_,
                    make_tuple(make_indexing_transform(kargs.num_tokens, token_id),
                               make_pass_through_transform(kargs.dim_size)),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                const auto o_gtile_window_ =
                    make_tile_window(o_scatter_view_,
                                     make_tuple(number<FusedMoeTileShape::kBlockM_0>{},
                                                number<FusedMoePipeline::kBlockN_1>{}),
                                     {0, 0});
                return o_gtile_window_;
            }();

            // do compute yeah
            FusedMoePipeline{}(a_gtile_window,
                               g_gtile_window,
                               u_gtile_window,
                               d_gtile_window,
                               o_gtile_window,
                               scale,
                               smem_0,
                               smem_1,
                               kargs.dim_size,
                               kargs.hidden_size);
            // tile_id += gridDim.x;
            // epilogue not used
        }
    }
};

} // namespace ck_tile
