// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include <string>
#include <type_traits>

//
// [indexing implementation-1]
// using M_a as constexpr block_size to partition all tokens into different slices
// each slice map to one expert, and one expert can have multiple slices
// e.g. num_experts = 6, top_k=3, M_a = 4, input_tokens = 5
// before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
//                            tok-0      tok-1      tok-2      tok-3      tok-4
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float
//           number)
//
// token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 5, 5]]
//  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
// weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
//
// max_tokens_post_padded : top_k * input_tokens + num_experts * (M_a - 1)
// * this could be larger than actual, since actual tokens are on GPU
//
// sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6,
// 0, 1, 2, 5]
//                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|- exp-4 -|-
//                          exp-5 -|
// sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *,
// c, f, i, o]
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
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float
//           number)
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
//
//
namespace ck_tile {

// This is scatter/gather b2b group-gemm
template <typename TilePartitioner_, typename FusedMoePipeline_, typename EpiloguePipeline_>
struct FusedMoeKernel
{
    using TilePartitioner                         = ck_tile::remove_cvref_t<TilePartitioner_>;
    using FusedMoePipeline                        = ck_tile::remove_cvref_t<FusedMoePipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FusedMoePipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FusedMoePipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FusedMoePipeline::Problem::kBlockPerCu;

    using ADataType         = ck_tile::remove_cvref_t<typename FusedMoePipeline::ADataType>;
    using GDataType         = ck_tile::remove_cvref_t<typename FusedMoePipeline::GDataType>;
    using UDataType         = ck_tile::remove_cvref_t<typename FusedMoePipeline::UDataType>;
    using DDataType         = ck_tile::remove_cvref_t<typename FusedMoePipeline::DDataType>;
    using ODataType         = ck_tile::remove_cvref_t<typename FusedMoePipeline::ODataType>;
    using AccDataType       = ck_tile::remove_cvref_t<typename FusedMoePipeline::AccDataType>;
    using ScaleDataType     = ck_tile::remove_cvref_t<typename FusedMoePipeline::ScaleDataType>;
    using DLayout           = ck_tile::remove_cvref_t<typename FusedMoePipeline::DLayout>;
    using FusedMoeTileShape = ck_tile::remove_cvref_t<typename FusedMoePipeline::FusedMoeTileShape>;

    static constexpr bool kPadDimSize    = FusedMoePipeline::kPadDimSize;
    static constexpr bool kPadHiddenSize = FusedMoePipeline::kPadHiddenSize;

    static constexpr bool kPadSeqLenQ       = FusedMoePipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FusedMoePipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FusedMoePipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FusedMoePipeline::kPadHeadDimV;
    static constexpr auto BiasEnum          = FusedMoePipeline::BiasEnum;
    static constexpr bool kStoreLSE         = FusedMoePipeline::kStoreLSE;
    static constexpr bool kHasDropout       = FusedMoePipeline::kHasDropout;
    static constexpr bool kDoFp8StaticQuant = FusedMoePipeline::Problem::kDoFp8StaticQuant;
    using FmhaMask                 = ck_tile::remove_cvref_t<typename FusedMoePipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<ck_tile::fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<ck_tile::bf8_t> { static constexpr const char * name = "bf8"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        using bfs = typename FusedMoePipeline::BlockFmhaShape;
        using gbr = typename bfs::Gemm0BlockWarps;
        using gwt = typename bfs::Gemm0WarpTile;
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadSeqLenK) n += "sk";
            if (kPadHeadDimQ) n += "d";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_fwd_d") + _TS_(bfs::kK0BlockLength) + "_" + _SS_(t2s<ADataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_" + _SS_(TilePartitioner::name) + "_"
            "b" + _TS_(bfs::kM0) + "x" + _TS_(bfs::kN0) + "x" + _TS_(bfs::kK0) + "x" +
                    _TS_(bfs::kN1) + "x" + _TS_(bfs::kK1) + "x" + _TS_(bfs::kK0BlockLength) + "_" +
            "r" + _TS_(gbr::at(ck_tile::number<0>{})) + "x" + _TS_(gbr::at(ck_tile::number<1>{})) + "x" + _TS_(gbr::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(gwt::at(ck_tile::number<0>{})) + "x" + _TS_(gwt::at(ck_tile::number<1>{})) + "x" + _TS_(gwt::at(ck_tile::number<2>{})) + "_" +
            (kBlockPerCuInput == -1 ? "" : ("o" + _TS_(kBlockPerCu) + "_")) + _SS_(FusedMoePipeline::name) + "_" +
            "v" + (std::is_same_v<DLayout, ck_tile::tensor_layout::gemm::RowMajor> ? "r" : "c") + (pn.empty() ? "" : "_" + pn) +
            (BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? _SS_("") : (_SS_("_") + BlockAttentionBiasEnumToStr<BiasEnum>::name)) + 
            (kHasMask ? "_" + _SS_(FmhaMask::name) : "") + (kStoreLSE ? "_lse" : "" ) + (kHasDropout ? "_dropout" : "" ) + (kDoFp8StaticQuant ? "_squant" : "" );
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
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

        ck_tile::index_t dim_size;
        ck_tile::index_t hidden_size;
        ck_tile::index_t num_tokens;  // input number of tokens for current iteration
        ck_tile::index_t num_experts; // number of groups
        // ck_tile::index_t top_k;      // need this?

        ck_tile::index_t stride_a;
        ck_tile::index_t stride_g;
        ck_tile::index_t stride_u;
        ck_tile::index_t stride_d;
        ck_tile::index_t stride_o;

        ck_tile::index_t stride_g_expert;
        ck_tile::index_t stride_u_expert;
        ck_tile::index_t stride_d_expert;
    };

    using Kargs = FusedMoeCommonKargs; // std::conditional_t<kIsGroupMode, FusedMoeGroupModeKargs,
                                       // FusedMoeBatchModeKargs>;

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

        ck_tile::index_t dim_size;
        ck_tile::index_t hidden_size;
        ck_tile::index_t num_tokens;  // input number of tokens for current iteration
        ck_tile::index_t num_experts; // number of groups
        // ck_tile::index_t top_k;      // need this?

        ck_tile::index_t stride_a;
        ck_tile::index_t stride_g;
        ck_tile::index_t stride_u;
        ck_tile::index_t stride_d;
        ck_tile::index_t stride_o;

        ck_tile::index_t stride_g_expert;
        ck_tile::index_t stride_u_expert;
        ck_tile::index_t stride_d_expert;
    };
    using Hargs = FusedMoeCommonHargs;

    CK_TILE_HOST static constexpr ToKargs(const Hargs hargs) { return kargs; }

    CK_TILE_HOST static constexpr auto GridSize(index_t num_cu, index_t blocks_per_cu)
    {
        return TilePartitioner::GridSize(num_cu, blocks_per_cu);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FusedMoePipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        // __shared__ char smem_ptr[GetSmemSize()];
        ck_tile::index_t num_sorted_tiles = __builtin_amdgcn_readfirstlane(
            *reinterpret_cast<const ck_tile::index_t*>(kargs.num_sorted_tiles_ptr));
        ck_tile::index_t tile_id = __builtin_amdgcn_readfirstlane(blockIdx.x;);

        // persistent loop
        while(true)
        {
            const auto [sorted_tile_id, hidden_tile_id] =
                TilePartitioner{}(tile_id, num_sorted_tiles, kargs.hidden_size);
            if(sorted_tile_id >= num_sorted_tiles)
                return;

            ck_tile::index_t expert_id =
                __builtin_amdgcn_readfirstlane(reinterpret_cast<const ck_tile::index_t*>(
                    kargs.sorted_expert_ids_ptr)[sorted_tile_id]);

            // index along hidden_size
            ck_tile::index_t hidden_id =
                __builtin_amdgcn_readfirstlane(hidden_tile_id * FusedMoeTileShape::kN_g);

            const auto a_coord = FusedMoePipeline::GetAIndex(); // 2d thread offset, [i_row, i_col]
            const auto token_coord =
                a_coord[number<0>{}] + sorted_tile_id * FusedMoeTileShape::kM_a;

            index_t token_id =
                reinterpret_cast<const index_t*>(kargs.sorted_token_ids_ptr)[token_coord];
            ScaleDataType scale =
                reinterpret_cast<const ScaleDataType*>(kargs.sorted_weight_ptr)[token_coord];

            const auto a_gtile_window = [&]() {
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

                const auto a_gtile_window_ = make_tile_window(
                    a_gather_view_,
                    make_tuple(number<FusedMoeTileShape::kM_a>{}, number<FmhaPipeline::kK_a>{}),
                    {0, 0});
                return a_gtile_window_;
            }();

            const auto g_gtile_window = [&]() {
                const GDataType* g_ptr =
                    reinterpret_cast<const GDataType*>(kargs.g_ptr) +
                    static_cast<long_index_t>(expert_id) * kargs.stride_g_expert +
                    hidden_id * kargs.stride_g;
                const auto g_view_ = make_naive_tensor_view<address_space_enum::global>(
                    g_ptr,
                    make_tuple(kargs.hidden_size, kargs.dim_size),
                    make_tuple(kargs.stride_g, 1),
                    number<FusedMoePipeline::kAlignmentG>{},
                    number<1>{});
                const auto g_view_1_ = pad_tensor_view(
                    g_view_,
                    make_tuple(number<FusedMoeShape::kN_g>{}, number<FusedMoeShape::kK_a>{}),
                    sequence<kPadHiddenSize, kPadDimSize>{});

                const auto g_gtile_window_ = make_tile_window(
                    g_view_1_,
                    make_tuple(number<FusedMoeTileShape::kN_g>{}, number<FmhaPipeline::kK_a>{}),
                    {0, 0});
                return g_gtile_window_;
            }();

            const auto u_gtile_window = [&]() {
                const UDataType* u_ptr =
                    reinterpret_cast<const UDataType*>(kargs.u_ptr) +
                    static_cast<long_index_t>(expert_id) * kargs.stride_u_expert +
                    hidden_id * kargs.stride_u;
                const auto u_view_ = make_naive_tensor_view<address_space_enum::global>(
                    u_ptr,
                    make_tuple(kargs.hidden_size, kargs.dim_size),
                    make_tuple(kargs.stride_u, 1),
                    number<FusedMoePipeline::kAlignmentU>{},
                    number<1>{});
                const auto u_view_1_ = pad_tensor_view(
                    u_view_,
                    make_tuple(number<FusedMoeShape::kN_u>{}, number<FusedMoeShape::kK_a>{}),
                    sequence<kPadHiddenSize, kPadDimSize>{});
                const auto u_gtile_window_ = make_tile_window(
                    u_view_1_,
                    make_tuple(number<FusedMoeShape::kN_u>{}, number<FusedMoeShape::kK_a>{}),
                    {0, 0});
                return u_gtile_window_;
            }();

            const auto d_gtile_window = [&]() {
                const DDataType* d_ptr = [&]() {
                    if constexpr(std::is_same_v<DLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                            static_cast<long_index_t>(expert_id) * kargs.stride_d_expert +
                            hidden_id* kargs.stride_d;
                    }
                    else
                    {
                        reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                            static_cast<long_index_t>(expert_id) * kargs.stride_d_expert +
                            hidden_id;
                    }
                }();
                if constexpr(std::is_same_v<DLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                {
                    const auto d_view_ = make_naive_tensor_view<address_space_enum::global>(
                        d_ptr,
                        make_tuple(kargs.hidden_size, kargs.dim_size),
                        make_tuple(kargs.stride_d, 1),
                        number<FusedMoePipeline::kAlignmentD>{},
                        number<1>{});
                    const auto d_view_1_ = pad_tensor_view(
                        d_view_,
                        make_tuple(number<FusedMoeShape::kK_y>{}, number<FusedMoeShape::kN_d>{}),
                        sequence<kPadHiddenSize, kPadDimSize>{});

                    const auto d_gtile_window_ = make_tile_window(
                        d_view_1_,
                        make_tuple(number<FusedMoeShape::kK_y>{}, number<FusedMoeShape::kN_d>{}),
                        {0, 0});
                    return d_gtile_window_;
                }
                else
                {
                    const auto d_view_ = make_naive_tensor_view<address_space_enum::global>(
                        d_ptr,
                        make_tuple(kargs.dim_size, kargs.hidden_size),
                        make_tuple(kargs.stride_d, 1),
                        number<FusedMoePipeline::kAlignmentD>{},
                        number<1>{});
                    const auto d_view_1_ = pad_tensor_view(
                        d_view_,
                        make_tuple(number<FusedMoeShape::kN_d>{}, number<FusedMoeShape::kK_y>{}),
                        sequence<kPadHiddenSize, kPadDimSize>{});

                    const auto d_gtile_window_ = make_tile_window(
                        d_view_1_,
                        make_tuple(number<FusedMoeShape::kN_d>{}, number<FusedMoeShape::kK_y>{}),
                        {0, 0});
                    return d_gtile_window_;
                }
            }();

            auto o_gtile_window = [&]() {
                const ODataType* o_ptr = reinterpret_cast<const ODataType*>(kargs.o_ptr);
                const auto o_view_     = make_naive_tensor_view<address_space_enum::global>(
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

                const auto o_gtile_window_ = make_tile_window(
                    o_scatter_view_,
                    make_tuple(number<FusedMoeTileShape::kM_a>{}, number<FmhaPipeline::kK_a>{}),
                    {0, 0});
                return o_gtile_window_;
            }();

            // do compute yeah
            FusedMoePipeline{}(a_gtile_window,
                               g_gtile_window,
                               u_gtile_window,
                               d_gtile_window,
                               o_gtile_window,
                               scale);

            tile_id += gridDim.x;
        }
    }
};

} // namespace ck_tile
