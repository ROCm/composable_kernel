// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"
#include <string>
#include <type_traits>

// clang-format off
// [indexing implementation-1]
// using M_a as constexpr block_size to partition all tokens into different slices
// each slice map to one expert, and one expert can have multiple slices
// e.g. num_experts = 6, topk=3, M_a = 4, input_tokens = 5
// before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
//                            tok-0      tok-1      tok-2      tok-3      tok-4
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float number)
//
// token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 2, 5]]
//  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
// weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
//
// max_num_tokens_padded : topk * input_tokens + num_experts * (M_a - 1)
// * this could be larger than actual, since actual tokens are on GPU
//
// sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
//                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
// sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
//
// * length is max_num_tokens_padded, actual size is num_tokens_post_padded_ptr
//
// sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 4, 5]
// * length is (max_num_tokens_padded + block_size - 1) / block_size
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
//  1) sorted_token_ids_ptr [max_num_tokens_padded]
//  2) sorted_weight_ptr
//  3) sorted_expert_ids_ptr
//  4）num_tokens_post_padded_ptr/num_sorted_tiles_ptr (select one)
//
//   max_num_tokens_padded: opk_ids.numel() + num_experts * (block_size - 1)
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

// m: num_tokens (or token*input-batch)
// k: intermediate_size
// n: intermediate_size used between 2 FC (TP slice this)
// e: num expert
// if doing pre-shuffle
// nr : n / Block_Nr
// kr : k / Block_Kr
// w  : fattened 1d wave buffer
// struct FusedMoeGemmHostArgs
// {
//     const void* a_ptr;              // [m, k], input token
//     const void* a_scale_ptr;        // [m, 1], token scale
//     const void* g_ptr;              // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
//     const void* d_ptr;              // [e, n, k], pre-shuffle([e, nr, kr, w])
//     const void* g_scale_ptr;        // [e, 1, n], gate(up) scale
//     const void* d_scale_ptr;        // [e, 1, k], down scale
//     const void* y_smooth_scale_ptr; // [e, 1, n], smooth-quant-scale for 2nd gemm input
//     void* o_ptr;                    // [m, k], output token

//     const void* sorted_token_ids_ptr;  // [max_num_tokens_padded]
//     const void* sorted_weight_ptr;     // [max_num_tokens_padded]
//     const void* sorted_expert_ids_ptr; // [(max_num_tokens_padded + block_size - 1) / block_size]
//     const void* num_sorted_tiles_ptr;  // [1]

//     index_t hidden_size;       // k
//     index_t intermediate_size; // n / TP, for Gate. if Gate+Up, Down need divide by 2
//     index_t num_tokens;        // input number of tokens for current iteration
//     index_t num_experts;       // number of groups
//     index_t topk;              // need this?

//     index_t stride_token; // for input/output, stride for each row, should >= hidden_size
// };

// This is scatter/gather b2b group-gemm
template <typename Partitioner_, typename Pipeline_, typename Epilogue_>
struct FusedMoeGemmGlKernel
{
    using Partitioner = remove_cvref_t<Partitioner_>;
    using Pipeline    = remove_cvref_t<Pipeline_>;
    using Epilogue    = remove_cvref_t<Epilogue_>; // TODO: not used
    // static constexpr index_t kBlockPerCu = Pipeline::kBlockPerCu;
    // static_assert(kBlockPerCu > 0);

    using BlockShape = typename Pipeline::BlockShape; // this is FusedMoeGemmShape
    static constexpr index_t BlockSize_ = BlockShape::BlockSize;

    using ADataType            = typename Pipeline::Problem::ADataType;
    using GDataType            = typename Pipeline::Problem::GDataType;
    using DDataType            = typename Pipeline::Problem::DDataType;
    using AccDataType          = typename Pipeline::Problem::AccDataType;
    using ODataType            = typename Pipeline::Problem::ODataType;
    using AScaleDataType       = typename Pipeline::Problem::AScaleDataType;
    using GScaleDataType       = typename Pipeline::Problem::GScaleDataType;
    using DScaleDataType       = typename Pipeline::Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Pipeline::Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Pipeline::Problem::TopkWeightDataType;
    using IndexDataType        = typename Pipeline::Problem::IndexDataType;
    using YDataType            = typename Pipeline::Problem::YDataType;

    using Traits = typename Pipeline::Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<bf8_t> { static constexpr const char * name = "bf8"; };
    template <> struct t2s<int8_t> { static constexpr const char * name = "int8"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
#define _SS_ std::string
#define _TS_ std::to_string
        // clang-format off
        using S_ = BlockShape;

        auto prec_str = [&] () {
            std::string base_str = _SS_(t2s<ADataType>::name);
            if (!std::is_same_v<ADataType, GDataType>) {
                base_str += _SS_("_") + _SS_(t2s<GDataType>::name);
            }
            return base_str;
        }();

        return _SS_("fused_moe_") + _SS_(prec_str) + "_" +
             _TS_(S_::Block_M0) + "x" + _TS_(S_::Block_N0) + "x" + _TS_(S_::Block_K0) + "x" + _TS_(S_::Block_N1) + "_" +
             _TS_(S_::WarpPerBlock_M0) + "x" + _TS_(S_::WarpPerBlock_N0) + "x" + _TS_(S_::WarpPerBlock_K0) + "_" +
             _TS_(S_::Warp_M0) + "x" + _TS_(S_::Warp_N0) + "x" + _TS_(S_::Warp_K0) + "_" + _SS_(Pipeline::name);
#undef _SS_
#undef _TS_
        // clang-format on
    }

    struct FusedMoeGemmKargs
    {
        const void* a_ptr;              // [m, k], input token
        const void* a_scale_ptr;        // [m, 1], token scale
        const void* g_ptr;              // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
        const void* d_ptr;              // [e, n, k], pre-shuffle([e, nr, kr, w])
        const void* g_scale_ptr;        // [e, 1, n], gate(up) scale
        const void* d_scale_ptr;        // [e, 1, k], down scale
        const void* y_smooth_scale_ptr; // [e, 1, n], smooth-quant-scale for 2nd gemm input
        void* o_ptr;                    // [m, k], output token

        const void* sorted_token_ids_ptr;
        const void* sorted_weight_ptr;
        const void* sorted_expert_ids_ptr;
        const void* num_sorted_tiles_ptr;

        index_t hidden_size;       // k
        index_t intermediate_size; // n / TP, for Gate. if Gate+Up, Down need divide by 2
        index_t num_tokens;        // input number of tokens for current iteration
        index_t num_experts;       // number of groups
        index_t topk;              // need this?

        index_t stride_token; // for input/output, stride for each row, should >= hidden_size
        index_t max_num_tokens_padded; // size of sorted_token_ids_ptr
        void* c_ptr;
    };

    // TODO: switch karg based on
    using Kargs = FusedMoeGemmKargs;
    using Hargs = FusedMoeGemmHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        // TODO: hargs/kargs not guranteed to be the same
        return bit_cast<Kargs>(hargs);
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        // constexpr index_t block_m = BlockShape::Block_M0;
        int max_num_tokens_padded = hargs.max_num_tokens_padded;
        // hargs.topk * hargs.num_tokens + hargs.num_experts * block_m - hargs.topk;
        // printf("xxx max_num_tokens_padded:%d\n", max_num_tokens_padded);
        return Partitioner::GridSize(max_num_tokens_padded, hargs.intermediate_size);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(BlockSize_); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        // __shared__ char smem_ptr[GetSmemSize()];
        IndexDataType num_sorted_tiles = __builtin_amdgcn_readfirstlane(
            *reinterpret_cast<const IndexDataType*>(kargs.num_sorted_tiles_ptr));
        constexpr index_t hidden_radio_0 = IsGateOnly ? 1 : 2;

        index_t expert_stride_0 = kargs.intermediate_size * hidden_radio_0 * kargs.hidden_size;
        index_t expert_stride_1 = kargs.intermediate_size * kargs.hidden_size;

        __shared__ CK_TILE_LDS_ADDR ADataType smem[GetSmemSize()];

        // note this is in unit of tile, need multiple tile size to get the index(block_m and
        // block_n)
        const auto [sorted_tile_id, intermediate_tile_id] =
            Partitioner{}(num_sorted_tiles, kargs.intermediate_size);
        if(sorted_tile_id >= num_sorted_tiles)
            return;

        const IndexDataType expert_id = __builtin_amdgcn_readfirstlane(
            reinterpret_cast<const IndexDataType*>(kargs.sorted_expert_ids_ptr)[sorted_tile_id]);

        index_t idx_m0 = __builtin_amdgcn_readfirstlane(sorted_tile_id * BlockShape::Block_M0);
        index_t idx_n0 =
            __builtin_amdgcn_readfirstlane(intermediate_tile_id * BlockShape::Block_N0);

        const index_t* sorted_token_ids_ptr =
            reinterpret_cast<const index_t*>(kargs.sorted_token_ids_ptr);

        const auto a_window = [&]() {
            // A is already pre-padded in previous kernel
            const ADataType* a_ptr = reinterpret_cast<const ADataType*>(kargs.a_ptr);
            const auto a_view_     = make_naive_tensor_view<address_space_enum::global>(
                a_ptr,
                make_tuple(kargs.num_tokens, kargs.hidden_size),
                make_tuple(kargs.stride_token, 1),
                number<Pipeline::kAlignmentA>{},
                number<1>{});

            // gather is here use indexing transform
            const auto a_gather_view_ = transform_tensor_view(
                a_view_,
                make_tuple(make_indexing_transform_with_adaptor(
                               kargs.max_num_tokens_padded,
                               indexing_adaptor<index_t>{sorted_token_ids_ptr}),
                           make_pass_through_transform(kargs.hidden_size)),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            const auto a_window_ = make_tile_window(
                a_gather_view_,
                make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
                {idx_m0, 0});
            return a_window_;
        }();

        const auto g_window = [&]() {
            const GDataType* g_ptr = reinterpret_cast<const GDataType*>(kargs.g_ptr) +
                                     static_cast<long_index_t>(expert_id) * expert_stride_0;
            const auto g_view_ = make_naive_tensor_view<address_space_enum::global>(
                g_ptr,
                make_tuple(kargs.intermediate_size, kargs.hidden_size),
                make_tuple(kargs.hidden_size, 1),
                number<Pipeline::kAlignmentG>{},
                number<1>{});

            const auto g_view_1_ = pad_tensor_view(
                g_view_,
                make_tuple(number<BlockShape::Block_N0>{}, number<BlockShape::Block_K0>{}),
                sequence<PadIntermediateSize, PadHiddenSize>{});

            const auto g_window_ = make_tile_window(
                g_view_1_,
                make_tuple(number<BlockShape::Block_N0>{}, number<BlockShape::Block_K0>{}),
                {idx_n0, 0});

            return g_window_;
        }();

        auto c_window = [&]() {
            YDataType* c_ptr = reinterpret_cast<YDataType*>(kargs.c_ptr);
            // note interm_idx_nr is along the gemm-k dim of 2nd gemm
            auto c_view_ = make_naive_tensor_view<address_space_enum::global>(
                c_ptr,
                make_tuple(kargs.num_tokens, kargs.intermediate_size),
                make_tuple(kargs.intermediate_size, 1),
                number<Pipeline::kAlignmentD>{},
                number<1>{});

            auto c_window_ = make_tile_window(
                c_view_,
                make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_N0>{}),
                {0, 0});
            return c_window_;
        }();

        const auto d_window = [&]() {
            const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                     static_cast<long_index_t>(expert_id) * expert_stride_1;
            // note interm_idx_nr is along the gemm-k dim of 2nd gemm
            const auto d_view_ = make_naive_tensor_view<address_space_enum::global>(
                d_ptr,
                make_tuple(kargs.hidden_size, kargs.intermediate_size),
                make_tuple(kargs.intermediate_size, 1),
                number<Pipeline::kAlignmentD>{},
                number<1>{});

            const auto d_view_1_ = pad_tensor_view(
                d_view_,
                make_tuple(number<BlockShape::Block_N1>{}, number<BlockShape::Block_K1>{}),
                sequence<PadHiddenSize, PadIntermediateSize>{});

            const auto d_window_ = make_tile_window(
                d_view_1_,
                make_tuple(number<BlockShape::Block_N1>{}, number<BlockShape::Block_K1>{}),
                {0, idx_n0});
            return d_window_;
        }();

        auto o_window = [&]() {
            ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr);
            auto o_view_     = make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::atomic_add>(
                o_ptr,
                make_tuple(kargs.num_tokens, kargs.hidden_size),
                make_tuple(kargs.stride_token, 1),
                number<Pipeline::kAlignmentO>{},
                number<1>{});

            // gather is here
            auto o_scatter_view_ = transform_tensor_view(
                o_view_,
                make_tuple(make_indexing_transform_with_adaptor(
                               kargs.max_num_tokens_padded,
                               indexing_adaptor<index_t>{sorted_token_ids_ptr}),
                           make_pass_through_transform(kargs.hidden_size)),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            auto o_padd_view_ = pad_tensor_view(
                o_scatter_view_,
                make_tuple(number<BlockShape::Block_M1>{}, number<BlockShape::Block_N1>{}),
                sequence<true, 0>{});

            auto o_window_ = make_tile_window(
                o_padd_view_,
                make_tuple(number<BlockShape::Block_M1>{}, number<BlockShape::Block_N1>{}),
                {idx_m0, 0});
            return o_window_;
        }();

        const auto w_window = [&]() {
            const TopkWeightDataType* w_ptr =
                reinterpret_cast<const TopkWeightDataType*>(kargs.sorted_weight_ptr);
            const auto w_view_ = make_naive_tensor_view<address_space_enum::global>(
                w_ptr,
                make_tuple(kargs.max_num_tokens_padded),
                make_tuple(1),
                number<1>{},
                number<1>{});

            const auto w_window_ =
                make_tile_window(w_view_, make_tuple(number<BlockShape::Block_M0>{}), {idx_m0});
            return w_window_;
        }();

        // do compute yeah
        Pipeline{}(a_window,
                   g_window,
                   d_window,
                   w_window,
                   o_window,
                   smem,
                   kargs.hidden_size,
                   kargs.intermediate_size,
                   c_window);
    }
};

} // namespace ck_tile
