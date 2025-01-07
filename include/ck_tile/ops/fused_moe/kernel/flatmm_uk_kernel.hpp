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
// * Note on token_id_per_expert/sorted_token_ids_ptr data:
// currently we do not have topk information from the data of token_id_per_expert/sorted_token_ids_ptr.
// In some cases(like smooth-quant), we need topk information to indexing into tokens quant from 
// different expert smooth quant. So we modify the number stored inside token_id_per_expert/sorted_token_ids_ptr
//
//       32bit    0........23 24.....31 bit
//      (data) -> (token_id | topk_id)
// low 24 bit is for token id, top 8 bit is for topk id
//
// the input after smooth-quant is [token, topk, hidden_dim], originally it is [token, hidden_dim]
// the input scale for token is [topk, token, 1], the smooth-quant scale for first gemm is [expert, interm_dim]
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
struct FlatmmUkHostArgs
{
    const void* a_ptr;  // [m, k], input token
    const void* b_ptr;  // [m, k], input token
    const void* c_ptr;  // [m, k], output token
    const void* sa_ptr;  // [m, k], output token
    const void* sb_ptr;  // [m, k], output token
    void* d_ptr;        // [m, k], output token
    void* d_f16_ptr;        // [m, k], output token
    void* dbg_int_ptr;  // [m, k], output token
    void* dbg_fp8_ptr; // [m, k], output token
    void* dbg_f16_ptr; // [m, k], output token
    void* dbg_fp32_ptr; // [m, k], output token

    index_t hidden_size;       // K
    index_t intermediate_size; // N
    index_t num_tokens;        // M

    index_t num_experts;  // number of groups
    index_t topk;         // need this?
    index_t stride_token; // for input/output, stride for each row, should >= hidden_size
};

// This is scatter/gather b2b group-gemm
template <typename Pipeline_, typename Epilogue_>
struct FlatmmUkKernel
{
    using Pipeline    = remove_cvref_t<Pipeline_>;
    using Epilogue    = remove_cvref_t<Epilogue_>; // TODO: not used
    // static constexpr index_t kBlockPerCu = Pipeline::kBlockPerCu;
    // static_assert(kBlockPerCu > 0);

    using BlockShape = typename Pipeline::BlockShape; // this is FusedMoeGemmShape
    static constexpr index_t BlockSize_ = BlockShape::BlockSize;

    using ADataType            = typename Pipeline::Problem::ADataType;
    using GDataType            = typename Pipeline::Problem::GDataType;
    using DDataType            = typename Pipeline::Problem::AccDataType;
    using AccDataType          = typename Pipeline::Problem::AccDataType;
    using ODataType            = typename Pipeline::Problem::ODataType;
    using AScaleDataType       = typename Pipeline::Problem::AScaleDataType;
    using GScaleDataType       = typename Pipeline::Problem::GScaleDataType;
    using DScaleDataType       = typename Pipeline::Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Pipeline::Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Pipeline::Problem::TopkWeightDataType;
    using IndexDataType        = typename Pipeline::Problem::IndexDataType;
    using YDataType            = typename Pipeline::Problem::YDataType;

    using Traits                = typename Pipeline::Problem::Traits;
    static constexpr bool UseUK = true;

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
        const void* a_ptr;  // [m, k], input token
        const void* b_ptr;  // [m, k], input token
        const void* c_ptr;  // [m, k], output token
        const void* sa_ptr;  // [m, k], output token
        const void* sb_ptr;  // [m, k], output token
        void* d_ptr;        // [m, k], output token
        void* d_f16_ptr;        // [m, k], output token
        void* dbg_int_ptr;  // [m, k], output token
        void* dbg_fp8_ptr; // [m, k], output token
        void* dbg_f16_ptr; // [m, k], output token
        void* dbg_fp32_ptr; // [m, k], output token

        index_t hidden_size;       // K
        index_t intermediate_size; // N
        index_t num_tokens;        // M

        index_t num_experts;  // number of groups
        index_t topk;         // need this?
        index_t stride_token; // for input/output, stride for each row, should >= hidden_size
    };

    // TODO: switch karg based on
    using Kargs = FusedMoeGemmKargs;
    using Hargs = FlatmmUkHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        // TODO: hargs/kargs not guranteed to be the same
        return bit_cast<Kargs>(hargs);
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        index_t ms = ck_tile::integer_divide_ceil(hargs.num_tokens, BlockShape::Block_M);
        index_t ns = ck_tile::integer_divide_ceil(hargs.intermediate_size, BlockShape::Block_N);
        return dim3(ns, ms, 1);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(BlockSize_); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
#if 0
        if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
        {
            printf("[KERNEL] FlatmmUkKernel =====\n");
            printf("[KERNEL] blockDim: [%d, %d], gridDim: [%d, %d]\n",
            static_cast<int>(blockDim.x),
            static_cast<int>(blockDim.y),
            static_cast<int>(gridDim.x),
            static_cast<int>(gridDim.y));
            printf("[KERNEL] lds = %.3f (KB)\n", GetSmemSize() / 1024.0f);
        }

        [[maybe_unused]] uint32_t tidx = threadIdx.x; // 0~255
        [[maybe_unused]] uint32_t tidy = threadIdx.y; // 0~0
        [[maybe_unused]] uint32_t bidx = blockIdx.x;  // 0~1
        [[maybe_unused]] uint32_t bidy = blockIdx.y;  // 0~51
        [[maybe_unused]] uint32_t bdmx = blockDim.x;  // 256
        [[maybe_unused]] uint32_t bdmy = blockDim.y;  // 1
        [[maybe_unused]] uint32_t gdmx = gridDim.x;   // 2
        [[maybe_unused]] uint32_t gdmy = gridDim.y; // 52
        [[maybe_unused]] uint32_t gid = ((bdmx * bdmy) * gdmx) * bidy 
                                        + (bdmx * bdmy) * bidx 
                                        + bdmx * tidy
                                        + tidx;

        [[maybe_unused]]int * dbg_int = static_cast<int*>(kargs.dbg_int_ptr);
        [[maybe_unused]]short * dbg_bf16 = static_cast<short*>(kargs.dbg_bf16_ptr);
        [[maybe_unused]]float * dbg_fp32 = static_cast<float*>(kargs.dbg_fp32_ptr);

        dbg_int[gid] = -1;
        dbg_fp32[gid] = -1.0f;
#endif

        __shared__ CK_TILE_LDS_ADDR char smem[GetSmemSize()];

        Pipeline{}(kargs, smem);
    }
};

} // namespace ck_tile
