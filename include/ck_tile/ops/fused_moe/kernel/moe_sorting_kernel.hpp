// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

#define MOE_SORTING_MOCK_ID(token_id_, topk_id_) \
    static_cast<uint32_t>(((token_id_)&0x00ffffff) | (((topk_id_)&0xff) << 24))

#ifndef MOE_SORTING_USE_EX_KERNEL
#define MOE_SORTING_USE_EX_KERNEL 1
#endif

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
// max_num_tokens_padded : topk * input_tokens + num_experts * M_a - topk (updated)
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
// the input after smooth-quant is [topk, token, hidden_dim], originally it is [token, hidden_dim]
// the input scale for token is [topk, token, 1], the smooth-quant scale for first gemm is [expert, interm_dim]
//
// sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 4, 5]
// * length is (max_num_tokens_padded + block_size - 1) / block_size
//
// num_tokens_post_padded_ptr : [28]
// num_sorted_tiles_ptr : [7]
//
// skip_experts_with_zero_tokens(SkipExpertsWithZeroTokens)
// if enabled, the expert with no tokens will be skipped, in stead of padding to at least 1 unit_size(M_a)
//
//                                            (pack below tensor, skip element marked with `-`)
//                           Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  -  -  -  -  Y  Y  Y  Y
// sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
//                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
// sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
//                          
//
// sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 5]
// num_tokens_post_padded_ptr : [24]
// 
// * local_expert_mask : indicate local expert mask used on current GPU (used for EP case)
//   and modify the output expert-ID, because we will only have enbaled expert on specific GPU.
//   we call expert input to this kernel as "global expert id", output as "local expert id"
//
// * local_expert_mask : [1, 0, 1, 1, 0, 1] (mask out expert-id=1, 4)
//
//                                            (pack below tensor, skip element marked with `-`)
//                         Y  Y  Y  Y  -  -  -  -  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  -  -  -  -  Y  Y  Y  Y
// sorted_token_ids_ptr : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
//                        |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
// sorted_weight_ptr    : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
//
// sorted_expert_ids_ptr  : [0, 1, 2, 2, 3] (note original it was exper-id= 0, 2, 3, 5, but we produce "local expert id")
// num_tokens_post_padded_ptr : [20]
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


CK_TILE_HOST constexpr auto moe_sorting_get_smem_row_col(int num_tokens_, int num_experts_)
{
    /*               num_experts + 1
    *   +--------------------------------------+
    *   |                                      |
    *   |                                      |
    *   |                                      |    * -> sub-tokens
    *   |                                      |
    *   |                                      |
    *   +--------------------------------------+
    *   |                                      |    2 -> cumsum buffer
    *   +--------------------------------------+
    *
    */
    int smem_cols = num_experts_ + 1;  // usually experts is power of 2. padding here
    int smem_rows = [&](){
        index_t target_occupancy_ = 2;
        constexpr index_t total_ = 65536 / sizeof(int);
        constexpr index_t sub_unroll = 8;
        constexpr index_t cumsum_bufs = 2;  // 1 for cumsum, 1 for cnt
        // at lease 2 lines, one for sub_token unroll, one for cumsum
        // should be enough
        if ((total_ / target_occupancy_) < ((cumsum_bufs+sub_unroll) * smem_cols)) {
            if ((total_ / 1) < ((cumsum_bufs+sub_unroll) * smem_cols))
                throw std::runtime_error("too many num_experts, can't allocate smem");
            target_occupancy_ = 1;
        }
        int r = total_ / target_occupancy_ / smem_cols;

        // round to sub_unroll multipl
        int r_for_sub_token = r - cumsum_bufs;
        r_for_sub_token = min(r_for_sub_token, num_tokens_);
        r_for_sub_token = (r_for_sub_token + sub_unroll - 1) / sub_unroll * sub_unroll;
        r_for_sub_token = max(r_for_sub_token, 1);

        if(r_for_sub_token > 1)
        {
            int r_unroll_ = r_for_sub_token / sub_unroll;
            

            // round to 1x/2x/4x/8x number of sub_unroll
            int clz_ = __builtin_clz(r_unroll_); // 0b1:31 0b2:30, 0b3:30, 0b4:29
            int mask_ = (1 << (31 - clz_)) - 1;

            
            mask_ = mask_ > 0b111 ? 0b111 : mask_;  //clamp to 8x at most
            mask_ = ~mask_;
            //printf("r_unroll_:%d, clz:%d, mask:%x\n", r_unroll_, clz_, mask_); fflush(stdout);

            r_for_sub_token = (r_unroll_ & mask_) * sub_unroll;
        }

        // final check
        if( (r_for_sub_token + cumsum_bufs * smem_cols *  target_occupancy_ ) >= total_ ) {
            throw std::runtime_error("can't run this kernel, request LDS over size");
        }

        return r_for_sub_token + cumsum_bufs;
    }();

    // printf("r:%d, c:%d\n", smem_rows, smem_cols);

    return ck_tile::make_tuple(smem_rows, smem_cols);
}

struct MoeSortingHostArgs
{
    const void* p_topk_ids;     // [token, topk]
    const void* p_weights;      // [token, topk]

    const void* p_local_expert_mask;

    void* p_sorted_token_ids;
    void* p_sorted_weights;
    void* p_sorted_expert_ids;
    void* p_total_tokens_post_pad;
    // we fused the setzero of output of fused-moe buffer
    // set this pointer to nullptr will skip this operation
    void* p_moe_buf;
    index_t tokens;
    index_t unit_size;      // this is the M_a of fused-moe kernel
    index_t num_experts;
    index_t topk;
    index_t moe_buf_bytes;  // byte size of p_moe_buf
};

template <typename Problem_>
struct MoeSortingKernel
{
    using Problem = remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;

    struct Kargs
    {
        const void* p_topk_ids;
        const void* p_weights;
        const void* p_local_expert_mask;
        void* p_sorted_token_ids;
        void* p_sorted_weights;
        void* p_sorted_expert_ids;
        void* p_total_tokens_post_pad;
        void* p_moe_buf;
        index_t tokens;
        index_t num_experts;
        index_t moe_buf_bytes;

        index_t tokens_per_thread;
        index_t smem_rows;
        mdiv unit_size_mdiv;
        mdiv topk_mdiv;
        mdiv expert_mdiv;
        // mdiv sub_tokens_mdiv;
    };

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& h)
    {
        // TODO: assume num-experts not too much
        return dim3(1 + ck_tile::integer_divide_ceil(h.moe_buf_bytes, BlockSize(h).x * 16));
    }

    CK_TILE_HOST static constexpr auto BlockSize(const Hargs& h)
    {
#if MOE_SORTING_USE_EX_KERNEL
        (void)h;
        return dim3(256);
#else
        return dim3(ck_tile::integer_least_multiple(h.num_experts, ck_tile::get_warp_size()));
#endif
    }

    // in byte
    CK_TILE_HOST static constexpr auto GetSmemSize(const Hargs& h)
    {
#if MOE_SORTING_USE_EX_KERNEL
        auto [smem_rows, smem_cols] = moe_sorting_get_smem_row_col(h.tokens, h.num_experts);
        return smem_rows * smem_cols * sizeof(int);
#else
        const auto blocks = BlockSize(h);
        // usually num_experts is power of 2, we pad 1 dword here for the row-size
        return ((blocks.x + 1) * (h.num_experts + 1) + (h.num_experts + 1)) * sizeof(index_t);
#endif
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_topk_ids              = h.p_topk_ids;
        k.p_weights               = h.p_weights;
        k.p_local_expert_mask     = h.p_local_expert_mask;
        k.p_sorted_token_ids      = h.p_sorted_token_ids;
        k.p_sorted_weights        = h.p_sorted_weights;
        k.p_sorted_expert_ids     = h.p_sorted_expert_ids;
        k.p_moe_buf               = h.p_moe_buf;
        k.p_total_tokens_post_pad = h.p_total_tokens_post_pad;
        k.tokens                  = h.tokens;
        k.num_experts             = h.num_experts;
        k.moe_buf_bytes           = h.moe_buf_bytes;

        const auto blocks   = BlockSize(h);
        k.tokens_per_thread = integer_divide_ceil(h.tokens * h.topk, blocks.x);
        k.unit_size_mdiv    = mdiv{static_cast<uint32_t>(h.unit_size)};
        k.topk_mdiv         = mdiv{static_cast<uint32_t>(h.topk)};
        k.smem_rows         = [&](){
            auto [r_, c_] = moe_sorting_get_smem_row_col(h.tokens, h.num_experts);
            (void) c_;
            return r_;
        }();
        k.expert_mdiv      = mdiv{static_cast<uint32_t>(h.num_experts)};
        // k.sub_tokens_mdiv  = mdiv{static_cast<uint32_t>(k.smem_rows - 1)};
        return k;
    }

    // [a, b, c, d....] -> [a, a+b, a+b+c, a+b+c+d, ....]
    // NOTE: wave_size need at least be 16!! dpp 16 is one row
    template <typename data_t, int wave_size>
    __device__ inline void wave_cumsum(data_t& thread_data) const
    {
        // wave_size must be power of 2
        constexpr int row_mask    = 0xf;
        constexpr int bank_mask   = 0xf;
        constexpr bool bound_ctrl = true;   // ! out-of-bound is zero !
        auto reduce_op = [&](auto x_, auto y_) { return x_ + y_; };

        if constexpr(wave_size > 1)
        {
            thread_data = reduce_op(
                thread_data,
                __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                            0x111,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl))); // row_shr:1
        }

        if constexpr(wave_size > 2)
        {
            thread_data = reduce_op(
                thread_data,
                __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                            0x112,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl))); // row_shr:2
        }
        if constexpr(wave_size > 4)
        {
            thread_data =
                reduce_op(thread_data,
                        __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                        0x114,
                                                                        row_mask,
                                                                        bank_mask,
                                                                        bound_ctrl))); // row_shr:4
        }
        if constexpr(wave_size == 8) {
            
            // wave-size=8 need one extra shift
            thread_data =
                reduce_op(thread_data,
                        __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                        0x118,
                                                                        row_mask,
                                                                        bank_mask,
                                                                        bound_ctrl))); // row_shr:8
#if 0
            constexpr int bank_mask_0_7 = 0b1100;
            auto reduce_op_r = [&](auto x_, auto y_) { return x_ - y_; };
            thread_data = reduce_op_r(thread_data, __builtin_bit_cast(data_t,
                                                    __builtin_amdgcn_update_dpp(0, /* old value */
                                                        __builtin_bit_cast(int, thread_data),
                                                        0x157,
                                                        row_mask,
                                                        bank_mask_0_7,
                                                        bound_ctrl))// row_newbcast:7
                                                        );
#else
            data_t xxx =__builtin_bit_cast(data_t, 
                            __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x157,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl)); // row_newbcast:7
            
            data_t yyy = (__lane_id() / 8) % 2 == 0 ? 0 : xxx;
            thread_data = thread_data - yyy;
#endif
            
        }
        if constexpr(wave_size > 8)
        {
            thread_data =
                reduce_op(thread_data,
                        __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                        0x118,
                                                                        row_mask,
                                                                        bank_mask,
                                                                        bound_ctrl))); // row_shr:8
        }

        if constexpr(wave_size > 16)
        {
            // now row-0, row-0+row-1, row-1+row-2, row-2+row-3
            int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 1) << 2, __builtin_bit_cast(int, thread_data));
            v_remote_tmp = __lane_id() >= 16 ? v_remote_tmp : 0;
            thread_data = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
        }

        if constexpr(wave_size > 32)
        {
            // lane-id 48...63->31
            int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 17) << 2, __builtin_bit_cast(int, thread_data));
            v_remote_tmp = __lane_id() >= 32 ? v_remote_tmp : 0;
            thread_data = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
        }
    }

    // reduce single pixel within a wave
    template <typename T, typename F, index_t wave_size_ = warpSize>
    __device__ static constexpr T wave_reduce(T local, F reduce_f, number<wave_size_> = {})
    {
        // constexpr int wave_size = 64;
        // constexpr int reduce_stage = 6; // 1<<6=64
        // clang-format off
        constexpr int reduce_stage = [](){
            if constexpr(wave_size_ == 2) return 1;
            else if constexpr(wave_size_ == 4) return 2;
            else if constexpr(wave_size_ == 8) return 3;
            else if constexpr(wave_size_ == 16) return 4;
            else if constexpr(wave_size_ == 32) return 5;
            else if constexpr(wave_size_ == 64) return 6;
            else return 0;
        }();
        // clang-format on
        T v_local = local;
#pragma unroll reduce_stage
        for(int i_stage = 0; i_stage < reduce_stage; i_stage++)
        {
            int src_lane = __lane_id() ^ (1 << i_stage);
            int32_t v_remote_tmp =
                __builtin_amdgcn_ds_bpermute(src_lane << 2, bit_cast<int32_t>(v_local));
            T v_remote = bit_cast<T>(v_remote_tmp);
            v_local    = reduce_f(v_local, v_remote);
        }
        return v_local;
    }

    CK_TILE_DEVICE index_t calc_index(index_t total_col, index_t row, index_t col) const
    {
        return row * total_col + col;
    }

    CK_TILE_DEVICE void moe_buf_set_zero_kernel(uint8x16_t* buf, index_t buf_bytes) const
    {
        const index_t offset = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
        if(offset < buf_bytes / 16)
        {
            buf[offset] = uint8x16_t{0};
        }
    }

    CK_TILE_DEVICE void moe_align_block_size_kernel(const IndexType* __restrict__ topk_id,
                                                    const WeightType* __restrict__ weights,
                                                    index_t* p_sorted_token_ids,
                                                    WeightType* p_sorted_weights,
                                                    index_t* p_sorted_expert_ids,
                                                    index_t* p_total_tokens_post_pad,
                                                    const index_t num_experts,
                                                    const index_t tokens_per_thread,
                                                    const index_t numel,
                                                    const mdiv unit_size_mdiv,
                                                    const mdiv topk_mdiv,
                                                    void* smem) const
    {
        const index_t tid       = static_cast<index_t>(threadIdx.x);
        const index_t start_idx = tid * tokens_per_thread;

        index_t* shared_mem = reinterpret_cast<index_t*>(smem);

        index_t* tokens_cnts = shared_mem; // 2d: (blockDim.x + 1, num_experts)
        index_t* cumsum = shared_mem + (blockDim.x + 1) * (num_experts + 1); // 1: (num_experts + 1)

        for(int i = 0; i < num_experts; ++i)
        {
            tokens_cnts[calc_index(num_experts + 1, tid + 1, i)] = 0;
        }

#pragma unroll Problem_::InternalLoadUnroll
        for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
        {
            ++tokens_cnts[calc_index(num_experts + 1, tid + 1, topk_id[i])];
        }
        __syncthreads();

#if 1
        if(tid < num_experts)
        {
            tokens_cnts[calc_index(num_experts + 1, 0, tid)] = 0;
            index_t local_c[8];
            index_t prev_c = 0;
            // TODO: manually unroll. pragma unroll does not work well when we have dependency
            for(int i = 1; i <= static_cast<index_t>(blockDim.x); i += 8)
            {
                local_c[0] = tokens_cnts[calc_index(num_experts + 1, i + 0, tid)];
                local_c[1] = tokens_cnts[calc_index(num_experts + 1, i + 1, tid)];
                local_c[2] = tokens_cnts[calc_index(num_experts + 1, i + 2, tid)];
                local_c[3] = tokens_cnts[calc_index(num_experts + 1, i + 3, tid)];
                local_c[4] = tokens_cnts[calc_index(num_experts + 1, i + 4, tid)];
                local_c[5] = tokens_cnts[calc_index(num_experts + 1, i + 5, tid)];
                local_c[6] = tokens_cnts[calc_index(num_experts + 1, i + 6, tid)];
                local_c[7] = tokens_cnts[calc_index(num_experts + 1, i + 7, tid)];

                local_c[0] += prev_c;
                local_c[1] += local_c[0];
                local_c[2] += local_c[1];
                local_c[3] += local_c[2];
                local_c[4] += local_c[3];
                local_c[5] += local_c[4];
                local_c[6] += local_c[5];
                local_c[7] += local_c[6];
                prev_c = local_c[7];

                tokens_cnts[calc_index(num_experts + 1, i + 0, tid)] = local_c[0];
                tokens_cnts[calc_index(num_experts + 1, i + 1, tid)] = local_c[1];
                tokens_cnts[calc_index(num_experts + 1, i + 2, tid)] = local_c[2];
                tokens_cnts[calc_index(num_experts + 1, i + 3, tid)] = local_c[3];
                tokens_cnts[calc_index(num_experts + 1, i + 4, tid)] = local_c[4];
                tokens_cnts[calc_index(num_experts + 1, i + 5, tid)] = local_c[5];
                tokens_cnts[calc_index(num_experts + 1, i + 6, tid)] = local_c[6];
                tokens_cnts[calc_index(num_experts + 1, i + 7, tid)] = local_c[7];
            }
        }
#else
        // TODO: below code still working, but slow in expert=32/topk=5 case. Put here for future
        // heuristic
        {
            if(tid < num_experts)
                tokens_cnts[calc_index(num_experts + 1, 0, tid)] = 0;
            for(int i = 0; i < num_experts; i += 8)
            {
                index_t local_c[8];
#pragma unroll
                for(int j = 0; j < 8; j++)
                {
                    local_c[j] = tokens_cnts[calc_index(num_experts + 1, tid + 1, i + j)];
                }

#pragma unroll
                for(int j = 0; j < 8; j++)
                {
                    wave_cumsum<int, 64>(local_c[j]);
                }

#pragma unroll
                for(int j = 0; j < 8; j++)
                {
                    tokens_cnts[calc_index(num_experts + 1, tid + 1, i + j)] = local_c[j];
                }
            }
        }
#endif

        __syncthreads();
        if constexpr(Problem::ExpertTile == 0)
        {
            if(tid == 0)
            {
                cumsum[0] = 0;
                for(int i = 1; i <= num_experts; ++i)
                {
                    auto current_units = [&]() {
                        index_t x_ = tokens_cnts[calc_index(num_experts + 1, blockDim.x, i - 1)] +
                                     unit_size_mdiv.divisor - 1;
                        index_t y_ = unit_size_mdiv.div(x_);
                        return max(y_, 1) * unit_size_mdiv.divisor;
                    }();
                    cumsum[i] = cumsum[i - 1] + current_units;
                }
                *p_total_tokens_post_pad = cumsum[num_experts];
            }
        }
        else
        {
            // TODO: we have out-of-bound read here. But result is still OK (will ignore tid >=
            // expert) for simplicity, not check experts here.
            int local_cnt          = tokens_cnts[calc_index(num_experts + 1, blockDim.x, tid)];
            int blocks_pers_expert = unit_size_mdiv.div(local_cnt + unit_size_mdiv.divisor - 1);
            int padded_tokens_per_expert = max(blocks_pers_expert, 1) * unit_size_mdiv.divisor;
            int local_cumsum             = padded_tokens_per_expert;
            wave_cumsum<int, 64>(local_cumsum);

            if(tid == (num_experts - 1))
            {
                cumsum[0]                = 0;
                *p_total_tokens_post_pad = local_cumsum;
            }
            if(tid < num_experts)
            {
                cumsum[tid + 1] = local_cumsum;
            }
        }

        __syncthreads();
        if(tid < num_experts)
        {
            int e_start = cumsum[tid];
            int e_end   = cumsum[tid + 1];
            for(int i = e_start; i < e_end; i += unit_size_mdiv.divisor)
            {
                p_sorted_expert_ids[unit_size_mdiv.div(i)] = tid;
            }
        }

#pragma unroll Problem_::InternalLoadUnroll
        for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
        {
            index_t expert_id     = topk_id[i];
            index_t local_cnt     = tokens_cnts[calc_index(num_experts + 1, tid, expert_id)];
            index_t rank_post_pad = local_cnt + cumsum[expert_id];
#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
            uint32_t curr_token_id, curr_topk_id;
            topk_mdiv.divmod(i, curr_token_id, curr_topk_id);
            p_sorted_token_ids[rank_post_pad] = MOE_SORTING_MOCK_ID(curr_token_id, curr_topk_id);
#else
            p_sorted_token_ids[rank_post_pad] = topk_mdiv.div(i);
#endif
            p_sorted_weights[rank_post_pad]                          = weights[i];
            tokens_cnts[calc_index(num_experts + 1, tid, expert_id)] = local_cnt + 1;
        }

        if constexpr(Problem::ExpertTile == 0)
        {
            const index_t prefill_token = topk_mdiv.div(numel);
            if(tid < num_experts)
            {
                index_t expert_offset =
                    cumsum[tid] + tokens_cnts[calc_index(num_experts + 1, blockDim.x, tid)];
                index_t expert_end = cumsum[tid + 1];
                while(expert_offset < expert_end)
                {
#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
                    p_sorted_token_ids[expert_offset] =
                        MOE_SORTING_MOCK_ID(prefill_token, topk_mdiv.divisor);
#else
                    p_sorted_token_ids[expert_offset] = prefill_token;
#endif
                    p_sorted_weights[expert_offset] = static_cast<WeightType>(0.0);
                    expert_offset++;
                }
            }
        }
        else
        {
            const index_t prefill_token = topk_mdiv.div(numel);
            // TODO: only support expert-tile like 8, 16, 32
            static constexpr index_t experts_per_wave = warpSize / Problem::ExpertTile;
            {
                index_t eid           = tid / experts_per_wave;
                index_t expert_offset = cumsum[eid] +
                                        tokens_cnts[calc_index(num_experts + 1, blockDim.x, eid)] +
                                        tid % experts_per_wave;
                index_t expert_end = cumsum[eid + 1];
                if(eid < num_experts)
                {
                    while(expert_offset < expert_end)
                    {
#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
                        p_sorted_token_ids[expert_offset] =
                            MOE_SORTING_MOCK_ID(prefill_token, topk_mdiv.divisor);
#else
                        p_sorted_token_ids[expert_offset] = prefill_token;
#endif
                        p_sorted_weights[expert_offset] = static_cast<WeightType>(0.0);
                        expert_offset += experts_per_wave;
                    }
                }
            }
        }
    }

    // only support index_t, and single pixel access
    struct simple_smem_indexer
    {
        index_t* smem;
        index_t row_stride;

        // this is 2D
        CK_TILE_DEVICE simple_smem_indexer(index_t* smem_, index_t row_stride_)
            : smem(smem_), row_stride(row_stride_)
        {
        }
        CK_TILE_DEVICE const index_t& operator()(index_t i_row, index_t i_col) const
        {
            return smem[i_row * row_stride + i_col];
        }
        CK_TILE_DEVICE index_t& operator()(index_t i_row, index_t i_col)
        {
            return smem[i_row * row_stride + i_col];
        }

        // this is 1D or linear
        CK_TILE_DEVICE simple_smem_indexer(index_t* smem_) : smem(smem_), row_stride(0) {}
        CK_TILE_DEVICE const index_t& operator()(index_t idx) const { return smem[idx]; }
        CK_TILE_DEVICE index_t& operator()(index_t idx) { return smem[idx]; }
    };

    CK_TILE_DEVICE void
    moe_align_block_size_kernel_ex(const IndexType* __restrict__ topk_id,
                                   const WeightType* __restrict__ weights,
                                   const IndexType* __restrict__ local_expert_mask,
                                   index_t* p_sorted_token_ids,
                                   WeightType* p_sorted_weights,
                                   index_t* p_sorted_expert_ids,
                                   index_t* p_total_tokens_post_pad,
                                   const index_t num_experts,
                                   const index_t tokens,
                                   const mdiv unit_size_mdiv,
                                   const mdiv topk_mdiv,
                                   const mdiv expert_mdiv,
                                   const index_t smem_rows,
                                   void* smem) const
    {
        const index_t tid            = static_cast<index_t>(threadIdx.x);
        const index_t wid            = __builtin_amdgcn_readfirstlane(tid / warpSize);
        const index_t lid            = __lane_id();
        constexpr index_t block_size = 256;           // blockDim.x;
        const index_t sub_tokens     = smem_rows - 2; // sub_tokens_mdiv.divisor;
        const index_t topk           = topk_mdiv.divisor;
        auto f_sum                   = [](auto x_, auto y_) { return x_ + y_; };

        const index_t smem_cols = num_experts + 1;

        simple_smem_indexer smem_cumsum{reinterpret_cast<index_t*>(smem) + 0};
        simple_smem_indexer smem_cumdup{reinterpret_cast<index_t*>(smem) + smem_cols};
        simple_smem_indexer smem_tokens{reinterpret_cast<index_t*>(smem) + 2 * smem_cols,
                                        smem_cols};

        // #pragma unroll 8
        for(int i = tid; i < (sub_tokens * num_experts); i += block_size)
        {
            uint32_t curr_token_id, curr_expert_id;
            expert_mdiv.divmod(i, curr_token_id, curr_expert_id);
            smem_tokens(curr_token_id, curr_expert_id) = 0;
        }
        __syncthreads();

        for(int i_token = 0; i_token < tokens; i_token += sub_tokens)
        {
            // NOTE: below for loop can't have barrier inside!!
            for(int i = tid; i < (sub_tokens * topk); i += block_size)
            {
                uint32_t curr_token_id, curr_topk_id;
                topk_mdiv.divmod(i, curr_token_id, curr_topk_id);
                int i_t = i_token + curr_token_id;

                if(i_t < tokens)
                {
                    int eid = topk_id[i_t * topk + curr_topk_id];

                    if constexpr(Problem::SubTokenOneShot)
                        smem_tokens(curr_token_id, eid) = curr_topk_id + 1;
                    else
                        smem_tokens(curr_token_id, eid)++;
                }
                __builtin_amdgcn_s_waitcnt(0xc07f);
            }
            __syncthreads(); // make sure different i_token iteration not overlap by different wave
        }

        // counting
        if(tid == 0)
        {
            smem_cumsum(0) = 0;
            // smem_cumdup(0) = 0;
        }

        {
            constexpr int lane_group_sz = 8;
            int lane_group_id           = tid / lane_group_sz;
            int lane_group_os           = tid % lane_group_sz;
            constexpr int lane_group_nm = block_size / lane_group_sz;

            for(int i_e = lane_group_id; i_e < num_experts; i_e += lane_group_nm)
            {
                index_t local_c[Problem::SubTokenTile];
                index_t cnt = 0;

                for(int i = 0; i < sub_tokens; i += 8 * Problem::SubTokenTile)
                {
#pragma unroll Problem::SubTokenTile
                    for(int j = 0; j < Problem::SubTokenTile; j++)
                    {
                        local_c[j] = smem_tokens(i + j * 8 + lane_group_os, i_e);
                        if constexpr(Problem::SubTokenOneShot)
                        {
                            local_c[j] = local_c[j] != 0 ? 1 : 0;
                        }
                    }

#pragma unroll Problem::SubTokenTile
                    for(int j = 0; j < Problem::SubTokenTile; j++)
                    {
                        cnt += wave_reduce(local_c[j], f_sum, number<8>{});
                    }
                }
                if(lane_group_os == 0)
                    smem_cumsum(i_e + 1) = cnt;
            }
        }

        if constexpr(Problem::LocalExpertMasking)
        {
            smem_cumdup(0) = 0;
            for(int i_e = tid; i_e < num_experts; i_e += block_size)
            {
                // reuse this buffer
                smem_cumdup(i_e + 1) = local_expert_mask[i_e];
            }
        }

        __syncthreads();

        {
            if(wid == 0)
            {
                // NOTE: under this block can never use __syncthreads!
                int i_e_          = 0;
                int local_cumsum_ = 0;
                for(; i_e_ < num_experts; i_e_ += warpSize)
                {
                    int pre_cumsum_ = smem_cumsum(lid == 0 ? i_e_ : 0);
                    int local_cnt   = smem_cumsum(i_e_ + lid + 1);
                    int blocks_pers_expert =
                        unit_size_mdiv.div(local_cnt + unit_size_mdiv.divisor - 1);

                    int pre_cumsum_masking = [&]() {
                        if constexpr(Problem::LocalExpertMasking)
                            return smem_cumdup(lid == 0 ? i_e_ : 0);
                        else
                            return 0; // not used
                    }();
                    int local_masking = [&]() {
                        if constexpr(Problem::LocalExpertMasking)
                            return smem_cumdup(i_e_ + lid + 1);
                        else
                            return 0; // not used
                    }();
                    int padded_tokens_per_expert = [&]() {
                        int x_ = [&]() {
                            if constexpr(Problem::SkipExpertsWithZeroTokens)
                            {
                                // if local_cnt is zero, blocks_pers_expert will be zero
                                // this is what we want to achieve
                                return blocks_pers_expert * unit_size_mdiv.divisor;
                            }
                            else
                            {
                                return max(blocks_pers_expert, 1) * unit_size_mdiv.divisor;
                            }
                        }();
                        if constexpr(Problem::LocalExpertMasking)
                        {
                            return local_masking ? x_ : 0;
                        }
                        else
                            return x_;
                    }();

                    local_cumsum_ = padded_tokens_per_expert;
                    local_cumsum_ += pre_cumsum_; // note pre_cumsum must be added after local
                                                  // cumsum padded in case local cumsum is zero, but
                                                  // pre_sumsum has value, which will result int
                                                  // zero local cumsum(but we want at least padded)
                    wave_cumsum<int, warpSize>(local_cumsum_);

                    if((i_e_ + lid) < num_experts)
                        smem_cumsum(i_e_ + lid + 1) = local_cumsum_;

                    if constexpr(Problem::LocalExpertMasking)
                    {
                        local_masking += pre_cumsum_masking;
                        wave_cumsum<int, warpSize>(local_masking);
                        if((i_e_ + lid) < num_experts)
                            smem_cumdup(i_e_ + lid + 1) = local_masking;
                    }

                    // NOTE: this waitcnt is a must, compiler will not generate waitcnt lgkmcnt()
                    // for above write however __syncthreads will cause barrier with waves other
                    // than 0(which is not we want)
                    __builtin_amdgcn_s_waitcnt(0xc07f);
                }
                if((lid + i_e_ - warpSize) == (num_experts - 1))
                {
                    *p_total_tokens_post_pad = local_cumsum_;
                }
            }
            __syncthreads();
        }

        for(int i_e = tid; i_e < num_experts; i_e += block_size)
        {
            int e_start = smem_cumsum(i_e);
            int e_end   = smem_cumsum(i_e + 1);

            int expert_id = [&]() {
                if constexpr(Problem::LocalExpertMasking)
                {
                    // local expert id from cumsum
                    return smem_cumdup(i_e);
                }
                else
                    return i_e;
            }();

            smem_cumdup(i_e) = e_start; // duplicate cumsum for later use
            if constexpr(Problem::SkipExpertsWithZeroTokens)
            {
                if(e_start == e_end) // skip zero token expert
                    continue;
            }

            if constexpr(Problem::LocalExpertMasking)
            {
                if(local_expert_mask[i_e] == 0)
                    continue;
            }

            for(int i = e_start; i < e_end; i += unit_size_mdiv.divisor)
            {
                p_sorted_expert_ids[unit_size_mdiv.div(i)] = expert_id;
            }
        }
        smem_cumdup(num_experts) = smem_cumsum(num_experts);

        // fill the p_sorted_token_ids/p_sorted_weights
        for(int i_token = 0; i_token < tokens; i_token += sub_tokens)
        {
            if constexpr(!Problem::SubTokenOneShot)
            {
                // clear every time
                for(int i = tid; i < (sub_tokens * num_experts); i += block_size)
                {
                    uint32_t curr_token_id, curr_expert_id;
                    expert_mdiv.divmod(i, curr_token_id, curr_expert_id);
                    smem_tokens(curr_token_id, curr_expert_id) = 0;
                }
                __syncthreads();

                // load again
                for(int i = tid; i < (sub_tokens * topk); i += block_size)
                {
                    uint32_t curr_token_id_, curr_topk_id_;
                    topk_mdiv.divmod(i, curr_token_id_, curr_topk_id_);
                    int curr_token_id = static_cast<int>(curr_token_id_);
                    int curr_topk_id  = static_cast<int>(curr_topk_id_);
                    int i_t           = i_token + curr_token_id;
                    if(i_t < tokens)
                    {
                        int eid                         = topk_id[i_t * topk + curr_topk_id];
                        smem_tokens(curr_token_id, eid) = curr_topk_id + 1; // at least 1
                    }
                }
                __syncthreads();
            }

            {
                constexpr int lane_group_sz = 8;
                int lane_group_id           = tid / lane_group_sz;
                int lane_group_os           = tid % lane_group_sz;
                constexpr int lane_group_nm = block_size / lane_group_sz;
                for(int eid = lane_group_id; eid < num_experts; eid += lane_group_nm)
                {
                    if constexpr(Problem::LocalExpertMasking)
                    {
                        if(local_expert_mask[eid] == 0)
                            continue;
                    }
                    int position = smem_cumsum(eid);
                    for(int i_sub_token = lane_group_os; i_sub_token < sub_tokens;
                        i_sub_token += lane_group_sz)
                    {
                        auto x = smem_tokens(i_sub_token, eid);

                        int local_cnt_cache = x != 0 ? 1 : 0;
                        int local_cnt       = local_cnt_cache;
                        wave_cumsum<int, lane_group_sz>(local_cnt);
                        if(x != 0)
                        {
                            // now x is topk value
#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
                            p_sorted_token_ids[position + local_cnt - 1] =
                                MOE_SORTING_MOCK_ID(i_token + i_sub_token, x - 1);
#else
                            p_sorted_token_ids[position + local_cnt - 1] = i_token + i_sub_token;
#endif
                            p_sorted_weights[position + local_cnt - 1] =
                                weights[(i_token + i_sub_token) * topk + x - 1];
                        }

                        int remote_cnt = __builtin_amdgcn_ds_bpermute(
                            (lane_group_sz * (lane_group_id + 1) - 1) << 2, local_cnt);

                        position += remote_cnt;
                    }
                    smem_cumsum(eid) = position;
                }
            }
            __syncthreads();
        }

        // add the skip number
        for(int eid = tid; eid < num_experts; eid += block_size)
        {
            int e_start = smem_cumsum(eid);
            int e_end   = smem_cumdup(eid + 1);
            if constexpr(Problem::SkipExpertsWithZeroTokens)
            {
                if(e_start == e_end) // skip zero token expert
                    continue;
            }
            while(e_start < e_end)
            {
#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
                p_sorted_token_ids[e_start] = MOE_SORTING_MOCK_ID(tokens, topk);
#else
                p_sorted_token_ids[e_start] = tokens;
#endif
                p_sorted_weights[e_start] = static_cast<WeightType>(0.0);
                e_start++;
            }
        }
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        if(blockIdx.x > 0)
        {
            if(kargs.p_moe_buf)
            {
                moe_buf_set_zero_kernel(reinterpret_cast<uint8x16_t*>(kargs.p_moe_buf),
                                        kargs.moe_buf_bytes);
            }
            return;
        }
        const size_t numel = kargs.tokens * kargs.topk_mdiv.divisor;
        extern __shared__ char smem[];
#if MOE_SORTING_USE_EX_KERNEL
        (void)numel;
        return moe_align_block_size_kernel_ex(
            static_cast<const IndexType*>(kargs.p_topk_ids),
            static_cast<const WeightType*>(kargs.p_weights),
            static_cast<const IndexType*>(kargs.p_local_expert_mask),
            static_cast<IndexType*>(kargs.p_sorted_token_ids),
            static_cast<WeightType*>(kargs.p_sorted_weights),
            static_cast<IndexType*>(kargs.p_sorted_expert_ids),
            static_cast<IndexType*>(kargs.p_total_tokens_post_pad),
            kargs.num_experts,
            kargs.tokens,
            kargs.unit_size_mdiv,
            kargs.topk_mdiv,
            kargs.expert_mdiv,
            kargs.smem_rows,
            smem);
#else
        return moe_align_block_size_kernel(static_cast<const IndexType*>(kargs.p_topk_ids),
                                           static_cast<const WeightType*>(kargs.p_weights),
                                           static_cast<IndexType*>(kargs.p_sorted_token_ids),
                                           static_cast<WeightType*>(kargs.p_sorted_weights),
                                           static_cast<IndexType*>(kargs.p_sorted_expert_ids),
                                           static_cast<IndexType*>(kargs.p_total_tokens_post_pad),
                                           kargs.num_experts,
                                           kargs.tokens_per_thread,
                                           numel,
                                           kargs.unit_size_mdiv,
                                           kargs.topk_mdiv,
                                           smem);
#endif
    }
};

#undef MOE_SORTING_MOCK_ID

} // namespace ck_tile
