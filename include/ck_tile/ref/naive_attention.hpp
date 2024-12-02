// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <thread>
#include <string>

namespace ck_tile {

enum class naive_attention_layout_enum
{
    BSHD,  // [batch, seqlen, nhead, hdim]
    BHSD,  // [batch, nhead, seqlen, hdim]
    BS3HD, // [batch, nhead, 3, seqlen, hdim], used when qkv are packed
    PHSD,  // [pages, nhead, page_size, hdim]
    PHSDX, // [pages, nhead, page_size/x, hdim, x], where pages*page_size = original seqlen
    PHDS,  // [pages, nhead, hdim, page_size], where pages*page_size = original seqlen
};

// will used to specialize kernel variation
enum class naive_attention_variation_enum
{
    FLASH_BATCHED = 0, // standard flash attention, or xformer/sdpa, used for training
    FLASH_GROUPED,
    DECODE_PAGED, // decode attn, where kv token from another buffer called kvcache
};

// TODO: for simplicity, this will be used as host/device arg
struct naive_attention_fwd_args
{
    void* q_ptr;
    void* k_ptr;
    void* v_ptr;
    void* o_ptr;
    void* context_len_ptr; // used when seqlen kv come from a pointer(each element is a number, not
                           // cumsum)
    void* page_table_ptr;  // [batch, num_blocks] seqlen_kv is in different block(paged attn)
    float scale_s;
    int hdim;
    int hdim_v; // could be cross-attn, where V and Q/K hdim are different
    int batch_q;
    int batch_kv;
    int batch_radio_kv; // batch_q / batch_kv
    int seqlen_q;
    int seqlen_kv; // if context_len_ptr is not nullptr, ignore this field
    int nhead_q;
    int nhead_kv;
    int nhead_radio_kv; // nhead_q / nhead_kv
    int page_size;      // if paged, the seqlen-kv per each block
};

// this is trait for host API
struct naive_attention_fwd_traits
{
    std::string q_type;
    std::string k_type;
    std::string v_type;
    std::string o_type;
    std::string q_layout;
    std::string k_layout;
    std::string v_layout;
    std::string o_layout;
    int variation; // sync with naive_attention_variation_enum
};

// this is trait for kernel template
template <naive_attention_variation_enum variation_>
struct naive_attention_fwd_kernel_traits
{
    static constexpr naive_attention_variation_enum variation = variation_;
};

// for simplicity, please do not use const-reference type for the template type
template <typename QType,
          typename KType,
          typename VType,
          typename OType,
          typename ComputeType,
          naive_attention_layout_enum QLayout,
          naive_attention_layout_enum KLayout,
          naive_attention_layout_enum VLayout,
          naive_attention_layout_enum OLayout,
          typename Traits>
struct naive_attention_fwd_kernel
{
    template <typename Acc>
    struct gemm2_vector_type;
    template <>
    struct gemm2_vector_type<float>
    {
        using type                = fp32x4_t;
        static constexpr int elem = 4;
    };
    using gemm2_vec_type                = typename gemm2_vector_type<ComputeType>::type;
    static constexpr int gemm2_vec_elem = gemm2_vector_type<ComputeType>::elem;

    __host__ __device__ naive_attention_fwd_kernel() {}

    template <typename T, naive_attention_layout_enum Layout>
    struct addresser
    {
        int b, s, h, d; // batch, seqlen, nhead, hdim
        T* base_ptr;
        __device__ addresser(int b_, int s_, int h_, int d_, void* base_ptr_)
            : b(b_), s(s_), h(h_), d(d_), base_ptr(reinterpret_cast<T*>(base_ptr_))
        {
        }

        // TODO: all the batch/nhead offset will accumulate to the base pointer
        __device__ T* get_base(int i_b, int i_h)
        {
            if constexpr(Layout == naive_attention_layout_enum::BSHD)
            {
                return base_ptr + i_b * s * h * d + i_h * d;
            }
            else if constexpr(Layout == naive_attention_layout_enum::BHSD)
            {
                return base_ptr + i_b * s * h * d + i_h * s * d;
            }
        }

        __device__ int get_offset(int i_s, int i_d)
        {
            if constexpr(Layout == naive_attention_layout_enum::BSHD)
            {
                return i_s * h * d + i_d;
            }
            else if constexpr(Layout == naive_attention_layout_enum::BHSD)
            {
                return i_s * d + i_d;
            }
        }

        // below set of API will directly use pointer inside this struct
        __device__ void update_base(int i_b, int i_h) { base_ptr = get_base(i_b, i_h); }
        __device__ T load(int i_s, int i_d) { return base_ptr[get_offset(i_s, i_d)]; }
        __device__ void store(T value, int i_s, int i_d) { base_ptr[get_offset(i_s, i_d)] = value; }
    };

    __device__ __host__ static constexpr int get_block_size() { return 256; }

    // for simpliciy, 1 WG always compute 1 token along q, compute all token along kv
    // compute all hdim from q, compute WG_SIZE hdim from v
    // 1) in prefill case, seqlen_q >= 1, seqlen_kv >= 1, batch_q=batch_kv
    // 2) in decode case, seqlen_q = 1, batch_q is input num-tokens, batch_kv is 1
    // 3) in paged-attn case, we still use 1 WG compute all the seqlen-kv for simplicity
    // TODO: could support split-kv to validate intermediate logsum
    __host__ static dim3 get_grid_size(naive_attention_fwd_args args)
    {
        constexpr int wg_size = get_block_size();
        return dim3(
            (args.hdim_v + wg_size - 1) / wg_size, args.seqlen_q, args.batch_q * args.nhead_q);
    }

    // reduce single pixel within a wave
    template <typename T, typename F>
    __device__ constexpr T wave_reduce(T local, F reduce_f)
    {
        // constexpr int wave_size = 64;
        constexpr int reduce_stage = 6; // 1<<6=64
        T v_local                  = local;
#pragma unroll
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

    // Note: this function must be called after wave_reduce
    template <typename T, typename F>
    __device__ constexpr T cross_wave_reduce(T local, F reduce_f, T* smem)
    {
        constexpr int waves     = 4;
        constexpr int wave_size = 64;
        int lane_id             = threadIdx.x % wave_size;

        __syncthreads();
        smem[threadIdx.x] = local;
        __syncthreads();

        // the data within single wave is the same
        // but for simplicity, we still use data from each lane.
        T v_local = smem[lane_id];
#pragma unroll
        for(int i_stage = 1; i_stage < waves; i_stage++)
        {
            T v_remote = smem[i_stage * wave_size + lane_id];
            v_local    = reduce_f(v_local, v_remote);
        }
        return v_local;
    }

    // kernel entry point
    __device__ void operator()(naive_attention_fwd_args args)
    {
        constexpr int wg_size = get_block_size();
        __shared__ ComputeType smem[wg_size * 2]; //  should enough
        addresser<QType, QLayout> q_addr{
            args.batch_q, args.seqlen_q, args.nhead_q, args.hdim, args.q_ptr};
        addresser<KType, KLayout> k_addr{
            args.batch_kv, args.seqlen_kv, args.nhead_kv, args.hdim, args.k_ptr};
        addresser<VType, VLayout> v_addr{
            args.batch_kv, args.seqlen_kv, args.nhead_kv, args.hdim_v, args.v_ptr};
        addresser<OType, OLayout> o_addr{
            args.batch_q, args.seqlen_q, args.nhead_q, args.hdim_v, args.o_ptr};

        int i_dv    = blockIdx.x * wg_size + threadIdx.x; // index of hdim_v
        int i_sq    = blockIdx.y;                         // index of seqlen_q
        int i_batch = blockIdx.z;                         // index of batch_q * nhead_q
        int i_bq    = i_batch / args.nhead_q;             // index of batch_q
        int i_hq    = i_batch % args.nhead_q;             // index of nhead_q

        int i_bk = i_bq / args.batch_radio_kv;
        int i_hk = i_hq / args.nhead_radio_kv;

        q_addr.update_base(i_bq, i_hq);
        k_addr.update_base(i_bk, i_hk);
        v_addr.update_base(i_bk, i_hk);
        o_addr.update_base(i_bq, i_hq);

        int seqlen_kv = args.seqlen_kv;
        auto f_max    = [](auto x_, auto y_) { return max(x_, y_); };
        auto f_sum    = [](auto x_, auto y_) { return x_ + y_; };

        ComputeType row_max = -numeric<ComputeType>::infinity();
        ComputeType o_acc   = {0};
        ComputeType l{0};

        int sk_loops = (seqlen_kv + wg_size - 1) / wg_size;

        for(int i_loop1 = 0; i_loop1 < sk_loops; i_loop1++)
        {
            int i_sk = i_loop1 * wg_size + threadIdx.x;
            // gemm-1
            ComputeType s_acc{0}; // clear for every loop
            if(i_sk < seqlen_kv)
            {
                for(auto i_dq = 0; i_dq < args.hdim; i_dq++)
                {
                    auto q = q_addr.load(i_sq, i_dq); // q will have duplicate load
                    auto k = k_addr.load(i_sk, i_dq);

                    s_acc += type_convert<ComputeType>(q) * type_convert<ComputeType>(k);
                }
                // scale
                s_acc *= type_convert<ComputeType>(args.scale_s * ck_tile::log2e_v<ComputeType>);
            }
            else
            {
                s_acc = -numeric<ComputeType>::infinity(); // out of bound need set to -INF
            }

            // s->p
            {
                // softmax, find max
                ComputeType old_max = row_max;
                ComputeType cur_max = wave_reduce(s_acc, f_max);

                cur_max = cross_wave_reduce(cur_max, f_max, smem);
                row_max = max(old_max, cur_max); // update row_max
                // softmax, exp(i_elem - max)
                ComputeType p = __builtin_amdgcn_exp2f(s_acc - row_max);

                // compute exp_sum
                ComputeType row_sum = wave_reduce(p, f_sum);
                row_sum             = cross_wave_reduce(row_sum, f_sum, smem);

                // l, pre-scall o_acc
                ComputeType tmp = __builtin_amdgcn_exp2f(old_max - row_max);
                l               = tmp * l + row_sum;
                o_acc *= tmp;

                // prepare the p into smem, to let every thread read same p and do 2nd gemm
                __syncthreads();
                smem[threadIdx.x] = p;
                __syncthreads();
            }

            // gemm-2, simple loop over vector by vector
            constexpr int gemm_2_loop = wg_size / gemm2_vec_elem;
            if(i_dv < args.hdim_v)
            {
                int sk_start = i_loop1 * wg_size; // we start from the first seqlen_kv element
                for(int i_loop2 = 0; i_loop2 < gemm_2_loop; i_loop2++)
                {
                    gemm2_vec_type p_vec = reinterpret_cast<gemm2_vec_type*>(smem)[i_loop2];

#pragma unroll
                    for(int i_j = 0; i_j < gemm2_vec_elem; i_j++)
                    {
                        int sv_offset = i_loop2 * gemm2_vec_elem + i_j;
                        auto v        = v_addr.load(sk_start + sv_offset, i_dv);
                        o_acc += p_vec[i_j] * type_convert<ComputeType>(v);
                    }
                }
            }
        }

        // post scale o_acc
        {
            ComputeType tmp = l == 0.f ? 0.f : 1.f / l; // in case masking
            o_acc *= tmp;
        }

        // store O
        if(i_dv < args.hdim_v)
            o_addr.store(type_convert<OType>(o_acc), i_sq, i_dv);
    }
};

#define CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_INTERNAL_()                                                        \
    {                                                                                                       \
        using ktraits_ =                                                                                    \
            naive_attention_fwd_kernel_traits<static_cast<naive_attention_variation_enum>(                  \
                variation_)>;                                                                               \
        using k_   = naive_attention_fwd_kernel<q_type_,                                                    \
                                              k_type_,                                                    \
                                              v_type_,                                                    \
                                              o_type_,                                                    \
                                              compute_type_,                                              \
                                              q_layout_,                                                  \
                                              k_layout_,                                                  \
                                              v_layout_,                                                  \
                                              o_layout_,                                                  \
                                              ktraits_>;                                                  \
        dim3 grids = k_::get_grid_size(a);                                                                  \
        r          = ck_tile::launch_kernel(s,                                                              \
                                   ck_tile::make_kernel(k_{}, grids, k_::get_block_size(), 0, a)); \
    }

#define CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_()                                                 \
    if(t.variation == 0 && t.q_layout == "bshd" && t.k_layout == "bshd" && t.v_layout == "bshd" && \
       t.o_layout == "bshd")                                                                       \
    {                                                                                              \
        constexpr auto q_layout_ = naive_attention_layout_enum::BSHD;                              \
        constexpr auto k_layout_ = naive_attention_layout_enum::BSHD;                              \
        constexpr auto v_layout_ = naive_attention_layout_enum::BSHD;                              \
        constexpr auto o_layout_ = naive_attention_layout_enum::BSHD;                              \
        constexpr int variation_ = 0;                                                              \
        CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_INTERNAL_();                                              \
    }                                                                                              \
    else if(t.variation == 0 && t.q_layout == "bhsd" && t.k_layout == "bhsd" &&                    \
            t.v_layout == "bhsd" && t.o_layout == "bhsd")                                          \
    {                                                                                              \
        constexpr auto q_layout_ = naive_attention_layout_enum::BHSD;                              \
        constexpr auto k_layout_ = naive_attention_layout_enum::BHSD;                              \
        constexpr auto v_layout_ = naive_attention_layout_enum::BHSD;                              \
        constexpr auto o_layout_ = naive_attention_layout_enum::BHSD;                              \
        constexpr int variation_ = 0;                                                              \
        CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_INTERNAL_();                                              \
    }

//
CK_TILE_HOST float naive_attention_fwd(naive_attention_fwd_traits t,
                                       naive_attention_fwd_args a,
                                       ck_tile::stream_config s)
{
    float r = -1;
    // TODO: do not explicitly create too much instance!
    if(t.q_type == "fp16" && t.k_type == "fp16" && t.v_type == "fp16" && t.o_type == "fp16")
    {
        using q_type_       = fp16_t;
        using k_type_       = fp16_t;
        using v_type_       = fp16_t;
        using o_type_       = fp16_t;
        using compute_type_ = float;
        CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_();
    }
    else if(t.q_type == "bf16" && t.k_type == "bf16" && t.v_type == "bf16" && t.o_type == "bf16")
    {
        using q_type_       = bf16_t;
        using k_type_       = bf16_t;
        using v_type_       = bf16_t;
        using o_type_       = bf16_t;
        using compute_type_ = float;
        CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_();
    }
    return r;
}

#undef CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_
#undef CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_INTERNAL_

} // namespace ck_tile
