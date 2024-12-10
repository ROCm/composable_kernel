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
    // PHSDX, // [pages, nhead, page_size/x, hdim, x], where <# used pages>*page_size = seqlen
    PHDSX, // [pages, nhead, hdim/x, page_size, x], where <# used pages>*page_size = seqlen
    PHDS,  // [pages, nhead, hdim, page_size], where <# used pages>*page_size = seqlen
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
    void* context_len_ptr; // [batch] used when seqlen kv come from a pointer(each element is a
                           // number, not cumsum)
    void* page_table_ptr;  // [batch, max_pages_per_seq] seqlen_kv is in different block(paged attn)
    void* kvscale_ptr;     // [nhead, 2(kv), hdim] used for kvcache dequant
    float scale_s;
    int hdim;
    int hdim_v; // could be cross-attn, where V and Q/K hdim are different
    int batch_q;
    int batch_kv;
    int batch_ratio_kv; // batch_q / batch_kv
    int seqlen_q;       // in decode case, this should be 1
    int seqlen_kv;      // if context_len_ptr is not nullptr, ignore this field
    int nhead_q;
    int nhead_kv;
    int nhead_ratio_kv; // nhead_q / nhead_kv
    int page_size;      // if paged, the seqlen-kv per each block
    int max_pages_per_seq;
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
          typename AccType,
          naive_attention_layout_enum QLayout,
          naive_attention_layout_enum KLayout,
          naive_attention_layout_enum VLayout,
          naive_attention_layout_enum OLayout,
          typename Traits>
struct naive_attention_fwd_kernel
{
    static constexpr bool is_kvcache_i8 =
        std::is_same_v<KType, int8_t> && std::is_same_v<VType, int8_t> && sizeof(QType) != 1;

    // kvcache-i8 will have per head scale, we apply this scale to Q/P matrix instead of original
    // K/V matrix. This can speed up conversion since Q/P usually is fp16/bf16/fp32
    static constexpr bool is_kvcache_i8_forward_quant = is_kvcache_i8;

    // TODO: hardcode
    using KVScaleType = float;
    using SoftmaxType = float;
    using PType       = VType; // src A of gemm2, same type as V

    using p_vec_type                = ext_vector_t<PType, 16 / sizeof(PType)>;
    static constexpr int p_vec_elem = vector_traits<p_vec_type>::vector_size;

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
                return base_ptr + i_b * s * h * d + i_h * d;
            else if constexpr(Layout == naive_attention_layout_enum::BHSD)
                return base_ptr + i_b * s * h * d + i_h * s * d;
        }

        __device__ int get_offset(int i_s, int i_d)
        {
            if constexpr(Layout == naive_attention_layout_enum::BSHD)
                return i_s * h * d + i_d;
            else if constexpr(Layout == naive_attention_layout_enum::BHSD)
                return i_s * d + i_d;
        }

        // below set of API will directly use pointer inside this struct
        __device__ void init(int i_b, int i_h) { base_ptr = get_base(i_b, i_h); }
        __device__ T load(int i_s, int i_d) { return base_ptr[get_offset(i_s, i_d)]; }
        __device__ void store(T value, int i_s, int i_d) { base_ptr[get_offset(i_s, i_d)] = value; }
    };

    template <typename T, naive_attention_layout_enum Layout>
    struct page_addresser
    {
        int s, h, d;                             // page_size, nhead, hdim
        static constexpr int x = 16 / sizeof(T); // pack 4 dword
        T* base_ptr;
        int* page_table_ptr; // TODO: page table always int
        int i_h;             // store current head

        __device__ page_addresser(int s_, int h_, int d_, void* base_ptr_, void* pptr_)
            : s(s_),
              h(h_),
              d(d_),
              base_ptr(reinterpret_cast<T*>(base_ptr_)),
              page_table_ptr(reinterpret_cast<int*>(pptr_))
        {
        }

        __device__ int64_t get_phy_page_idx(int i_s)
        {
            // dynamic compute page idx is simple but slow
            int page_idx = i_s / s;
            int phy      = page_table_ptr[page_idx];
            return static_cast<int64_t>(phy);
        }

        __device__ int get_phy_page_offset(int i_s)
        {
            // dynamic compute page idx is simple but slow
            return i_s % s;
        }

        __device__ int64_t get_offset(int i_s, int i_d)
        {
            int page_offset  = get_phy_page_offset(i_s);
            int64_t page_idx = get_phy_page_idx(i_s);
            int64_t base_    = page_idx * h * s * d;
            if constexpr(Layout == naive_attention_layout_enum::PHSD)
                return static_cast<int64_t>(i_h * s * d + page_offset * d + i_d) + base_;
            else if constexpr(Layout == naive_attention_layout_enum::PHDSX)
            {
                int d_r = i_d / x;
                int d_x = i_d % x;
                return static_cast<int64_t>(i_h * d * s + d_r * s * x + page_offset * x + d_x) +
                       base_;
            }
            else if constexpr(Layout == naive_attention_layout_enum::PHDS)
            {
                return static_cast<int64_t>(i_h * d * s + i_d * s + page_offset) + base_;
            }
        }

        // below set of API will directly use pointer inside this struct
        __device__ void init(int /*i_b*/, int i_h_) { i_h = i_h_; }
        __device__ T load(int i_s, int i_d) { return base_ptr[get_offset(i_s, i_d)]; }
        __device__ void store(T /*value*/, int /*i_s*/, int /*i_d*/) {}
    };

    template <typename T>
    struct kvscale_addresser
    {
        int h, d; // nhead, hdim
        T* base_ptr;
        __device__ kvscale_addresser(int h_, int d_, void* p_)
            : h(h_), d(d_), base_ptr(reinterpret_cast<T*>(p_))
        {
        }
        __device__ int get_offset(int i_h, int i_d, int i_kv /*0 or 1*/)
        {
            // [h, 2, d]
            return i_h * 2 * d + i_kv * d + i_d;
        }
        __device__ T load(int i_h, int i_d, int i_kv)
        {
            return base_ptr[get_offset(i_h, i_d, i_kv)];
        }
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
        auto g =
            dim3((args.hdim_v + wg_size - 1) / wg_size, args.seqlen_q, args.batch_q * args.nhead_q);
        return g;
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
    // Note: better not use this under if...else... with thread divergence (syncthreads)
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
        __shared__ char smem[wg_size * 4 * sizeof(float)]; //  should enough
        int i_dv    = blockIdx.x * wg_size + threadIdx.x;  // index of hdim_v
        int i_sq    = blockIdx.y;                          // index of seqlen_q
        int i_batch = blockIdx.z;                          // index of batch_q * nhead_q
        int i_bq    = i_batch / args.nhead_q;              // index of batch_q
        int i_hq    = i_batch % args.nhead_q;              // index of nhead_q

        int i_bk = i_bq / args.batch_ratio_kv;
        int i_hk = i_hq / args.nhead_ratio_kv;

        void* page_table_ptr = [&]() {
            if constexpr(Traits::variation == naive_attention_variation_enum::DECODE_PAGED)
            {
                return reinterpret_cast<int*>(args.page_table_ptr) + i_bq * args.max_pages_per_seq;
            }
            else
            {
                return nullptr;
            }
        }();

        auto q_addr = [&]() {
            if constexpr(Traits::variation == naive_attention_variation_enum::FLASH_BATCHED)
            {
                return addresser<QType, QLayout>{
                    args.batch_q, args.seqlen_q, args.nhead_q, args.hdim, args.q_ptr};
            }
            else if constexpr(Traits::variation == naive_attention_variation_enum::DECODE_PAGED)
            {
                return addresser<QType, QLayout>{
                    args.batch_q, args.seqlen_q, args.nhead_q, args.hdim, args.q_ptr};
            }
        }();
        auto k_addr = [&]() {
            if constexpr(Traits::variation == naive_attention_variation_enum::FLASH_BATCHED)
            {
                return addresser<KType, KLayout>{
                    args.batch_kv, args.seqlen_kv, args.nhead_kv, args.hdim, args.k_ptr};
            }
            else if constexpr(Traits::variation == naive_attention_variation_enum::DECODE_PAGED)
            {
                return page_addresser<KType, KLayout>{
                    args.page_size, args.nhead_kv, args.hdim, args.k_ptr, page_table_ptr};
            }
        }();
        auto v_addr = [&]() {
            if constexpr(Traits::variation == naive_attention_variation_enum::FLASH_BATCHED)
            {
                return addresser<VType, VLayout>{
                    args.batch_kv, args.seqlen_kv, args.nhead_kv, args.hdim_v, args.v_ptr};
            }
            else if constexpr(Traits::variation == naive_attention_variation_enum::DECODE_PAGED)
            {
                return page_addresser<VType, VLayout>{
                    args.page_size, args.nhead_kv, args.hdim_v, args.v_ptr, page_table_ptr};
            }
        }();
        auto o_addr = [&]() {
            if constexpr(Traits::variation == naive_attention_variation_enum::FLASH_BATCHED)
            {
                return addresser<OType, OLayout>{
                    args.batch_q, args.seqlen_q, args.nhead_q, args.hdim_v, args.o_ptr};
            }
            else if constexpr(Traits::variation == naive_attention_variation_enum::DECODE_PAGED)
            {
                return addresser<OType, OLayout>{
                    args.batch_q, args.seqlen_q, args.nhead_q, args.hdim_v, args.o_ptr};
            }
        }();

        q_addr.init(i_bq, i_hq);
        k_addr.init(i_bk, i_hk);
        v_addr.init(i_bk, i_hk);
        o_addr.init(i_bq, i_hq);

        auto f_max        = [](auto x_, auto y_) { return max(x_, y_); };
        auto f_sum        = [](auto x_, auto y_) { return x_ + y_; };
        auto f_absmax_f32 = [](float v_0_, float v_1_) {
            float rtn;
            asm volatile("v_max_f32 %0, abs(%1), abs(%2)" : "=v"(rtn) : "v"(v_0_), "v"(v_1_));
            return rtn;
        };

        int seqlen_kv = [&]() {
            if constexpr(Traits::variation == naive_attention_variation_enum::FLASH_BATCHED)
            {
                return args.seqlen_kv;
            }
            else if constexpr(Traits::variation == naive_attention_variation_enum::DECODE_PAGED)
            {
                return reinterpret_cast<int*>(args.context_len_ptr)[i_bq];
            }
        }();

        SoftmaxType row_max = -numeric<SoftmaxType>::infinity();
        SoftmaxType l{0};
        AccType o_acc = {0};

        int sk_loops   = (seqlen_kv + wg_size - 1) / wg_size;
        float qf_scale = .0f;
        kvscale_addresser<KVScaleType> kvscale_addr{args.nhead_kv, args.hdim, args.kvscale_ptr};

        if constexpr(is_kvcache_i8_forward_quant)
        {
            // AccType is i32 now, seqlen_q = 1, hdim up to 256
            float q   = 0;
            float k_s = 0;
            if(static_cast<int>(threadIdx.x) < args.hdim)
            {
                q   = type_convert<float>(q_addr.load(0, threadIdx.x));
                k_s = type_convert<float>(kvscale_addr.load(i_hk, threadIdx.x, 0));
            }
            // 1) we apply the k scale to q
            float q_forwarded = q * k_s;

            // 2) apply smooth-quant
            // find absmax
            float qf_max = wave_reduce(q_forwarded, f_absmax_f32);
            qf_max       = cross_wave_reduce(qf_max, f_absmax_f32, reinterpret_cast<float*>(smem));

            // per-token scale
            qf_scale = qf_max / 127.0;

            // devide by scale
            q = q / qf_scale;

            // fp32->i8
            int8_t quantized_q = static_cast<int8_t>(q);
            __syncthreads();
            reinterpret_cast<int8_t*>(smem)[threadIdx.x] = quantized_q;
            __syncthreads();

            // after above process, we have 2 data
            // 1) int8 q data stored in smem(no need to reload)
            // 2) per-token scale qf_scale, to be mul after 1st gemm
        }

        for(int i_loop1 = 0; i_loop1 < sk_loops; i_loop1++)
        {
            int i_sk = i_loop1 * wg_size + threadIdx.x;
            // gemm-1
            SoftmaxType s_softmax = -numeric<SoftmaxType>::infinity();
            if(i_sk < seqlen_kv)
            {
                AccType s_acc{0}; // clear for every loop
                for(auto i_dq = 0; i_dq < args.hdim; i_dq++)
                {
                    if constexpr(is_kvcache_i8_forward_quant)
                    {
                        int8_t q = reinterpret_cast<int8_t*>(smem)[i_dq];
                        auto k   = k_addr.load(i_sk, i_dq);

                        s_acc += type_convert<AccType>(q) * type_convert<AccType>(k);
                    }
                    else
                    {
                        auto q = q_addr.load(i_sq, i_dq); // q will have duplicate load
                        auto k = k_addr.load(i_sk, i_dq);

                        s_acc += type_convert<AccType>(q) * type_convert<AccType>(k);
                    }
                }
                // scale
                s_softmax = type_convert<SoftmaxType>(s_acc);
                s_softmax *=
                    type_convert<SoftmaxType>(args.scale_s * ck_tile::log2e_v<SoftmaxType>);
                if constexpr(is_kvcache_i8_forward_quant)
                {
                    s_softmax *= qf_scale; // post scale the per-token factor
                }
            }

            // s->p
            float pf_scale = 0.; // used for i8 quant
            {
                // softmax, find max
                SoftmaxType old_max = row_max;
                SoftmaxType cur_max = wave_reduce(s_softmax, f_max);

                cur_max = cross_wave_reduce(cur_max, f_max, reinterpret_cast<SoftmaxType*>(smem));
                row_max = max(old_max, cur_max); // update row_max
                // softmax, exp(i_elem - max)
                SoftmaxType p_compute = __builtin_amdgcn_exp2f(s_softmax - row_max);

                // compute exp_sum
                SoftmaxType row_sum = wave_reduce(p_compute, f_sum);
                row_sum = cross_wave_reduce(row_sum, f_sum, reinterpret_cast<SoftmaxType*>(smem));

                // l, pre-scall o_acc
                SoftmaxType tmp = __builtin_amdgcn_exp2f(old_max - row_max);
                l               = tmp * l + row_sum;
                o_acc           = type_convert<AccType>(type_convert<SoftmaxType>(o_acc) * tmp);

                // prepare the p_compute into smem, to let every thread read same p_compute and do
                // 2nd gemm
                if constexpr(is_kvcache_i8_forward_quant)
                {
                    float v_s = 0;
                    if(static_cast<int>(threadIdx.x) < args.hdim_v)
                    {
                        v_s = type_convert<float>(kvscale_addr.load(i_hk, threadIdx.x, 1));
                    }

                    // 1) we apply the v scale to p
                    float p_forwarded = p_compute * v_s;

                    // 2) apply smooth-quant
                    // find absmax
                    float pf_max = wave_reduce(p_forwarded, f_absmax_f32);
                    pf_max =
                        cross_wave_reduce(pf_max, f_absmax_f32, reinterpret_cast<float*>(smem));

                    // per-token scale
                    pf_scale = pf_max / 127.0;

                    // devide by scale
                    p_compute = p_compute / pf_scale;

                    // fp32->i8
                    int8_t quantized_p = static_cast<int8_t>(p_compute);
                    __syncthreads();
                    reinterpret_cast<int8_t*>(smem)[threadIdx.x] = quantized_p;
                    __syncthreads();
                    // after above process, we have 2 data
                    // 1) int8 p data stored in smem(no need to reload)
                    // 2) per-token scale pf_scale, to be mul after 2nd gemm
                }
                else
                {
                    __syncthreads();
                    reinterpret_cast<PType*>(smem)[threadIdx.x] = type_convert<PType>(p_compute);
                    __syncthreads();
                }
            }

            // gemm-2, simple loop over vector by vector
            constexpr int gemm_2_loop = wg_size / p_vec_elem;
            {
                AccType o_acc_local = {0};
                int sk_start = i_loop1 * wg_size; // we start from the first seqlen_kv element
                for(int i_loop2 = 0; i_loop2 < gemm_2_loop; i_loop2++)
                {
                    p_vec_type p_vec = reinterpret_cast<p_vec_type*>(smem)[i_loop2];
#pragma unroll
                    for(int i_j = 0; i_j < p_vec_elem; i_j++)
                    {
                        int sv_offset = i_loop2 * p_vec_elem + i_j;
                        int i_sv      = sk_start + sv_offset;

                        VType v = 0.f;
                        if(i_dv < args.hdim_v && i_sv < seqlen_kv)
                        {
                            v = v_addr.load(i_sv, i_dv);
                        }

                        o_acc_local += type_convert<AccType>(p_vec[i_j]) * type_convert<AccType>(v);
                    }
                }
                if constexpr(is_kvcache_i8_forward_quant)
                {
                    // apply pr scale to local acc
                    o_acc_local =
                        type_convert<AccType>(type_convert<float>(o_acc_local) * pf_scale);
                }
                o_acc += o_acc_local;
            }
        }

        // post scale o_acc
        {
            SoftmaxType tmp = l == 0.f ? 0.f : 1.f / l; // in case masking
            o_acc           = type_convert<AccType>(type_convert<SoftmaxType>(o_acc) * tmp);
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
                                              acc_type_,                                                  \
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
    }                                                                                              \
    else if(t.variation == 2 && t.q_layout == "bhsd" && t.k_layout == "phdsx" &&                   \
            t.v_layout == "phds" && t.o_layout == "bhsd")                                          \
    {                                                                                              \
        constexpr auto q_layout_ = naive_attention_layout_enum::BHSD;                              \
        constexpr auto k_layout_ = naive_attention_layout_enum::PHDSX;                             \
        constexpr auto v_layout_ = naive_attention_layout_enum::PHDS;                              \
        constexpr auto o_layout_ = naive_attention_layout_enum::BHSD;                              \
        constexpr int variation_ = 2;                                                              \
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
        using q_type_   = fp16_t;
        using k_type_   = fp16_t;
        using v_type_   = fp16_t;
        using o_type_   = fp16_t;
        using acc_type_ = float;
        CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_();
    }
    else if(t.q_type == "bf16" && t.k_type == "bf16" && t.v_type == "bf16" && t.o_type == "bf16")
    {
        using q_type_   = bf16_t;
        using k_type_   = bf16_t;
        using v_type_   = bf16_t;
        using o_type_   = bf16_t;
        using acc_type_ = float;
        CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_();
    }
    else if(t.q_type == "bf16" && t.k_type == "int8" && t.v_type == "int8" && t.o_type == "bf16")
    {
        using q_type_   = bf16_t;
        using k_type_   = int8_t;
        using v_type_   = int8_t;
        using o_type_   = bf16_t;
        using acc_type_ = int32_t; // NOTE!
        CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_();
    }
    else if(t.q_type == "fp16" && t.k_type == "int8" && t.v_type == "int8" && t.o_type == "fp16")
    {
        using q_type_   = fp16_t;
        using k_type_   = int8_t;
        using v_type_   = int8_t;
        using o_type_   = fp16_t;
        using acc_type_ = int32_t; // NOTE!
        CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_();
    }
    return r;
}

#undef CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_LAOYUT_
#undef CK_TILE_DISPATCH_NAIVE_ATTEN_FWD_INTERNAL_

} // namespace ck_tile
