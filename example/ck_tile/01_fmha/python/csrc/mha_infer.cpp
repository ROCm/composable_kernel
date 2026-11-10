// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Inference attention: fp16/bf16 with masks and bias, without the training-only state (dropout,
// its random values and RNG). mha_fwd serves training too and pays for it on every call; when
// the kernel runs in a few microseconds that host work dominates, so this entry does only what
// inference needs. It shares mha_fwd's setup (fmha_common.hpp, mha_fwd_common.hpp).

#include "mha_fwd_common.hpp"

#include <limits>
#include <optional>
#include <tuple>

namespace ck_tile::fmha::torch_api {

std::tuple<at::Tensor, at::Tensor>
mha_infer(const at::Tensor& q_, // b x s_q [x h] x d
          const at::Tensor& k_, // b x s_k [x h_k] x d
          const at::Tensor& v_, // b x s_k [x h_k] x d
          const float softmax_scale,
          bool is_causal,
          int window_size_left,
          int window_size_right,
          const std::optional<at::Tensor>& alibi_slopes_, // h or b x h
          const std::optional<at::Tensor>& bias_,         // s_q x s_k, shared by batches and heads
          const bool return_lse,
          const bool lse_log2,
          std::optional<at::Tensor> out_)
{
    // The kernels need a contiguous last dimension (head dim); other strides are free.
    const auto last_dim_contiguous = [](const at::Tensor& t) {
        return t.stride(-1) == 1 ? t : t.contiguous();
    };
    at::Tensor q       = last_dim_contiguous(q_);
    const at::Tensor k = last_dim_contiguous(k_);
    const at::Tensor v = k_.is_same(v_) ? k : last_dim_contiguous(v_);

    const auto dtype = q.scalar_type();
    TORCH_CHECK_NOT_IMPLEMENTED(dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16,
                                "infer supports fp16 and bf16");
    TORCH_CHECK(k.scalar_type() == dtype && v.scalar_type() == dtype,
                "query, key and value must have the same dtype");
    TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device(),
                "query, key and value must be on the same CUDA device");

    // [b, s, d] is one head; [b, s, h, d] is the flash-attention layout.
    const int64_t rank = q.dim();
    TORCH_CHECK((rank == 3 || rank == 4) && k.dim() == rank && v.dim() == rank,
                "query, key and value must all be [b, s, d] or all [b, s, h, d]");
    const bool has_heads = rank == 4;

    const int batch_size  = q.size(0);
    int seqlen_q          = q.size(1);
    const int seqlen_k    = k.size(1);
    int num_heads         = has_heads ? q.size(2) : 1;
    const int num_heads_k = has_heads ? k.size(2) : 1;
    const int head_size   = q.size(-1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(num_heads > 0 && num_heads_k > 0 && num_heads % num_heads_k == 0,
                "number of key/value heads must divide number of query heads");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size && v.size(1) == seqlen_k &&
                    (!has_heads || v.size(2) == num_heads_k) && k.size(-1) == head_size &&
                    v.size(-1) == head_size,
                "key and value shapes do not match query");
    TORCH_CHECK_NOT_IMPLEMENTED(head_size > 0 && head_size <= 256 && head_size % 8 == 0,
                                "head dimension must be a multiple of 8 in [8, 256]");

    detail::check_bias(bias_, alibi_slopes_, q, seqlen_q, seqlen_k);
    if(alibi_slopes_.has_value())
    {
        const auto& slopes = *alibi_slopes_;
        TORCH_CHECK(slopes.scalar_type() == at::kFloat && slopes.device() == q.device(),
                    "ALiBi slopes must be float32 on the query's device");
        TORCH_CHECK(slopes.sizes() == at::IntArrayRef({num_heads}) ||
                        slopes.sizes() == at::IntArrayRef({batch_size, num_heads}),
                    "ALiBi slopes must be [nheads] or [batch_size, nheads]");
    }
    const bias_enum bias_type = bias_.has_value()           ? bias_enum::elementwise_bias
                                : alibi_slopes_.has_value() ? bias_enum::alibi
                                                            : bias_enum::no_bias;

    // causal=true is the same as causal=false for one query
    if(seqlen_q == 1 && !alibi_slopes_.has_value())
        is_causal = false;
    const mask_info mask =
        detail::make_mask_info(is_causal, window_size_left, window_size_right, seqlen_q, seqlen_k);

    at::Tensor out;
    if(out_.has_value())
    {
        out = std::move(out_.value());
        TORCH_CHECK(out.scalar_type() == dtype && out.device() == q.device() &&
                        out.sizes() == q.sizes() && out.stride(-1) == 1,
                    "out must match query in dtype, device and shape, with contiguous last "
                    "dimension");
    }
    else
    {
        out = at::empty_like(q, at::MemoryFormat::Contiguous);
    }

    // decode with grouped KV heads: one block of h / h_k rows per KV head (fmha_common.hpp); the
    // kernel writes out through a view of it
    const bool seqlenq_ngroups_swapped =
        has_heads && detail::use_ngroups_swap(
                         seqlen_q, num_heads, num_heads_k, mask, bias_type != bias_enum::no_bias);
    at::Tensor out_run = out;
    if(seqlenq_ngroups_swapped)
    {
        q         = detail::to_ngroups_layout(q, num_heads_k);
        out_run   = detail::to_ngroups_layout(out, num_heads_k);
        seqlen_q  = num_heads / num_heads_k;
        num_heads = num_heads_k;
    }

    // LSE [b, h, s_q] as the kernel writes it ([b, s_q] for the head-less layout)
    at::Tensor lse;
    if(return_lse)
    {
        const auto opts = q.options().dtype(at::kFloat);
        lse             = has_heads ? at::empty({batch_size, num_heads, seqlen_q}, opts)
                                    : at::empty({batch_size, seqlen_q}, opts);
    }
    const auto result = [&] {
        // swapped back, [b, h_k, g] is [b, h, 1]
        return std::make_tuple(out,
                               seqlenq_ngroups_swapped && return_lse
                                   ? lse.reshape({batch_size, num_heads * seqlen_q, 1})
                                   : lse);
    };

    if(seqlen_q == 0)
        return result();
    if(seqlen_k == 0)
    {
        out.zero_();
        if(return_lse)
            lse.fill_(std::numeric_limits<float>::infinity());
        return result();
    }

    at::cuda::CUDAGuard device_guard{q.device()};
    const ck_tile::stream_config stream_config{detail::current_stream()};

    const fmha_fwd_traits traits =
        get_ck_fmha_fwd_traits(mask,
                               dtype == at::ScalarType::Half ? "fp16" : "bf16",
                               head_size,
                               false,
                               return_lse,
                               bias_type);
    // the head-less layout as its one-head view [b, s, 1, d] ([b, 1, s_q] for the LSE): the
    // strides fmha_func passes for it, so both pick the same kernels (q.stride(2) of [b, s, d] is
    // the head dim's 1, which the d128 qr_tdm_sched O-layout check refuses as a head stride)
    const auto heads_view = [&](const at::Tensor& t) { return has_heads ? t : t.unsqueeze(2); };
    fmha_fwd_args args    = get_ck_fmha_fwd_args(return_lse,
                                              false, // no dropout random values
                                              mask,
                                              batch_size,
                                              seqlen_q,
                                              seqlen_k,
                                              num_heads,
                                              num_heads_k,
                                              head_size,
                                              heads_view(q),
                                              heads_view(k),
                                              heads_view(v),
                                              alibi_slopes_,
                                              bias_,
                                              heads_view(out_run),
                                              has_heads || !return_lse ? lse : lse.unsqueeze(1),
                                              at::Tensor(), // dropout random values
                                              softmax_scale,
                                              0.0f, // no dropout
                                                 {nullptr, nullptr});
    args.lse_log2         = lse_log2 && return_lse; // the LSE in base 2

    const float t = detail::dispatch_lse_log2(args, lse, [&] {
        float r = -1.0f;
        if(num_heads > 1) // head grouping only splits heads
        {
            r = detail::maybe_dispatch_head_grouped_fwd(
                stream_config,
                traits,
                args,
                num_heads,
                num_heads_k,
                batch_size,
                seqlen_k,
                head_size,
                head_size,
                k.element_size(),
                v.element_size(),
                dtype,
                [&](const auto& grouped_traits, auto& grouped_args, const auto& grouped_sc) {
                    return fmha_fwd(grouped_traits, grouped_args, grouped_sc);
                });
        }
        return r < 0.0f ? fmha_fwd(traits, args, stream_config) : r;
    });
    TORCH_CHECK_NOT_IMPLEMENTED(t >= 0,
                                "No FMHA kernel supports these inference arguments: dtype=",
                                traits.data_type,
                                ", head_size=",
                                head_size,
                                ", mask=",
                                static_cast<int>(mask.type),
                                ", bias=",
                                static_cast<int>(bias_type),
                                ", return_lse=",
                                return_lse);
    return result();
}

} // namespace ck_tile::fmha::torch_api
