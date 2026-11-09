// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "mha_fwd_common.hpp"

#include <optional>
#include <string>

namespace ck_tile::fmha::torch_api {

std::vector<at::Tensor>
mha_fwd(at::Tensor& q,       // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor>&
            out_, // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor>& alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_dropout_randval,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> bias_,
        std::optional<at::Tensor> q_descale_,
        std::optional<at::Tensor> k_descale_,
        std::optional<at::Tensor> v_descale_,
        std::string qscale,
        bool return_lse,
        int block_size_q,
        int block_size_kv,
        bool lse_log2)
{
    TORCH_CHECK(softcap == 0.0f, "CK Tile FMHA Python bindings do not support softcap");
    auto q_dtype      = q.dtype();
    const bool is_fp8 = q.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                        q.scalar_type() == at::ScalarType::Float8_e4m3fnuz;
    TORCH_CHECK_NOT_IMPLEMENTED(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16 || is_fp8,
                                "CK Tile FMHA only supports fp16, bf16 and fp8 e4m3 data types");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    std::string q_dtype_str = is_fp8 ? "fp8bf16" : q_dtype == torch::kFloat16 ? "fp16" : "bf16";

    CK_TILE_FMHA_CHECK_DEVICE(q);
    CK_TILE_FMHA_CHECK_DEVICE(k);
    CK_TILE_FMHA_CHECK_DEVICE(v);
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                "query, key and value must be on the same device");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "query, key and value must be four-dimensional");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size  = sizes[0];
    int seqlen_q          = sizes[1];
    int num_heads         = sizes[2];
    const int head_size   = sizes[3];
    const int seqlen_k    = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    detail::check_alibi(alibi_slopes_, q, batch_size, num_heads);
    TORCH_CHECK_NOT_IMPLEMENTED(head_size > 0 && head_size <= 256,
                                "CK only supports head dimension in [1, 256]");
    TORCH_CHECK_NOT_IMPLEMENTED(
        head_size % 8 == 0,
        "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads > 0 && num_heads_k > 0 && num_heads % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    const bool perhead           = qscale == "perhead";
    const bool blockscale        = qscale == "blockscale";
    quant_scale_enum qscale_type = quant_scale_enum::no_scale;
    TORCH_CHECK(qscale == "no_scale" || qscale == "pertensor" || perhead || blockscale,
                "Unsupported FP8 qscale: ",
                qscale);
    if(is_fp8)
    {
        qscale_type = perhead                 ? quant_scale_enum::perhead
                      : blockscale            ? quant_scale_enum::blockscale
                      : qscale == "pertensor" ? quant_scale_enum::pertensor
                                              : quant_scale_enum::no_scale;
        if(blockscale)
            TORCH_CHECK(block_size_q > 0 && block_size_kv > 0, "block sizes must be positive");
        TORCH_CHECK_NOT_IMPLEMENTED(p_dropout == 0.0f && !return_dropout_randval,
                                    "FP8 attention does not support dropout");
        TORCH_CHECK_NOT_IMPLEMENTED(!bias_.has_value() && !alibi_slopes_.has_value(),
                                    "FP8 attention does not support bias or ALiBi");
        TORCH_CHECK_NOT_IMPLEMENTED(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
                                    "FP8 attention supports forward only");
        const std::string arch = at::cuda::getDeviceProperties(q.get_device())->gcnArchName;
        const bool use_fnuz    = arch.rfind("gfx94", 0) == 0;
        TORCH_CHECK(q.scalar_type() == (use_fnuz ? at::ScalarType::Float8_e4m3fnuz
                                                 : at::ScalarType::Float8_e4m3fn),
                    "FP8 dtype does not match the GPU: expected ",
                    use_fnuz ? "float8_e4m3fnuz" : "float8_e4m3fn");
        TORCH_CHECK_NOT_IMPLEMENTED(
            !perhead || (arch.rfind("gfx1250", 0) == 0 && head_size <= 128),
            "FP8 per-head attention requires gfx1250 and head dimension at most 128");
        const auto check_descale = [&](const std::optional<at::Tensor>& descale,
                                       int heads,
                                       int seqlen,
                                       int block_size,
                                       const char* name) {
            if(qscale_type == quant_scale_enum::no_scale)
            {
                TORCH_CHECK(!descale.has_value(), "no_scale does not accept descales");
                return;
            }
            TORCH_CHECK(descale.has_value(), name, " is required for FP8 attention");
            const auto& tensor = descale.value();
            TORCH_CHECK(
                tensor.device() == q.device(), name, " must be on the same device as query");
            TORCH_CHECK(
                tensor.scalar_type() == at::ScalarType::Float, name, " must have dtype float32");
            TORCH_CHECK_NOT_IMPLEMENTED(!tensor.requires_grad(),
                                        "FP8 attention supports forward only");
            if(blockscale)
            {
                TORCH_CHECK(
                    tensor.sizes() ==
                        torch::IntArrayRef(
                            {batch_size, heads, ck_tile::integer_divide_ceil(seqlen, block_size)}),
                    name,
                    " must have shape [batch_size, nheads, ceil(seqlen / block_size)]");
                TORCH_CHECK(tensor.stride(-1) == 1, name, " must have contiguous block dimension");
            }
            else if(perhead)
            {
                TORCH_CHECK(tensor.sizes() == torch::IntArrayRef({batch_size, heads}),
                            name,
                            " must have shape [batch_size, nheads]");
            }
            else
            {
                TORCH_CHECK(tensor.numel() == 1, name, " must contain one scale");
            }
        };
        check_descale(q_descale_, num_heads, seqlen_q, block_size_q, "q_descale");
        check_descale(k_descale_, num_heads_k, seqlen_k, block_size_kv, "k_descale");
        check_descale(v_descale_, num_heads_k, seqlen_k, block_size_kv, "v_descale");
    }
    else
    {
        TORCH_CHECK(!q_descale_.has_value() && !k_descale_.has_value() && !v_descale_.has_value() &&
                        qscale == "no_scale",
                    "descale is only supported for FP8 inputs");
    }

    detail::check_bias(bias_, alibi_slopes_, q, seqlen_q, seqlen_k);
    const auto bias_type = bias_.has_value()           ? bias_enum::elementwise_bias
                           : alibi_slopes_.has_value() ? bias_enum::alibi
                                                       : bias_enum::no_bias;

    // causal=true is the same as causal=false in this case
    if(seqlen_q == 1 && !alibi_slopes_.has_value())
    {
        is_causal = false;
    }
    const mask_info mask =
        detail::make_mask_info(is_causal, window_size_left, window_size_right, seqlen_q, seqlen_k);

    const bool seqlenq_ngroups_swapped =
        detail::use_ngroups_swap(
            seqlen_q, num_heads, num_heads_k, mask, bias_type != bias_enum::no_bias) &&
        p_dropout == 0.f && !perhead && !blockscale;
    if(seqlenq_ngroups_swapped)
    {
        q         = detail::to_ngroups_layout(q, num_heads_k);
        seqlen_q  = num_heads / num_heads_k;
        num_heads = num_heads_k;
    }

    CK_TILE_FMHA_CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CK_TILE_FMHA_CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CK_TILE_FMHA_CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    at::Tensor out;
    const auto out_dtype = is_fp8 ? at::ScalarType::BFloat16 : q.scalar_type();
    if(out_.has_value())
    {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_dtype, "Output has an incorrect dtype");
        CK_TILE_FMHA_CHECK_DEVICE(out);
        TORCH_CHECK(out.device() == q.device(), "Output must be on the same device as query");
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CK_TILE_FMHA_CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
        if(seqlenq_ngroups_swapped)
        {
            out = detail::to_ngroups_layout(out, num_heads_k);
        }
    }
    else
    {
        out = torch::empty_like(q, q.options().dtype(out_dtype));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto stream = detail::current_stream();

    auto opts        = q.options();
    bool has_lse     = return_lse;
    bool has_dropout = p_dropout > 0.0f;

    at::Tensor softmax_lse;
    // TODO - check gradient, only training require lse
    softmax_lse = has_lse
                      ? torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32))
                      : torch::empty({0}, opts.dtype(torch::kFloat32));

    at::Tensor p;
    if(return_dropout_randval)
    {
        TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
        p = torch::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(torch::kUInt8));
    }
    else
    {
        p = torch::empty({0}, opts);
    }

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    auto rng_state         = torch::empty({2}, opts.dtype(torch::kInt64));
    auto rng_state_ptr     = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if(p_dropout > 0.0)
    {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        detail::init_philox_state(gen, counter_offset, stream, rng_state_ptr);
    }

    if(seqlen_k > 0 && seqlen_q > 0)
    {
        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
        ck_tile::stream_config stream_config{stream};

        auto traits =
            get_ck_fmha_fwd_traits(mask, q_dtype_str, head_size, has_dropout, has_lse, bias_type);

        auto args = get_ck_fmha_fwd_args(has_lse,
                                         return_dropout_randval,
                                         mask,
                                         batch_size,
                                         seqlen_q,
                                         seqlen_k,
                                         num_heads,
                                         num_heads_k,
                                         head_size,
                                         q,
                                         k,
                                         v,
                                         alibi_slopes_,
                                         bias_,
                                         out,
                                         softmax_lse,
                                         p,
                                         softmax_scale,
                                         p_dropout,
                                         drop_seed_offset);

        traits.qscale_type = qscale_type;
        args.lse_log2      = lse_log2 && has_lse; // the LSE in base 2
        if(is_fp8 && qscale_type != quant_scale_enum::no_scale)
        {
            args.q_descale_ptr = q_descale_.value().data_ptr();
            args.k_descale_ptr = k_descale_.value().data_ptr();
            args.v_descale_ptr = v_descale_.value().data_ptr();
            if(perhead || blockscale)
            {
                args.block_scale_size_q     = block_size_q;
                args.block_scale_size_kv    = block_size_kv;
                args.nhead_stride_q_descale = q_descale_.value().stride(1);
                args.nhead_stride_k_descale = k_descale_.value().stride(1);
                args.nhead_stride_v_descale = v_descale_.value().stride(1);
                args.batch_stride_q_descale = q_descale_.value().stride(0);
                args.batch_stride_k_descale = k_descale_.value().stride(0);
                args.batch_stride_v_descale = v_descale_.value().stride(0);
            }
        }

        const float t = detail::dispatch_lse_log2(args, softmax_lse, [&] {
            float r = detail::maybe_dispatch_head_grouped_fwd(
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
                q.scalar_type(),
                [&](const auto& grouped_traits, auto& grouped_args, const auto& grouped_sc) {
                    return fmha_fwd(grouped_traits, grouped_args, grouped_sc);
                });
            return r < 0.0f ? fmha_fwd(traits, args, stream_config) : r;
        });
        TORCH_CHECK_NOT_IMPLEMENTED(t >= 0,
                                    "No FMHA forward kernel supports these arguments: dtype=",
                                    q_dtype_str,
                                    ", head_size=",
                                    head_size,
                                    ", qscale=",
                                    qscale,
                                    ", return_lse=",
                                    return_lse);
    }
    else
    {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if(seqlenq_ngroups_swapped)
    {
        out = detail::from_ngroups_layout(out);
        q   = detail::from_ngroups_layout(q);
        if(has_lse)
            softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse, p, rng_state};
}

} // namespace ck_tile::fmha::torch_api
