// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"
#include "ck_tile/ref/naive_attention.hpp"
#include "sageattn_fwd.hpp"
#include "utils.hpp"
#include "ck_tile/utility/json_dump.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <cmath>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

template <typename... Args>
inline void dump_sageattn_fwd_json_results(Args&&... args)
{
    dump_fmha_fwd_json_results(std::forward<Args>(args)...);
}

enum class fwd_result
{
    success,
    failure,
    invalid_args,
    no_instance,
};

// different threshold for different dtype
template <typename DataTypeConfig>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<SageAttentionFwdBf16>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<SageAttentionFwdFp8Bf16>(std::string /*init_method*/)
{
    // atol=0.18: Q, K, V quantization (FP8 E4M3 ~0.0625/element) + 2 GEMM accumulations
    // + softmax sensitivity. Empirically tuned; tightening below 0.15 causes false positives.
    double rtol = 1e-2;
    double atol = 1.8e-1;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<SageAttentionFwdI8Fp8Bf16>(std::string /*init_method*/)
{
    // atol=0.18: K, V still FP8 (dominant error source). Matches FP8xFP8 despite
    // lower Q quantization error (int8 ~0.0078 vs fp8 ~0.0625) to avoid test fragility.
    double rtol = 1e-2;
    double atol = 1.8e-1;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<SageAttentionFwdI4Fp8Bf16>(std::string /*init_method*/)
{
    // atol=0.19: +0.01 over FP8 due to coarse Q quantization (int4 ~0.125, only 16 levels).
    // Attention pattern becomes "blocky"; softmax amplifies logit clustering.
    double rtol = 1e-2;
    double atol = 1.9e-1;
    return ck_tile::make_tuple(rtol, atol);
}

template <typename DataTypeConfig>
fwd_result sageattn_fwd_run(mode_enum mode,
                            ck_tile::index_t batch,
                            ck_tile::index_t nhead,
                            ck_tile::index_t nhead_k,
                            std::vector<ck_tile::index_t> seqlen_qs,
                            std::vector<ck_tile::index_t> seqlen_ks,
                            ck_tile::index_t hdim_q,
                            ck_tile::index_t hdim_v,
                            std::vector<ck_tile::index_t> seqlen_qpads,
                            std::vector<ck_tile::index_t> seqlen_kpads,
                            std::vector<ck_tile::index_t> q_eff_lens_per_batch,
                            std::vector<ck_tile::index_t> kv_eff_lens_per_batch,
                            bool i_perm,
                            bool o_perm,
                            float scale_s,
                            bool is_v_rowmajor,
                            std::string mask_str,
                            std::string qscale_str,
                            std::string init_method,
                            uint32_t seed,
                            int do_validation,
                            const ck_tile::stream_config& stream_config,
                            std::optional<std::string> json = std::nullopt)
{
    const std::string data_type = []() {
        if constexpr(std::is_same_v<DataTypeConfig, SageAttentionFwdFp16>)
            return "fp16";
        else if constexpr(std::is_same_v<DataTypeConfig, SageAttentionFwdBf16>)
            return "bf16";
        else if constexpr(std::is_same_v<DataTypeConfig, SageAttentionFwdFp8Bf16>)
            return "fp8bf16";
        else if constexpr(std::is_same_v<DataTypeConfig, SageAttentionFwdI8Fp8Bf16>)
            return "i8fp8bf16";
        else if constexpr(std::is_same_v<DataTypeConfig, SageAttentionFwdI4Fp8Bf16>)
            return "i4fp8bf16";
        else
            static_assert(false);
    }();

    if(nhead_k < 0)
        nhead_k = nhead;
    if(nhead % nhead_k != 0)
    {
        std::cerr << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return fwd_result::invalid_args;
    }

    std::mt19937 random_engine(seed != 0 ? seed : std::random_device{}());
    auto next_seed = [&random_engine]() { return static_cast<unsigned int>(random_engine()); };

    if(hdim_v < 0)
        hdim_v = hdim_q;

    // Check padding usage
    const bool has_group_q_padding =
        mode == mode_enum::group && (!seqlen_qpads.empty() && seqlen_qpads[0] > 0);
    const bool has_group_k_padding =
        mode == mode_enum::group && (!seqlen_kpads.empty() && seqlen_kpads[0] > 0);
    const bool has_group_padding   = has_group_q_padding || has_group_k_padding;
    const bool has_batch_q_padding = mode == mode_enum::batch && !q_eff_lens_per_batch.empty();
    const bool has_batch_k_padding = mode == mode_enum::batch && !kv_eff_lens_per_batch.empty();
    const bool has_batch_padding   = has_batch_q_padding || has_batch_k_padding;

    std::tie(seqlen_qs, seqlen_ks, seqlen_qpads, seqlen_kpads) =
        generate_missing_seqlens(mode,
                                 batch,
                                 seqlen_qs,
                                 seqlen_ks,
                                 seqlen_qpads,
                                 seqlen_kpads,
                                 /*seqlen_k_min=*/0,
                                 false, // need_append_kvcache not supported
                                 random_engine);
    for(ck_tile::index_t wb = 0; wb < batch; ++wb)
    {
        if(seqlen_kpads[wb] > 0 && seqlen_kpads[wb] < seqlen_ks[wb])
        {
            std::cerr << "kpad must be greater than or equal to seqlen for k" << std::endl;
            return fwd_result::invalid_args;
        }
        if(seqlen_qpads[wb] > 0 && seqlen_qpads[wb] < seqlen_qs[wb])
        {
            std::cerr << "qpad must be greater than or equal to seqlen for q" << std::endl;
            return fwd_result::invalid_args;
        }
    }

    if(scale_s == .0f)
        scale_s = 1.0f / ck_tile::sqrt(static_cast<float>(hdim_q)); // TODO: q ? v ?

    mask_info mask =
        mask_info::decode(mask_str, seqlen_qs[0], seqlen_ks[0]); // TODO: we don't need x/y anymore

    quant_scale_info qscale = quant_scale_info::decode(qscale_str);

    // PERWARP mode: Q=32 (warp size), K=64 (2x warp size)
    // BLOCKSCALE mode: Q=128 (tile size), K=128
    // PERTHREAD mode: Q=4 (tokens/scale), K=16 (tokens/scale)
    // Note: V uses per-channel scale, not block scale
    const ck_tile::index_t block_scale_size_q_ = (qscale.type == quant_scale_enum::perwarp) ? 32
                                                 : (qscale.type == quant_scale_enum::perthread)
                                                     ? 4
                                                     : 128;
    const ck_tile::index_t block_scale_size_k_ = (qscale.type == quant_scale_enum::perthread) ? 16
                                                 : (qscale.type == quant_scale_enum::perwarp) ? 64
                                                                                              : 128;

    // blockscale, perwarp, or perthread
    const bool qscale_uses_bwp = qscale.type == quant_scale_enum::blockscale ||
                                 qscale.type == quant_scale_enum::perwarp ||
                                 qscale.type == quant_scale_enum::perthread;

    const auto seqstart_q_host              = to_seqstarts(seqlen_qs);
    const auto seqstart_k_host              = to_seqstarts(seqlen_ks);
    const auto seqstart_q_with_padding_host = to_seqstarts(seqlen_qpads);
    const auto seqstart_k_with_padding_host = to_seqstarts(seqlen_kpads);

    // Optional batch-mode cumulative seqlen overrides
    std::vector<ck_tile::index_t> cuq_cum, cukv_cum;
    if(mode == mode_enum::batch)
    {
        auto calculate_cumulative = [&](std::vector<ck_tile::index_t>& per_batch_vec,
                                        std::vector<ck_tile::index_t>& cum_vec) {
            if(!per_batch_vec.empty() && per_batch_vec[0] != -1)
            {
                if(per_batch_vec.size() < static_cast<size_t>(batch))
                {
                    per_batch_vec.resize(batch, per_batch_vec.back());
                }
                cum_vec.resize(batch + 1);
                cum_vec[0] = 0;
                for(int i = 0; i < batch; ++i)
                    cum_vec[i + 1] = cum_vec[i] + per_batch_vec[i];
            }
        };

        calculate_cumulative(q_eff_lens_per_batch, cuq_cum);
        calculate_cumulative(kv_eff_lens_per_batch, cukv_cum);
    }

    using TypeConfig = SageAttentionFwdTypeConfig<DataTypeConfig>;

    using QDataType           = typename TypeConfig::QDataType;
    using KDataType           = typename TypeConfig::KDataType;
    using VDataType           = typename TypeConfig::VDataType;
    using SaccDataType        = typename TypeConfig::SaccDataType;
    using SMPLComputeDataType = typename TypeConfig::SMPLComputeDataType;
    using PDataType           = typename TypeConfig::PDataType;
    using OaccDataType        = typename TypeConfig::OaccDataType;
    using ODataType           = typename TypeConfig::ODataType;

    constexpr ck_tile::index_t q_packed_size =
        ck_tile::is_packed_type_v<QDataType> ? ck_tile::numeric_traits<QDataType>::PackedSize : 1;
    constexpr ck_tile::index_t k_packed_size =
        ck_tile::is_packed_type_v<KDataType> ? ck_tile::numeric_traits<KDataType>::PackedSize : 1;
    constexpr bool is_q_i4                  = std::is_same_v<QDataType, ck_tile::pk_int4_t>;
    constexpr bool is_k_i4                  = std::is_same_v<KDataType, ck_tile::pk_int4_t>;
    constexpr bool need_q_i4_permute        = is_q_i4 && !is_k_i4;
    constexpr bool need_k_i4_permute        = is_k_i4 && !is_q_i4;
    const ck_tile::index_t hdim_q_storage_q = hdim_q / q_packed_size;
    const ck_tile::index_t hdim_q_storage_k = hdim_q / k_packed_size;
    if constexpr(ck_tile::is_packed_type_v<QDataType>)
    {
        if(hdim_q % q_packed_size != 0)
        {
            std::cerr << "hdim_q must be divisible by packed size for QDataType, got hdim_q="
                      << hdim_q << ", packed_size=" << q_packed_size << std::endl;
            return fwd_result::invalid_args;
        }
        if constexpr(need_q_i4_permute)
        {
            if(hdim_q % 8 != 0)
            {
                std::cerr << "hdim_q must be divisible by 8 for pk_int4_t QDataType, got hdim_q="
                          << hdim_q << std::endl;
                return fwd_result::invalid_args;
            }
        }
    }
    if constexpr(ck_tile::is_packed_type_v<KDataType>)
    {
        if(hdim_q % k_packed_size != 0)
        {
            std::cerr << "hdim_q must be divisible by packed size for KDataType, got hdim_q="
                      << hdim_q << ", packed_size=" << k_packed_size << std::endl;
            return fwd_result::invalid_args;
        }
        if constexpr(need_k_i4_permute)
        {
            if(hdim_q % 8 != 0)
            {
                std::cerr << "hdim_q must be divisible by 8 for pk_int4_t KDataType, got hdim_q="
                          << hdim_q << std::endl;
                return fwd_result::invalid_args;
            }
        }
    }

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<int32_t>::min(); // we will use max seqlen to decide grid size
    auto max_seqlen_k = std::numeric_limits<int32_t>::min();
    {
        for(ck_tile::index_t wb = 0; wb < batch; ++wb)
        {
            const int32_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            const int32_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

            if(max_seqlen_q < real_seqlen_q)
            {
                max_seqlen_q = real_seqlen_q;
            }

            if(max_seqlen_k < real_seqlen_k)
            {
                max_seqlen_k = real_seqlen_k;
            }

            flop += nhead * (static_cast<std::size_t>(2) * mask.get_unmaskarea() * hdim_q +
                             static_cast<std::size_t>(2) * mask.get_unmaskarea() * hdim_v);

            num_byte += nhead * (sizeof(QDataType) * real_seqlen_q * hdim_q_storage_q +
                                 sizeof(ODataType) * real_seqlen_q * hdim_v);
            num_byte += nhead_k * (sizeof(KDataType) * real_seqlen_k * hdim_q_storage_k +
                                   sizeof(VDataType) * hdim_v * real_seqlen_k);
        }
    }

    static const auto get_lengths = [](bool permute,
                                       ck_tile::index_t b /*batch*/,
                                       ck_tile::index_t h /*nhead*/,
                                       ck_tile::index_t s /*seqlen*/,
                                       ck_tile::index_t d /*hdim*/) {
        if(permute)
            return std::array<ck_tile::index_t, 4>{b, h, s, d};
        else
            return std::array<ck_tile::index_t, 4>{b, s, h, d};
    };

    // host memory for storing all the tensor elements
    const ck_tile::index_t shape_batch = (mode == mode_enum::batch ? batch : 1);
    // physical(padded) total seqlen_q for group when s_qpad is provided; else use logical
    const ck_tile::index_t shape_seqlen_q =
        (mode == mode_enum::batch ? seqlen_qs[0]
                                  : (has_group_q_padding && !seqstart_q_with_padding_host.empty()
                                         ? seqstart_q_with_padding_host.back()
                                         : seqstart_q_host.back()));
    const ck_tile::index_t shape_seqlen_k =
        (mode == mode_enum::batch ? seqlen_ks[0]
                                  : (has_group_k_padding && !seqstart_k_with_padding_host.empty()
                                         ? seqstart_k_with_padding_host.back()
                                         : seqstart_k_host.back()));

    // Calculate number of blocks for blockscale mode
    ck_tile::index_t i_block_scale_q = 0;
    ck_tile::index_t i_block_scale_k = 0;
    std::vector<int32_t> block_scale_seqstart_q_host{0};
    std::vector<int32_t> block_scale_seqstart_k_host{0};

    if(mode == mode_enum::group)
    {
        for(ck_tile::index_t wb = 0; wb < batch; ++wb)
        {
            const int32_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            const int32_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];
            i_block_scale_q += ck_tile::integer_divide_ceil(real_seqlen_q, block_scale_size_q_);
            i_block_scale_k += ck_tile::integer_divide_ceil(real_seqlen_k, block_scale_size_k_);
            block_scale_seqstart_q_host.push_back(i_block_scale_q);
            block_scale_seqstart_k_host.push_back(i_block_scale_k);
        }
    }

    const ck_tile::index_t num_block_scale_q =
        (mode == mode_enum::batch)
            ? ck_tile::integer_divide_ceil(shape_seqlen_q, block_scale_size_q_)
            : i_block_scale_q;
    const ck_tile::index_t num_block_scale_k =
        (mode == mode_enum::batch)
            ? ck_tile::integer_divide_ceil(shape_seqlen_k, block_scale_size_k_)
            : i_block_scale_k;

    ck_tile::HostTensor<QDataType> q_host(
        get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    ck_tile::HostTensor<KDataType> k_host(
        get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    ck_tile::HostTensor<VDataType> v_host(
        is_v_rowmajor ? get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v)
                      : get_lengths(i_perm, shape_batch, nhead_k, hdim_v, shape_seqlen_k));

    ck_tile::HostTensor<float> q_descale_host(
        qscale_uses_bwp ? std::array<ck_tile::index_t, 3>{shape_batch, nhead, num_block_scale_q}
                        : std::array<ck_tile::index_t, 3>{1, 1, 1});
    ck_tile::HostTensor<float> k_descale_host(
        qscale_uses_bwp ? std::array<ck_tile::index_t, 3>{shape_batch, nhead_k, num_block_scale_k}
                        : std::array<ck_tile::index_t, 3>{1, 1, 1});
    // BLOCKSCALE, PERWARP, and PERTHREAD V all use per-channel scale (col-major layout)
    ck_tile::HostTensor<float> v_descale_host(
        qscale_uses_bwp ? std::array<ck_tile::index_t, 3>{batch, nhead_k, hdim_v}
                        : std::array<ck_tile::index_t, 3>{1, 1, 1});

    ck_tile::HostTensor<ODataType> o_host(
        get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));

    const auto get_dtype_max = []<typename T>() {
        if constexpr(ck_tile::is_packed_type_v<T>)
            return 7.0f;
        else
            return ck_tile::type_convert<float>(ck_tile::numeric<T>::max());
    };

    if(init_method == "ui" || init_method == "0")
    {
        ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f, next_seed()}(q_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, next_seed()}(k_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, next_seed()}(v_host);
    }

    else if(init_method == "ni")
    {
        ck_tile::FillNormalDistributionIntegerValue<QDataType>{-3.f, 3.f, next_seed()}(q_host);
        ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, next_seed()}(k_host);
        ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, next_seed()}(v_host);
    }
    else if(init_method == "uf" || init_method == "1")
    {
        ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, next_seed()}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, next_seed()}(k_host);
        ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, next_seed()}(v_host);
    }
    else if(init_method == "nf")
    {
        ck_tile::FillNormalDistribution<QDataType>{0.f, 3.f, next_seed()}(q_host);
        ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, next_seed()}(k_host);
        ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, next_seed()}(v_host);
    }
    else if(init_method == "tf" || init_method == "2")
    {
        ck_tile::FillTrigValue<QDataType>{}(q_host);
        ck_tile::FillTrigValue<KDataType>{}(k_host);
        ck_tile::FillTrigValue<VDataType>{}(v_host);
    }
    else if(init_method == "3")
    {
        float q_dtype_max = get_dtype_max.template operator()<QDataType>();
        float k_dtype_max = get_dtype_max.template operator()<KDataType>();
        float v_dtype_max = get_dtype_max.template operator()<VDataType>();

        ck_tile::FillUniformDistribution<QDataType>{-q_dtype_max, q_dtype_max, next_seed()}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{-k_dtype_max, k_dtype_max, next_seed()}(k_host);
        ck_tile::FillUniformDistribution<VDataType>{-v_dtype_max, v_dtype_max, next_seed()}(v_host);
    }
    if(qscale.type == quant_scale_enum::pertensor)
    {
        float q_dtype_max = get_dtype_max.template operator()<QDataType>();
        float k_dtype_max = get_dtype_max.template operator()<KDataType>();
        float v_dtype_max = get_dtype_max.template operator()<VDataType>();

        float qkv_max     = 3.f;
        q_descale_host(0) = qkv_max / q_dtype_max;
        k_descale_host(0) = qkv_max / k_dtype_max;
        v_descale_host(0) = qkv_max / v_dtype_max;
    }
    else if(qscale_uses_bwp)
    {
        float q_dtype_max = get_dtype_max.template operator()<QDataType>();
        float k_dtype_max = get_dtype_max.template operator()<KDataType>();
        float v_dtype_max = get_dtype_max.template operator()<VDataType>();

        float qkv_max       = 3.f;
        float max_descale_q = qkv_max / q_dtype_max;
        float max_descale_k = qkv_max / k_dtype_max;
        float max_descale_v = qkv_max / v_dtype_max;

        ck_tile::FillUniformDistribution<float>{max_descale_q * 0.8f, max_descale_q, next_seed()}(
            q_descale_host);
        ck_tile::FillUniformDistribution<float>{max_descale_k * 0.8f, max_descale_k, next_seed()}(
            k_descale_host);

        // BLOCKSCALE, PERWARP, and PERTHREAD V all use per-channel scale (shape: [batch, nhead_k,
        // hdim_v])
        ck_tile::FillUniformDistribution<float>{max_descale_v * 0.8f, max_descale_v, next_seed()}(
            v_descale_host);
    }

    ck_tile::DeviceMem q_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem q_descale_buf(q_descale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_descale_buf(k_descale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_descale_buf(v_descale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqstart_q_padded_buf(seqstart_q_with_padding_host.empty()
                                                 ? 0
                                                 : seqstart_q_with_padding_host.size() *
                                                       sizeof(int32_t));
    ck_tile::DeviceMem seqstart_k_padded_buf(
        seqlen_kpads[0] < 0 ? 0 : seqstart_k_with_padding_host.size() * sizeof(int32_t));
    // Buffers for query per-sequence logical (unpadded) lengths (used in group mode with padding
    // enabled)
    ck_tile::DeviceMem seqlen_q_buf(has_group_q_padding ? seqlen_qs.size() * sizeof(int32_t) : 0);
    // Buffers for key/value per-sequence logical (unpadded) lengths (used in group mode with
    // padding enabled)
    ck_tile::DeviceMem seqlen_k_buf(has_group_k_padding ? seqlen_ks.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem cu_seqlen_q_buf(cuq_cum.empty() ? 0
                                                       : cuq_cum.size() * sizeof(ck_tile::index_t));
    ck_tile::DeviceMem cu_seqlen_kv_buf(
        cukv_cum.empty() ? 0 : cukv_cum.size() * sizeof(ck_tile::index_t));
    // Must match args.block_scale_seqstart_* (group + bs/pw/pth only). bf16 validation (qscale=n)
    // never binds these pointers; allocating only when the kernel uses them avoids empty uploads.
    const bool need_block_scale_seqstart_buf = mode == mode_enum::group && qscale_uses_bwp;
    ck_tile::DeviceMem block_scale_seqstart_q_buf(
        need_block_scale_seqstart_buf ? block_scale_seqstart_q_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem block_scale_seqstart_k_buf(
        need_block_scale_seqstart_buf ? block_scale_seqstart_k_host.size() * sizeof(int32_t) : 0);

    if constexpr(need_q_i4_permute)
    {
        auto q_host_dev = q_host;
        ck_tile::permute_vectors_i4x4_b(q_host_dev);
        q_buf.ToDevice(q_host_dev.data());
    }
    else
    {
        q_buf.ToDevice(q_host.data());
    }
    if constexpr(need_k_i4_permute)
    {
        auto k_host_dev = k_host;
        ck_tile::permute_vectors_i4x4_b(k_host_dev);
        k_buf.ToDevice(k_host_dev.data());
    }
    else
    {
        k_buf.ToDevice(k_host.data());
    }
    v_buf.ToDevice(v_host.data());
    q_descale_buf.ToDevice(q_descale_host.data());
    k_descale_buf.ToDevice(k_descale_host.data());
    v_descale_buf.ToDevice(v_descale_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    // Keep logical starts in seqstart_k; pass padded K via separate pointer
    seqstart_k.ToDevice(seqstart_k_host.data());
    seqstart_q_padded_buf.ToDevice(
        seqstart_q_with_padding_host.empty() ? nullptr : seqstart_q_with_padding_host.data());
    seqstart_k_padded_buf.ToDevice(seqlen_kpads[0] < 0 ? nullptr
                                                       : seqstart_k_with_padding_host.data());
    cu_seqlen_q_buf.ToDevice(cuq_cum.empty() ? nullptr : cuq_cum.data());
    cu_seqlen_kv_buf.ToDevice(cukv_cum.empty() ? nullptr : cukv_cum.data());
    seqlen_q_buf.ToDevice(has_group_q_padding ? seqlen_qs.data() : nullptr);
    seqlen_k_buf.ToDevice(has_group_k_padding ? seqlen_ks.data() : nullptr);
    block_scale_seqstart_q_buf.ToDevice(
        need_block_scale_seqstart_buf ? block_scale_seqstart_q_host.data() : nullptr);
    block_scale_seqstart_k_buf.ToDevice(
        need_block_scale_seqstart_buf ? block_scale_seqstart_k_host.data() : nullptr);

    // clang-format off
    auto layout_str = [&](bool permute){
        if(permute) return std::string("bhsd");
        else return std::string("bshd");
    };
    auto io_layout = [&](bool iperm_, bool operm_) {
        if(iperm_ == operm_) return layout_str(iperm_);
        else return layout_str(iperm_) + std::string("-") + layout_str(operm_);
    };
    // clang-format on

    std::cout << "[" << data_type << "|" << mode << "|" << io_layout(i_perm, o_perm)
              << "] b:" << batch << ", h:" << nhead << "/" << nhead_k << ", s:" << seqlen_qs[0]
              << "/" << seqlen_ks[0]
              << (seqlen_kpads[0] < 0 ? ""
                                      : (std::string("(") + std::to_string(seqlen_kpads[0]) + ")"))
              << ", d:" << hdim_q << "/" << hdim_v << ", scale_s:" << scale_s
              << ", qscale:" << qscale << ", mask:" << mask
              << ", v:" << (is_v_rowmajor ? "r" : "c");
    // Padding / effective length diagnostic logging
    auto print_vec = [&](const char* label, const std::vector<int>& v) {
        if(v.empty())
            return;
        std::cout << ", " << label << ":[";
        for(std::size_t i = 0; i < v.size(); ++i)
        {
            if(i)
                std::cout << ",";
            std::cout << v[i];
        }
        std::cout << "]";
    };

    if(has_group_padding)
    {
        bool has_qpad = !seqstart_q_with_padding_host.empty();
        bool has_kpad = (seqlen_kpads[0] >= 0);
        if(has_qpad)
        {
            print_vec("q_logical", seqlen_qs);
            print_vec("q_padded", seqlen_qpads);
        }
        if(has_kpad)
        {
            print_vec("k_logical", seqlen_ks);
            print_vec("k_padded", seqlen_kpads);
        }
    }
    else if(has_batch_padding)
    {
        // derive effective lengths from cumulative arrays if present
        if(!cuq_cum.empty())
        {
            std::vector<int> eff_q(batch);
            for(int b_i = 0; b_i < batch; ++b_i)
                eff_q[b_i] = static_cast<int>(cuq_cum[b_i + 1] - cuq_cum[b_i]);
            print_vec("q_eff", eff_q);
        }
        if(!cukv_cum.empty())
        {
            std::vector<int> eff_kv(batch);
            for(int b_i = 0; b_i < batch; ++b_i)
                eff_kv[b_i] = static_cast<int>(cukv_cum[b_i + 1] - cukv_cum[b_i]);
            print_vec("kv_eff", eff_kv);
        }
    }

    std::cout << std::flush;

    const auto init_traits = [&](auto& traits) {
        traits.hdim_q        = hdim_q;
        traits.hdim_v        = hdim_v;
        traits.data_type     = data_type;
        traits.is_v_rowmajor = is_v_rowmajor;
        traits.is_group_mode = (mode == mode_enum::group);
        traits.mask_type     = mask.type;
        traits.qscale_type   = qscale.type;
    };

    const auto init_args = [&, k_paddings_ = seqlen_kpads](auto& args) {
        // setup stride_* arguments
        const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_o = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_lse = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_o   = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q   = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k   = (nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_v   = (nhead_k * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_lse = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_o   = (nhead * shape_seqlen_q * hdim_v);
        // setup split_stride_* arguments (only used in split-kv kernel)

        args.q_ptr    = q_buf.GetDeviceBuffer();
        args.k_ptr    = k_buf.GetDeviceBuffer();
        args.v_ptr    = v_buf.GetDeviceBuffer();
        args.batch    = batch;
        args.seqlen_q = shape_seqlen_q; // unused in group mode
        args.hdim_q   = hdim_q;
        args.hdim_v   = hdim_v;
        args.nhead_q  = nhead;
        args.nhead_k  = nhead_k;

        args.stride_q       = stride_q;
        args.stride_k       = stride_k;
        args.stride_v       = stride_v;
        args.nhead_stride_q = nhead_stride_q;
        args.nhead_stride_k = nhead_stride_k;
        args.nhead_stride_v = nhead_stride_v;
        args.batch_stride_q = batch_stride_q;
        args.batch_stride_k = batch_stride_k;
        args.batch_stride_v = batch_stride_v;

        // Setup sageattn_fwd_args
        args.o_ptr = o_buf.GetDeviceBuffer();

        args.seqlen_k     = shape_seqlen_k; // unused in group mode (or kvcache enabled)
        args.max_seqlen_q = max_seqlen_q;

        args.scale_s = scale_s;

        args.stride_o         = stride_o;
        args.nhead_stride_lse = nhead_stride_lse;
        args.nhead_stride_o   = nhead_stride_o;
        args.batch_stride_lse = batch_stride_lse;
        args.batch_stride_o   = batch_stride_o;

        args.window_size_left  = mask.left;
        args.window_size_right = mask.right;
        args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

        args.q_descale_ptr = q_descale_buf.GetDeviceBuffer();
        args.k_descale_ptr = k_descale_buf.GetDeviceBuffer();
        args.v_descale_ptr = v_descale_buf.GetDeviceBuffer();

        // BLOCKSCALE/PERWARP/PERTHREAD parameters
        if(qscale_uses_bwp)
        {
            args.nhead_stride_q_descale = num_block_scale_q;
            args.nhead_stride_k_descale = num_block_scale_k;
            // BLOCKSCALE, PERWARP, and PERTHREAD V all use per-channel scale: stride = hdim_v
            args.nhead_stride_v_descale = hdim_v;

            if(mode == mode_enum::batch)
            {
                args.batch_stride_q_descale = nhead * num_block_scale_q;
                args.batch_stride_k_descale = nhead_k * num_block_scale_k;
                // BLOCKSCALE, PERWARP, and PERTHREAD V all use per-channel scale: batch_stride =
                // nhead_k * hdim_v
                args.batch_stride_v_descale = nhead_k * hdim_v;
            }
            else // group mode
            {
                // BLOCKSCALE, PERWARP, and PERTHREAD all use block_scale_seqstart in group mode
                // They differ only in block size: BLOCKSCALE (Q:128, K:128), PERWARP (Q:32, K:64),
                // PERTHREAD (Q:4, K:16)
                args.block_scale_seqstart_q_ptr = block_scale_seqstart_q_buf.GetDeviceBuffer();
                args.block_scale_seqstart_k_ptr = block_scale_seqstart_k_buf.GetDeviceBuffer();
                // BLOCKSCALE, PERWARP, and PERTHREAD V all use per-channel scale: batch_stride =
                // nhead_k * hdim_v
                args.batch_stride_v_descale = nhead_k * hdim_v;
            }

            args.block_scale_size_q = block_scale_size_q_;
            args.block_scale_size_k = block_scale_size_k_;
        }

        // Sequence length and padding parameters (mode-specific)
        if(mode == mode_enum::group)
        {
            // Group mode: use physical (padded) cumulative starts + logical per-sequence
            // lengths

            // Physical cumulative starts (including padding)
            args.seqstart_q_ptr = has_group_q_padding && !seqstart_q_with_padding_host.empty()
                                      ? seqstart_q_padded_buf.GetDeviceBuffer()
                                      : seqstart_q.GetDeviceBuffer();
            args.seqstart_k_ptr = has_group_k_padding && !seqstart_k_with_padding_host.empty()
                                      ? seqstart_k_padded_buf.GetDeviceBuffer()
                                      : seqstart_k.GetDeviceBuffer();

            // Logical (unpadded) per-sequence lengths, used when padding is enabled
            args.seqlen_q_ptr = (has_group_q_padding && !seqstart_q_with_padding_host.empty())
                                    ? seqlen_q_buf.GetDeviceBuffer()
                                    : nullptr;
            args.seqlen_k_ptr = (has_group_k_padding && !seqstart_k_with_padding_host.empty())
                                    ? seqlen_k_buf.GetDeviceBuffer()
                                    : nullptr;
            // Cumulative lengths not used in group mode
            args.cu_seqlen_q_ptr = nullptr;
            args.cu_seqlen_k_ptr = nullptr;
        }
        else // mode == mode_enum::batch
        {
            // Batch mode: use cumulative logical lengths for tail padding

            // seqstart pointers not used in batch mode
            args.seqstart_q_ptr = nullptr;
            args.seqstart_k_ptr = nullptr;

            // seqlen_q_ptr/seqlen_k_ptr not used in batch mode
            args.seqlen_q_ptr = nullptr;
            args.seqlen_k_ptr = nullptr;

            // Cumulative logical lengths for effective length handling
            args.cu_seqlen_q_ptr = has_batch_q_padding && !cuq_cum.empty()
                                       ? cu_seqlen_q_buf.GetDeviceBuffer()
                                       : nullptr;
            args.cu_seqlen_k_ptr = has_batch_k_padding && !cukv_cum.empty()
                                       ? cu_seqlen_kv_buf.GetDeviceBuffer()
                                       : nullptr;
        }
    };

    // Run main SageAttention forward kernel
    sageattn_fwd_traits sageattn_traits;
    init_traits(sageattn_traits);

    sageattn_fwd_args sageattn_args;
    init_args(sageattn_args);

    const float ave_time = sageattn_fwd(sageattn_traits, sageattn_args, stream_config);
    if(ave_time < 0.0f)
    {
        std::cout << ", not supported yet" << std::flush << std::endl;
        return fwd_result::no_instance;
    }
    const float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    const float gb_per_sec = num_byte / 1.E6 / ave_time;
    if(stream_config.time_kernel_)
    {
        std::cout << std::fixed << ", " << std::setprecision(3) << ave_time << " ms, "
                  << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2)
                  << gb_per_sec << " GB/s" << std::flush;
    }

    bool pass = true;
    if(do_validation == 0)
    {
        std::cout << std::flush << std::endl;
    }
    else
    {
        o_buf.FromDevice(o_host.data());

        constexpr bool supports_qscale =
            std::is_same_v<DataTypeConfig, SageAttentionFwdFp8Bf16> ||
            std::is_same_v<DataTypeConfig, SageAttentionFwdI8Fp8Bf16> ||
            std::is_same_v<DataTypeConfig, SageAttentionFwdI4Fp8Bf16>;

        float scale_s_host = scale_s;
        float scale_p_host = 1.0f;
        float scale_o_host = 1.0f;

        if(qscale.type == quant_scale_enum::pertensor)
        {
            scale_s_host = scale_s * q_descale_host(0) * k_descale_host(0);
            scale_p_host = ck_tile::type_convert<float>(ck_tile::numeric<PDataType>::max());
            scale_o_host = v_descale_host(0) / scale_p_host;
        }

        auto p_compute_element_func = [&]() {
            if constexpr(supports_qscale)
                return ck_tile::scales{scale_p_host};
            else
                return ck_tile::identity{};
        }();

        auto oacc_element_func = [&]() {
            if constexpr(std::is_same_v<ODataType, ck_tile::fp8_t> && supports_qscale)
                return ck_tile::make_composes(ck_tile::saturates<ck_tile::fp8_t>{},
                                              ck_tile::scales{scale_o_host});
            else if constexpr(supports_qscale)
                return ck_tile::scales{scale_o_host};
            else
                return ck_tile::identity{};
        }();

        for(ck_tile::index_t wb = 0; wb < batch; ++wb)
        {
            ck_tile::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            ck_tile::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];
            if(mode == mode_enum::batch)
            {
                if(!cuq_cum.empty())
                {
                    real_seqlen_q = cuq_cum[wb + 1] - cuq_cum[wb];
                }
                if(!cukv_cum.empty())
                {
                    real_seqlen_k = cukv_cum[wb + 1] - cukv_cum[wb];
                }
            }

            // adjust matrix index according to the mode
            const ck_tile::index_t b_idx       = (mode == mode_enum::batch ? wb : 0);
            const ck_tile::index_t cache_b_idx = b_idx;
            // Use physical offset if padding info is valid (not -1) and buffers are available
            const ck_tile::index_t query_offset =
                (mode == mode_enum::batch
                     ? 0
                     : ((seqstart_q_with_padding_host.empty() || seqlen_qpads[0] < 0)
                            ? seqstart_q_host[wb]
                            : seqstart_q_with_padding_host[wb]));
            const ck_tile::index_t key_offset =
                (mode == mode_enum::batch
                     ? 0
                     : ((seqstart_k_with_padding_host.empty() || seqlen_kpads[0] < 0)
                            ? seqstart_k_host[wb]
                            : seqstart_k_with_padding_host[wb]));

            ck_tile::HostTensor<QDataType> q_host_ref({nhead, real_seqlen_q, hdim_q});
            ck_tile::HostTensor<KDataType> k_host_ref({nhead, real_seqlen_k, hdim_q});
            ck_tile::HostTensor<VDataType> v_host_ref({nhead, hdim_v, real_seqlen_k});
            ck_tile::HostTensor<ODataType> o_host_ref({nhead, real_seqlen_q, hdim_v});

            ck_tile::HostTensor<SMPLComputeDataType> s_host_ref(
                {nhead, real_seqlen_q, real_seqlen_k});
            ck_tile::HostTensor<PDataType> p_host_ref({nhead, real_seqlen_q, real_seqlen_k});

            ck_tile::index_t nr = nhead / nhead_k;

            // clang-format off
            // permute
            if(i_perm) q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b_idx, i[0], i[1] + query_offset, i[2]); });
            else       q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b_idx, i[1] + query_offset, i[0], i[2]); });
            // clang-format on

            {
                // clang-format off
                if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(cache_b_idx, i[0] / nr, i[1] + key_offset, i[2]); });
                else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(cache_b_idx, i[1] + key_offset, i[0] / nr, i[2]); });
                // clang-format on
            }

            {
                if(is_v_rowmajor)
                {
                    // clang-format off
                    //                                v_host_ref: [nhead, hdim, seq], v_host: [b, h_k, s, d]
                    if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[0] / nr, i[2] + key_offset, i[1]); });
                    //                                v_host_ref: [nhead, hdim, seq], v_host: [b, s, h_k, d]
                    else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[2] + key_offset, i[0] / nr, i[1]); });
                    // clang-format on
                }
                else
                {
                    // clang-format off
                    if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[0] / nr, i[1], i[2] + key_offset); });
                    else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[1], i[0] / nr, i[2] + key_offset); });
                    // clang-format on
                }
            }

            // reference
            // For packed types (pk_int4_t), unpack to float for host reference GEMM
            auto unpack_to_float = [](const auto& packed_tensor) {
                auto dims = packed_tensor.mDesc.get_lengths();
                ck_tile::HostTensor<float> unpacked({static_cast<ck_tile::index_t>(dims[0]),
                                                     static_cast<ck_tile::index_t>(dims[1]),
                                                     static_cast<ck_tile::index_t>(dims[2])});
                unpacked.ForEach([&](auto& self, auto idx) {
                    auto packed = packed_tensor(idx[0], idx[1], idx[2]);
                    auto fp32x2 = ck_tile::pk_int4_t_to_fp32x2_t(packed);
                    self(idx)   = (idx[2] % 2 == 0) ? fp32x2[0] : fp32x2[1];
                });
                return unpacked;
            };

            if(qscale_uses_bwp)
            {
                const ck_tile::index_t q_offset =
                    (mode == mode_enum::batch) ? 0 : block_scale_seqstart_q_host[wb];
                const ck_tile::index_t k_offset =
                    (mode == mode_enum::batch) ? 0 : block_scale_seqstart_k_host[wb];
                if constexpr(ck_tile::is_packed_type_v<QDataType>)
                {
                    auto q_f32 = unpack_to_float(q_host_ref);
                    auto k_f32 = unpack_to_float(k_host_ref);
                    ck_tile::reference_batched_quant_gemm<float,
                                                          float,
                                                          SaccDataType,
                                                          SMPLComputeDataType>(
                        q_f32,
                        k_f32,
                        s_host_ref,
                        ck_tile::idx_identity{},
                        ck_tile::idx_identity{},
                        [&](auto idx, auto value) {
                            return value * scale_s *
                                   q_descale_host(b_idx,
                                                  std::get<0>(idx),
                                                  q_offset +
                                                      std::get<1>(idx) / block_scale_size_q_) *
                                   k_descale_host(b_idx,
                                                  std::get<0>(idx) / nr,
                                                  k_offset +
                                                      std::get<2>(idx) / block_scale_size_k_);
                        });
                }
                else
                {
                    ck_tile::reference_batched_quant_gemm<QDataType,
                                                          KDataType,
                                                          SaccDataType,
                                                          SMPLComputeDataType>(
                        q_host_ref,
                        k_host_ref,
                        s_host_ref,
                        ck_tile::idx_identity{},
                        ck_tile::idx_identity{},
                        [&](auto idx, auto value) {
                            return value * scale_s *
                                   q_descale_host(b_idx,
                                                  std::get<0>(idx),
                                                  q_offset +
                                                      std::get<1>(idx) / block_scale_size_q_) *
                                   k_descale_host(b_idx,
                                                  std::get<0>(idx) / nr,
                                                  k_offset +
                                                      std::get<2>(idx) / block_scale_size_k_);
                        });
                }
            }
            else
            {
                if constexpr(ck_tile::is_packed_type_v<QDataType>)
                {
                    auto q_f32 = unpack_to_float(q_host_ref);
                    auto k_f32 = unpack_to_float(k_host_ref);
                    ck_tile::
                        reference_batched_gemm<float, float, SaccDataType, SMPLComputeDataType>(
                            q_f32,
                            k_f32,
                            s_host_ref,
                            ck_tile::identity{},
                            ck_tile::identity{},
                            ck_tile::scales(scale_s_host));
                }
                else
                {
                    ck_tile::reference_batched_gemm<QDataType,
                                                    KDataType,
                                                    SaccDataType,
                                                    SMPLComputeDataType>(
                        q_host_ref,
                        k_host_ref,
                        s_host_ref,
                        ck_tile::identity{},
                        ck_tile::identity{},
                        ck_tile::scales(scale_s_host));
                }
            }

            if(mask.type == mask_enum::no_mask)
            {
                ck_tile::reference_batched_masking<SaccDataType>(
                    s_host_ref, SageAttnMasks::NoMask{real_seqlen_q, real_seqlen_k});
            }
            else if(mask.type == mask_enum::window_generic)
            {
                // Match device: kernel sets is_top_left from (mask_type == MASK_FROM_TOP_LEFT);
                // window_generic maps to MASK_GENERIC, so is_top_left is false (not the default).
                ck_tile::reference_batched_masking<SaccDataType>(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<SageAttnMasks::GenericMask>(
                        mask.left, mask.right, 0, real_seqlen_q, real_seqlen_k, false));
            }
            else
            {
                // if left window size is negative, means causal
                // else means generic (for current batch)
                if(mask.left < 0)
                    ck_tile::reference_batched_masking<SaccDataType>(
                        s_host_ref,
                        ck_tile::make_generic_attention_mask_from_lr_window<
                            SageAttnMasks::CausalMask>(mask.left,
                                                       mask.right,
                                                       0,
                                                       real_seqlen_q,
                                                       real_seqlen_k,
                                                       mask.type == mask_enum::mask_top_left));
                else
                    ck_tile::reference_batched_masking<SaccDataType>(
                        s_host_ref,
                        ck_tile::make_generic_attention_mask_from_lr_window<
                            SageAttnMasks::GenericMask>(mask.left,
                                                        mask.right,
                                                        0,
                                                        real_seqlen_q,
                                                        real_seqlen_k,
                                                        mask.type == mask_enum::mask_top_left));
            }
            const ck_tile::HostTensor<SaccDataType> masked_s_host_ref = s_host_ref;
            ck_tile::reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(
                s_host_ref, p_host_ref, p_compute_element_func);

            if(qscale_uses_bwp)
            {
                // BLOCKSCALE, PERWARP, and PERTHREAD V all use per-channel scale (col-major)
                // v_descale shape: [batch, nhead_k, hdim_v]
                // Access by channel index: std::get<1>(idx) is the hdim dimension
                ck_tile::
                    reference_batched_quant_gemm<PDataType, VDataType, OaccDataType, ODataType>(
                        p_host_ref,
                        v_host_ref,
                        o_host_ref,
                        ck_tile::idx_identity{},
                        [&](auto idx, auto value) {
                            return ck_tile::type_convert<float>(value) *
                                   v_descale_host(wb,
                                                  std::get<0>(idx) / nr,
                                                  std::get<1>(idx)); // channel index
                        },
                        ck_tile::idx_identity{});
            }
            else
            {
                ck_tile::reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
                    p_host_ref,
                    v_host_ref,
                    o_host_ref,
                    ck_tile::identity{},
                    ck_tile::identity{},
                    oacc_element_func);
            }

            ck_tile::HostTensor<ODataType> o_host_result({nhead, real_seqlen_q, hdim_v});
            // clang-format off
            // permute
            if(o_perm) o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b_idx, idx[0], idx[1] + query_offset, idx[2]); });
            else       o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b_idx, idx[1] + query_offset, idx[0], idx[2]); });
            // clang-format on

            auto [rtol, atol] = get_elimit<DataTypeConfig>(init_method);
            bool cur_pass     = ck_tile::check_err(o_host_result,
                                               o_host_ref,
                                               std::string("OUT Error: Incorrect results!"),
                                               rtol,
                                               atol);
            pass &= cur_pass;
            if(!cur_pass)
            {
                std::cerr << "OUT mismatch found at batch: " << wb << std::endl
                          << "\tseqlen_q: " << real_seqlen_q << std::endl
                          << "\tseqlen_k: " << real_seqlen_k << std::endl
                          << "\tseqstart_q (logical): " << seqstart_q_host << std::endl
                          << "\tseqstart_q (physical): " << seqstart_q_with_padding_host
                          << std::endl
                          << "\tseqstart_k (logical): " << seqstart_k_host << std::endl
                          << "\tseqstart_k (physical): " << seqstart_k_with_padding_host
                          << std::endl
                          << "\tquery_offset used: " << query_offset << std::endl
                          << "\tkey_offset used: " << key_offset << std::endl;

                break;
            }
        }

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    if(json)
    {
        dump_sageattn_fwd_json_results(
            *json,
            data_type,
            mode == mode_enum::batch ? "batch" : "group",
            io_layout(i_perm, o_perm),
            batch,
            nhead,
            nhead_k,
            seqlen_qs[0],
            seqlen_ks[0],
            seqlen_kpads[0],
            hdim_q,
            hdim_v,
            scale_s,
            0.0f,  // p_drop (dropout disabled for sageattention)
            false, // lse (always disabled for sageattention)
            [&qscale]() {
                std::ostringstream ss;
                qscale.serialize(ss);
                return ss.str();
            }(),
            "no_bias",
            is_v_rowmajor ? "r" : "c",
            pass,
            ave_time,
            tflops,
            gb_per_sec);
    }

    return pass ? fwd_result::success : fwd_result::failure;
}
