// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#include <ck_tile/core/numeric/math.hpp>
#include <ck_tile/core/numeric/type_convert.hpp>

#define CK_TILE_ATTENTION_LOGITS_SOFT_CAP_TANH 0
#define CK_TILE_ATTENTION_LOGITS_SOFT_CAP_SOFTSIGN 1

#ifndef CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT
#define CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT CK_TILE_ATTENTION_LOGITS_SOFT_CAP_TANH
#endif

namespace ck_tile {

template <typename ImplMask>
struct StandardAttentionParams
{
    __device__ __host__ StandardAttentionParams(const ImplMask& impl_mask_, float sm_scale_)
        : impl_mask(impl_mask_), sm_scale(sm_scale_)
    {
    }

    const ImplMask& impl_mask;
    float sm_scale;
};

template <typename ImplMask, bool UseExp2 = false>
struct LogitsSoftCapParams
{
    __device__
    LogitsSoftCapParams(const ImplMask& impl_mask_, float sm_scale_, float logits_soft_cap_)
        : impl_mask(impl_mask_), sm_scale(sm_scale_), logits_soft_cap(logits_soft_cap_)
    {
        if(0.f < logits_soft_cap)
        {
            logits_soft_cap_rcp = __builtin_amdgcn_rcpf(logits_soft_cap);
        }
        else
        {
            logits_soft_cap_rcp = 0.f;
        }

        // move computation here to prevent compiler from generating inefficient instruction
        // sequence
        if constexpr(UseExp2)
        {
            logits_soft_cap     = log2e_v<float> * logits_soft_cap;
            logits_soft_cap_rcp = sm_scale * log2e_rcp_v<float> * logits_soft_cap_rcp;
        }
    }

    __host__
    LogitsSoftCapParams(const ImplMask& impl_mask_, float sm_scale_, float logits_soft_cap_)
        : impl_mask(impl_mask_), sm_scale(sm_scale_), logits_soft_cap(logits_soft_cap_)
    {
        if(0.f < logits_soft_cap)
        {
            logits_soft_cap_rcp = 1.f / logits_soft_cap;
        }
        else
        {
            logits_soft_cap_rcp = 0.f;
        }

        // move computation here to prevent compiler from generating inefficient instruction
        // sequence
        if constexpr(UseExp2)
        {
            logits_soft_cap     = log2e_v<float> * logits_soft_cap;
            logits_soft_cap_rcp = sm_scale * log2e_rcp_v<float> * logits_soft_cap_rcp;
        }
    }

    __device__ __host__ LogitsSoftCapParams(const ImplMask& impl_mask_,
                                            float sm_scale_,
                                            float logits_soft_cap_,
                                            float logits_soft_cap_rcp_)
        : impl_mask(impl_mask_),
          sm_scale(sm_scale_),
          logits_soft_cap(logits_soft_cap_),
          logits_soft_cap_rcp(logits_soft_cap_rcp_)
    {
        // move computation here to prevent compiler from generating inefficient instruction
        // sequence
        if constexpr(UseExp2)
        {
            logits_soft_cap     = log2e_v<float> * logits_soft_cap;
            logits_soft_cap_rcp = sm_scale * log2e_rcp_v<float> * logits_soft_cap_rcp;
        }
    }

    const ImplMask& impl_mask;
    float sm_scale;
    float logits_soft_cap;
    float logits_soft_cap_rcp;
};

struct StandardAttention
{
    __device__ __host__ StandardAttention() = default;

    template <typename Params, typename T>
    __device__ __forceinline__ T QueryTransform(const Params& params, T q) const
    {
        return type_convert<float>(q) * params.sm_scale;
    }

    /// NOTICE: For better performance, we simpliy transform thread buffer without calculating
    /// qo_idx/kv_idx.
    template <typename Params, typename T>
    __device__ __forceinline__ T LogitsTransform([[maybe_unused]] const Params& params,
                                                 T logits,
                                                 [[maybe_unused]] uint32_t batch_idx,
                                                 /*uint32_t qo_idx, uint32_t kv_idx,*/
                                                 [[maybe_unused]] uint32_t qo_head_idx,
                                                 [[maybe_unused]] uint32_t kv_head_idx) const
    {
        return logits;
    }

    template <typename Params>
    __device__ __forceinline__ bool LogitsMask(const Params& params,
                                               [[maybe_unused]] uint32_t batch_idx,
                                               uint32_t qo_idx,
                                               uint32_t kv_idx,
                                               [[maybe_unused]] uint32_t qo_head_idx,
                                               [[maybe_unused]] uint32_t kv_head_idx) const
    {
        return !params.impl_mask.IsOutOfBound(qo_idx, kv_idx);
    }
};

template <bool UseExp2 = false>
struct LogitsSoftCap
{
    __device__ __host__ LogitsSoftCap() = default;

    template <typename Params, typename T>
    __device__ __forceinline__ T QueryTransform(const Params& params, T q) const
    {
        if constexpr(UseExp2)
        {
            return q;
        }
        else
        {
            return type_convert<float>(q) * params.sm_scale;
        }
    }

    /// NOTICE: For better performance, we simpliy transform thread buffer without calculating
    /// qo_idx/kv_idx.
    template <typename Params, typename T>
    __device__ __forceinline__ T LogitsTransform(const Params& params,
                                                 T logits,
                                                 [[maybe_unused]] uint32_t batch_idx,
                                                 /*uint32_t qo_idx, uint32_t kv_idx,*/
                                                 [[maybe_unused]] uint32_t qo_head_idx,
                                                 [[maybe_unused]] uint32_t kv_head_idx) const
    {
        if constexpr(UseExp2)
        {
#if CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_TANH
            return params.logits_soft_cap *
                   tanh_fast<float>(type_convert<float>(logits) * params.logits_soft_cap_rcp);
#elif CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_SOFTSIGN
            return params.sm_scale * type_convert<float>(logits) *
                   rcp<float>(1.f + abs(type_convert<float>(logits) * params.logits_soft_cap_rcp));
#endif
        }
        else
        {
#if CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_TANH
            return params.logits_soft_cap *
                   tanhf(type_convert<float>(logits) * params.logits_soft_cap_rcp);
#elif CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_SOFTSIGN
            return type_convert<float>(logits) *
                   rcp<float>(1.f + abs(type_convert<float>(logits) * params.logits_soft_cap_rcp));
#endif
        }
    }

    template <typename Params>
    __device__ __forceinline__ bool LogitsMask(const Params& params,
                                               [[maybe_unused]] uint32_t batch_idx,
                                               uint32_t qo_idx,
                                               uint32_t kv_idx,
                                               [[maybe_unused]] uint32_t qo_head_idx,
                                               [[maybe_unused]] uint32_t kv_head_idx) const
    {
        return !params.impl_mask.IsOutOfBound(qo_idx, kv_idx);
    }
};

constexpr uint32_t CUSTOM_MASK     = 1U;
constexpr uint32_t SLIDING_WINDOW  = 2U;
constexpr uint32_t LOGITS_SOFT_CAP = 4U;
constexpr uint32_t ALIBI           = 8U;

template <uint32_t VARIANT_CODE, bool UseExp2 = false>
struct ComposedAttention
{
    static constexpr bool use_exp2 = UseExp2;

    static constexpr bool use_logits_soft_cap = (VARIANT_CODE & LOGITS_SOFT_CAP) != 0;

    __device__ __host__ ComposedAttention() = default;

    template <typename Params, typename T>
    __device__ __forceinline__ T QueryTransform(const Params& params, T q) const
    {
        if constexpr(use_logits_soft_cap && UseExp2)
        {
            return q;
        }
        return type_convert<float>(q) * params.sm_scale;
    }

    /// NOTICE: For better performance, we simpliy transform thread buffer without calculating
    /// qo_idx/kv_idx.
    template <typename Params, typename T>
    __device__ __forceinline__ T LogitsTransform(const Params& params,
                                                 T logits,
                                                 [[maybe_unused]] uint32_t batch_idx,
                                                 /*uint32_t qo_idx, uint32_t kv_idx,*/
                                                 [[maybe_unused]] uint32_t qo_head_idx,
                                                 [[maybe_unused]] uint32_t kv_head_idx) const
    {
        if constexpr(use_logits_soft_cap)
        {
            if constexpr(UseExp2)
            {
#if CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_TANH
                return params.logits_soft_cap *
                       tanh_fast<float>(type_convert<float>(logits) * params.logits_soft_cap_rcp);
#elif CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_SOFTSIGN
                return params.sm_scale * type_convert<float>(logits) *
                       rcp<float>(1.f +
                                  abs(type_convert<float>(logits) * params.logits_soft_cap_rcp));
#endif
            }
            else
            {
#if CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_TANH
                return params.logits_soft_cap *
                       tanhf(type_convert<float>(logits) * params.logits_soft_cap_rcp);
#elif CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT == CK_TILE_ATTENTION_LOGITS_SOFT_CAP_SOFTSIGN
                return type_convert<float>(logits) *
                       rcp<float>(1.f +
                                  abs(type_convert<float>(logits) * params.logits_soft_cap_rcp));
#endif
            }
        }
        return logits;
    }

    template <typename Params>
    __device__ __forceinline__ bool LogitsMask(const Params& params,
                                               [[maybe_unused]] uint32_t batch_idx,
                                               uint32_t qo_idx,
                                               uint32_t kv_idx,
                                               [[maybe_unused]] uint32_t qo_head_idx,
                                               [[maybe_unused]] uint32_t kv_head_idx) const
    {
        return !params.impl_mask.IsOutOfBound(qo_idx, kv_idx);
    }
};

} // namespace ck_tile
