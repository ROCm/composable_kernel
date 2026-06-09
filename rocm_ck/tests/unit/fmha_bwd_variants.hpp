// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Frozen spec table used by the FMHA BWD unit tests.
// Host-only header -- no CK Tile dependency, no HIP dependency.
//
// Three kernel families, three variant arrays, three consteval name lookups.
//
// Include contract: this header includes only _spec.hpp (not _api.hpp).
// Callers who need grid_size must include the _api.hpp header separately.

#pragma once

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>
#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>
#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

#include <iterator> // std::size
#include <string_view>

namespace rocm_ck {

// =========================================================================
// OGradDotO variants
// =========================================================================

struct FmhaBwdOGradDotOVariant
{
    const char* name;
    FmhaBwdOGradDotOSpec spec;
};

// clang-format off
static constexpr FmhaBwdOGradDotOVariant ALL_OGRAD_DOT_O_VARIANTS[] = {
    {"fmha_bwd_ograd_dot_o_fp16_d128_batch", makeSpec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_bf16_d128_batch", makeSpec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::BF16, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d64_batch", makeSpec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 64,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d128_group", makeSpec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_bf16_d128_group", makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::BF16, .hdim_v = 128,
                    .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d128_batch_npad", makeSpec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}})},
};
// clang-format on

static constexpr int ALL_OGRAD_DOT_O_VARIANTS_COUNT = std::size(ALL_OGRAD_DOT_O_VARIANTS);

// =========================================================================
// DqDkDv variants
// =========================================================================

struct FmhaBwdDQDKDVVariant
{
    const char* name;
    FmhaBwdDQDKDVSpec spec;
};

// clang-format off
static constexpr FmhaBwdDQDKDVVariant ALL_DQDKDV_VARIANTS[] = {
    {"fmha_bwd_dqdkdv_fp16_d128_batch", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_bf16_d128_batch", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::BF16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_cmask", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    // Bottom-right causal: same compiled spec as _cmask. The mask_type is
    // selected at runtime via args.scalars[fmha_bwd_dqdkdv_slots::MASK_TYPE].
    //
    // _cmask_br and _swa are reachable via name lookup
    // (fmha_bwd_dqdkdv_variant_spec("<exact name>")) or by iterating
    // ALL_DQDKDV_VARIANTS.
    {"fmha_bwd_dqdkdv_fp16_d128_batch_cmask_br", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    // Sliding-window attention: same compiled spec as _cmask. window_size_left
    // and window_size_right are runtime-parametrized via scalar slots.
    {"fmha_bwd_dqdkdv_fp16_d128_batch_swa", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_det", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.is_deterministic = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_group", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_bf16_d128_group", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::BF16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_group_cmask", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_group_det", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.is_deterministic = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_group_dropout", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.has_dropout = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_group_ebias", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_group_alibi", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.bias_type = FmhaBiasType::ALIBI,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_group_ebias_dbias", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE,
                       .has_bias_grad = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_ebias", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_bf16_d128_batch_ebias", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::BF16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_alibi", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.bias_type = FmhaBiasType::ALIBI,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_bf16_d128_batch_alibi", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::BF16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.bias_type = FmhaBiasType::ALIBI,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_ebias_dbias", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE,
                       .has_bias_grad = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_bf16_d128_batch_ebias_dbias", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::BF16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE,
                       .has_bias_grad = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_dropout", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.has_dropout = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_cmask_det", makeSpec(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                       .is_deterministic = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
};
// clang-format on

static constexpr int ALL_DQDKDV_VARIANTS_COUNT = std::size(ALL_DQDKDV_VARIANTS);

// =========================================================================
// ConvertDQ variants
// =========================================================================

struct FmhaBwdConvertDQVariant
{
    const char* name;
    FmhaBwdConvertDQSpec spec;
};

// clang-format off
static constexpr FmhaBwdConvertDQVariant ALL_CONVERT_DQ_VARIANTS[] = {
    {"fmha_bwd_convert_dq_fp16_d64_batch_det", makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64,
                   .mode = FmhaMode::BATCH},
        .algorithm = {}})},
    {"fmha_bwd_convert_dq_fp16_d64_group_det", makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64,
                   .mode = FmhaMode::GROUP},
        .algorithm = {}})},
    {"fmha_bwd_convert_dq_bf16_d64_batch_det", makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::BF16, .hdim_q = 64,
                   .mode = FmhaMode::BATCH},
        .algorithm = {}})},
    {"fmha_bwd_convert_dq_bf16_d64_group_det", makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::BF16, .hdim_q = 64,
                   .mode = FmhaMode::GROUP},
        .algorithm = {}})},
    {"fmha_bwd_convert_dq_fp16_d128_batch_det", makeSpec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {}})},
    {"fmha_bwd_convert_dq_fp16_d128_group_det", makeSpec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {}})},
    {"fmha_bwd_convert_dq_bf16_d128_batch_det", makeSpec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::BF16, .hdim_q = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {}})},
    {"fmha_bwd_convert_dq_bf16_d128_group_det", makeSpec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::BF16, .hdim_q = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {}})},
    {"fmha_bwd_convert_dq_fp16_d256_batch_det", makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 256,
                    .mode = FmhaMode::BATCH},
        .algorithm = {}})},
    {"fmha_bwd_convert_dq_fp16_d256_group_det", makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 256,
                    .mode = FmhaMode::GROUP},
        .algorithm = {}})},
    {"fmha_bwd_convert_dq_bf16_d256_batch_det", makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::BF16, .hdim_q = 256,
                    .mode = FmhaMode::BATCH},
        .algorithm = {}})},
    {"fmha_bwd_convert_dq_bf16_d256_group_det", makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::BF16, .hdim_q = 256,
                    .mode = FmhaMode::GROUP},
        .algorithm = {}})},
};
// clang-format on

static constexpr int ALL_CONVERT_DQ_VARIANTS_COUNT = std::size(ALL_CONVERT_DQ_VARIANTS);

// =========================================================================
// Consteval name-based lookup (compile-time variant selection by name)
// =========================================================================

/// Look up an OGradDotO variant spec by name at compile time.
consteval FmhaBwdOGradDotOSpec fmha_bwd_ograd_dot_o_variant_spec(const char* name)
{
    for(const auto& v : ALL_OGRAD_DOT_O_VARIANTS)
        if(std::string_view(v.name) == name)
            return v.spec;
    throw "unknown FMHA BWD OGradDotO variant name";
}

/// Look up a DqDkDv variant spec by name at compile time.
consteval FmhaBwdDQDKDVSpec fmha_bwd_dqdkdv_variant_spec(const char* name)
{
    for(const auto& v : ALL_DQDKDV_VARIANTS)
        if(std::string_view(v.name) == name)
            return v.spec;
    throw "unknown FMHA BWD DqDkDv variant name";
}

/// Look up a ConvertDQ variant spec by name at compile time.
consteval FmhaBwdConvertDQSpec fmha_bwd_convert_dq_variant_spec(const char* name)
{
    for(const auto& v : ALL_CONVERT_DQ_VARIANTS)
        if(std::string_view(v.name) == name)
            return v.spec;
    throw "unknown FMHA BWD ConvertDQ variant name";
}

} // namespace rocm_ck
