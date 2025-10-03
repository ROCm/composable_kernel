// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "test_gemm_streamk_types.hpp"

template<typename M_MacroTile,
         typename N_MacroTile,
         typename K_MacroTile,
         typename M_Warps,
         typename N_Warps,
         typename K_Warps,
         typename M_MmaTile,
         typename N_MmaTile,
         typename K_MmaTile,
         typename PipelineType,
         typename Persistent>
struct F16Layouts
{
    // clang-format off
    // For CDNA, we support [A, B, Acc, C] = [f16, f16, f32, f16] and [f16, f16, f32, f32]:
    using f16_f16_f32_f16 = Layouts<F16, F16, F32, F16, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>;
    using f16_f16_f32_f32 = Layouts<F16, F16, F32, F32, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>;
    using rrr = detail::combine_t<typename f16_f16_f32_f16::rrr, typename f16_f16_f32_f32::rrr>;
    using rrc = detail::combine_t<typename f16_f16_f32_f16::rrc, typename f16_f16_f32_f32::rrc>;
    using rcr = detail::combine_t<typename f16_f16_f32_f16::rcr, typename f16_f16_f32_f32::rcr>;
    using rcc = detail::combine_t<typename f16_f16_f32_f16::rcc, typename f16_f16_f32_f32::rcc>;
    using crr = detail::combine_t<typename f16_f16_f32_f16::crr, typename f16_f16_f32_f32::crr>;
    using crc = detail::combine_t<typename f16_f16_f32_f16::crc, typename f16_f16_f32_f32::crc>;
    using ccr = detail::combine_t<typename f16_f16_f32_f16::ccr, typename f16_f16_f32_f32::ccr>;
    using ccc = detail::combine_t<typename f16_f16_f32_f16::ccc, typename f16_f16_f32_f32::ccc>;
    // clang-format on
};

#define CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE) \
    DATA_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE

// #define CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE_X, MACRO_TILE_Y, MACRO_TILE, WORKGROUP, WAVE_TILE) \
//     DATA_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE

template<typename PipelineType,
         typename Persistent>
struct F16Set
{
    // clang-format off
    // 32x32x16
    // 2x2x1
    using f16_128x128x32_2x2x1_32x32x16 =  F16Layouts<I128, I128,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x128x64_2x2x1_32x32x16 =  F16Layouts<I128, I128,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x128x128_2x2x1_32x32x16 = F16Layouts<I128, I128, I128, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x128x32_2x2x1_32x32x16 =  F16Layouts<I256, I128,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x128x64_2x2x1_32x32x16 =  F16Layouts<I256, I128,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x256x32_2x2x1_32x32x16 =  F16Layouts<I128, I256,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x256x64_2x2x1_32x32x16 =  F16Layouts<I128, I256,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x256x32_2x2x1_32x32x16 =  F16Layouts<I256, I256,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x256x64_2x2x1_32x32x16 =  F16Layouts<I256, I256,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;

    // 32x32x16
    // 4x1x1
    using f16_128x128x32_4x1x1_32x32x16 =  F16Layouts<I128, I128,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x128x64_4x1x1_32x32x16 =  F16Layouts<I128, I128,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x128x128_4x1x1_32x32x16 = F16Layouts<I128, I128, I128, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x128x32_4x1x1_32x32x16 =  F16Layouts<I256, I128,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x128x64_4x1x1_32x32x16 =  F16Layouts<I256, I128,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x256x32_4x1x1_32x32x16 =  F16Layouts<I128, I256,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x256x64_4x1x1_32x32x16 =  F16Layouts<I128, I256,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x256x32_4x1x1_32x32x16 =  F16Layouts<I256, I256,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x256x64_4x1x1_32x32x16 =  F16Layouts<I256, I256,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;

    // 32x32x16
    // 1x4x1
    using f16_128x128x32_1x4x1_32x32x16 =  F16Layouts<I128, I128,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x128x64_1x4x1_32x32x16 =  F16Layouts<I128, I128,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x128x128_1x4x1_32x32x16 = F16Layouts<I128, I128, I128, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x128x32_1x4x1_32x32x16 =  F16Layouts<I256, I128,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x128x64_1x4x1_32x32x16 =  F16Layouts<I256, I128,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x256x32_1x4x1_32x32x16 =  F16Layouts<I128, I256,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_128x256x64_1x4x1_32x32x16 =  F16Layouts<I128, I256,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x256x32_1x4x1_32x32x16 =  F16Layouts<I256, I256,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using f16_256x256x64_1x4x1_32x32x16 =  F16Layouts<I256, I256,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;

    // 16x16x16
    // 2x2x1
    using f16_128x128x32_2x2x1_16x16x16 =  F16Layouts<I128, I128,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x128x64_2x2x1_16x16x16 =  F16Layouts<I128, I128,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x128x128_2x2x1_16x16x16 = F16Layouts<I128, I128, I128, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x128x32_2x2x1_16x16x16 =  F16Layouts<I256, I128,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x128x64_2x2x1_16x16x16 =  F16Layouts<I256, I128,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x256x32_2x2x1_16x16x16 =  F16Layouts<I128, I256,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x256x64_2x2x1_16x16x16 =  F16Layouts<I128, I256,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x256x32_2x2x1_16x16x16 =  F16Layouts<I256, I256,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x256x64_2x2x1_16x16x16 =  F16Layouts<I256, I256,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;

    // 16x16x16
    // 4x1x1
    using f16_128x128x32_4x1x1_16x16x16 =  F16Layouts<I128, I128,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x128x64_4x1x1_16x16x16 =  F16Layouts<I128, I128,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x128x128_4x1x1_16x16x16 = F16Layouts<I128, I128, I128, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x128x32_4x1x1_16x16x16 =  F16Layouts<I256, I128,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x128x64_4x1x1_16x16x16 =  F16Layouts<I256, I128,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x256x32_4x1x1_16x16x16 =  F16Layouts<I128, I256,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x256x64_4x1x1_16x16x16 =  F16Layouts<I128, I256,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x256x32_4x1x1_16x16x16 =  F16Layouts<I256, I256,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x256x64_4x1x1_16x16x16 =  F16Layouts<I256, I256,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;

    // 16x16x16
    // 1x4x1
    using f16_128x128x32_1x4x1_16x16x16 =  F16Layouts<I128, I128,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x128x64_1x4x1_16x16x16 =  F16Layouts<I128, I128,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x128x128_1x4x1_16x16x16 = F16Layouts<I128, I128, I128, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x128x32_1x4x1_16x16x16 =  F16Layouts<I256, I128,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x128x64_1x4x1_16x16x16 =  F16Layouts<I256, I128,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x256x32_1x4x1_16x16x16 =  F16Layouts<I128, I256,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_128x256x64_1x4x1_16x16x16 =  F16Layouts<I128, I256,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x256x32_1x4x1_16x16x16 =  F16Layouts<I256, I256,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using f16_256x256x64_1x4x1_16x16x16 =  F16Layouts<I256, I256,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    // clang-format on
};

#define CONCATENATE_FORMATTED_NAME(DATA_TYPE, LAYOUT_TYPE, PIPELINE_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT) \
    DATA_TYPE##_##LAYOUT_TYPE##_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT



#define FORMAT_MEMBER(FORMATTED_NAME, SET_NAME, PIPELINE_TYPE, PERSISTENT, MEMBER_NAME) \
    using FORMATTED_NAME = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::MEMBER_NAME;

#define EXPAND_MEMBER_LAYOUTS(SET_NAME, DATA_TYPE, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PIPELINE_TYPE, PERSISTENT) \
    FORMAT_MEMBER(CONCATENATE_FORMATTED_NAME(DATA_TYPE, rrr, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT), SET_NAME, PIPELINE_TYPE, PERSISTENT, CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE)::rrr) \
    FORMAT_MEMBER(CONCATENATE_FORMATTED_NAME(DATA_TYPE, rrc, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT), SET_NAME, PIPELINE_TYPE, PERSISTENT, CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE)::rrc) \
    FORMAT_MEMBER(CONCATENATE_FORMATTED_NAME(DATA_TYPE, rcr, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT), SET_NAME, PIPELINE_TYPE, PERSISTENT, CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE)::rcr) \
    FORMAT_MEMBER(CONCATENATE_FORMATTED_NAME(DATA_TYPE, rcc, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT), SET_NAME, PIPELINE_TYPE, PERSISTENT, CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE)::rcc) \
    FORMAT_MEMBER(CONCATENATE_FORMATTED_NAME(DATA_TYPE, crr, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT), SET_NAME, PIPELINE_TYPE, PERSISTENT, CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE)::crr) \
    FORMAT_MEMBER(CONCATENATE_FORMATTED_NAME(DATA_TYPE, crc, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT), SET_NAME, PIPELINE_TYPE, PERSISTENT, CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE)::crc) \
    FORMAT_MEMBER(CONCATENATE_FORMATTED_NAME(DATA_TYPE, ccr, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT), SET_NAME, PIPELINE_TYPE, PERSISTENT, CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE)::ccr) \
    FORMAT_MEMBER(CONCATENATE_FORMATTED_NAME(DATA_TYPE, ccc, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT), SET_NAME, PIPELINE_TYPE, PERSISTENT, CONCATENATE_MEMBER_NAME(DATA_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE)::ccc)

#define EXPAND_
EXPAND_MEMBER_LAYOUTS(F16Set, f16, compv3, 256x256x32, 2x2x1, 32x32x16, CompV3, NonPersistent);
// #define EXPAND_SET(SET_NAME, DATA_TYPE, PIPELINE_TYPE, MACRO_TILE, WORKGROUP, WAVE_TILE, PERSISTENT) \
//     using DATA_TYPE##_rrr_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::##DATATYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE::rrr; \
//     using DATA_TYPE##_rrc_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::##DATATYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE::rrc; \
//     using DATA_TYPE##_rcr_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::##DATATYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE::rcr; \
//     using DATA_TYPE##_rcc_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::##DATATYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE::rcc; \
//     using DATA_TYPE##_crr_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::##DATATYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE::crr; \
//     using DATA_TYPE##_crc_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::##DATATYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE::crc; \
//     using DATA_TYPE##_ccr_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::##DATATYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE::ccr; \
//     using DATA_TYPE##_ccc_##PIPELINE_TYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE##_##PERSISTENT = typename SET_NAME<PIPELINE_TYPE, PERSISTENT>::##DATATYPE##_##MACRO_TILE##_##WORKGROUP##_##WAVE_TILE::ccc;

// clang-format off
// 32x32x16
// 2x2x1
using f16_rrr_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rrr;
using f16_rrc_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rrc;
using f16_rcr_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rcr;
using f16_rcc_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rcc;
using f16_crr_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::crr;
using f16_crc_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::crc;
using f16_ccr_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::ccr;
using f16_ccc_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::ccc;

using f16_rrr_mem_128x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x64_2x2x1_32x32x16::rrr;
using f16_rrc_mem_128x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x64_2x2x1_32x32x16::rrc;
using f16_rcr_mem_128x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x64_2x2x1_32x32x16::rcr;
using f16_rcc_mem_128x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x64_2x2x1_32x32x16::rcc;
using f16_crr_mem_128x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x64_2x2x1_32x32x16::crr;
using f16_crc_mem_128x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x64_2x2x1_32x32x16::crc;
using f16_ccr_mem_128x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x64_2x2x1_32x32x16::ccr;
using f16_ccc_mem_128x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x64_2x2x1_32x32x16::ccc;

using f16_rrr_mem_128x128x128_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x128_2x2x1_32x32x16::rrr;
using f16_rrc_mem_128x128x128_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x128_2x2x1_32x32x16::rrc;
using f16_rcr_mem_128x128x128_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x128_2x2x1_32x32x16::rcr;
using f16_rcc_mem_128x128x128_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x128_2x2x1_32x32x16::rcc;
using f16_crr_mem_128x128x128_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x128_2x2x1_32x32x16::crr;
using f16_crc_mem_128x128x128_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x128_2x2x1_32x32x16::crc;
using f16_ccr_mem_128x128x128_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x128_2x2x1_32x32x16::ccr;
using f16_ccc_mem_128x128x128_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x128x128_2x2x1_32x32x16::ccc;

using f16_rrr_mem_256x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x32_2x2x1_32x32x16::rrr;
using f16_rrc_mem_256x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x32_2x2x1_32x32x16::rrc;
using f16_rcr_mem_256x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x32_2x2x1_32x32x16::rcr;
using f16_rcc_mem_256x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x32_2x2x1_32x32x16::rcc;
using f16_crr_mem_256x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x32_2x2x1_32x32x16::crr;
using f16_crc_mem_256x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x32_2x2x1_32x32x16::crc;
using f16_ccr_mem_256x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x32_2x2x1_32x32x16::ccr;
using f16_ccc_mem_256x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x32_2x2x1_32x32x16::ccc;

using f16_rrr_mem_256x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x64_2x2x1_32x32x16::rrr;
using f16_rrc_mem_256x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x64_2x2x1_32x32x16::rrc;
using f16_rcr_mem_256x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x64_2x2x1_32x32x16::rcr;
using f16_rcc_mem_256x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x64_2x2x1_32x32x16::rcc;
using f16_crr_mem_256x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x64_2x2x1_32x32x16::crr;
using f16_crc_mem_256x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x64_2x2x1_32x32x16::crc;
using f16_ccr_mem_256x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x64_2x2x1_32x32x16::ccr;
using f16_ccc_mem_256x128x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x128x64_2x2x1_32x32x16::ccc;

using f16_rrr_mem_128x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x32_2x2x1_32x32x16::rrr;
using f16_rrc_mem_128x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x32_2x2x1_32x32x16::rrc;
using f16_rcr_mem_128x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x32_2x2x1_32x32x16::rcr;
using f16_rcc_mem_128x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x32_2x2x1_32x32x16::rcc;
using f16_crr_mem_128x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x32_2x2x1_32x32x16::crr;
using f16_crc_mem_128x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x32_2x2x1_32x32x16::crc;
using f16_ccr_mem_128x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x32_2x2x1_32x32x16::ccr;
using f16_ccc_mem_128x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x32_2x2x1_32x32x16::ccc;

using f16_rrr_mem_128x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x64_2x2x1_32x32x16::rrr;
using f16_rrc_mem_128x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x64_2x2x1_32x32x16::rrc;
using f16_rcr_mem_128x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x64_2x2x1_32x32x16::rcr;
using f16_rcc_mem_128x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x64_2x2x1_32x32x16::rcc;
using f16_crr_mem_128x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x64_2x2x1_32x32x16::crr;
using f16_crc_mem_128x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x64_2x2x1_32x32x16::crc;
using f16_ccr_mem_128x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x64_2x2x1_32x32x16::ccr;
using f16_ccc_mem_128x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_128x256x64_2x2x1_32x32x16::ccc;

using f16_rrr_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rrr;
using f16_rrc_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rrc;
using f16_rcr_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rcr;
using f16_rcc_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rcc;
using f16_crr_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::crr;
using f16_crc_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::crc;
using f16_ccr_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::ccr;
using f16_ccc_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::ccc;

using f16_rrr_mem_256x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x64_2x2x1_32x32x16::rrr;
using f16_rrc_mem_256x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x64_2x2x1_32x32x16::rrc;
using f16_rcr_mem_256x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x64_2x2x1_32x32x16::rcr;
using f16_rcc_mem_256x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x64_2x2x1_32x32x16::rcc;
using f16_crr_mem_256x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x64_2x2x1_32x32x16::crr;
using f16_crc_mem_256x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x64_2x2x1_32x32x16::crc;
using f16_ccr_mem_256x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x64_2x2x1_32x32x16::ccr;
using f16_ccc_mem_256x256x64_2x2x1_32x32x16_NonPersistent = typename F16Set<Mem, NonPersistent>::f16_256x256x64_2x2x1_32x32x16::ccc;


// compv3
// 32x32x16
// 2x2x1
using f16_rrr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rrr;
using f16_rrc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rrc;
using f16_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rcr;
using f16_rcc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rcc;
using f16_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::crr;
using f16_crc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::crc;
using f16_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::ccr;
using f16_ccc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::ccc;

// compv3
// 32x32x16
// 2x2x1
using f16_rrr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rrr;
using f16_rrc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rrc;
using f16_rcr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rcr;
using f16_rcc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rcc;
using f16_crr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::crr;
using f16_crc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::crc;
using f16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::ccr;
using f16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::ccc;


// compv4
// 32x32x16
// 2x2x1
using f16_rrr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rrr;
using f16_rrc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rrc;
using f16_rcr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rcr;
using f16_rcc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::rcc;
using f16_crr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::crr;
using f16_crc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::crc;
using f16_ccr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::ccr;
using f16_ccc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_128x128x32_2x2x1_32x32x16::ccc;

// compv4
// 32x32x16
// 2x2x1
using f16_rrr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rrr;
using f16_rrc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rrc;
using f16_rcr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rcr;
using f16_rcc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rcc;
using f16_crr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::crr;
using f16_crc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::crc;
using f16_ccr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::ccr;
using f16_ccc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV4, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::ccc;

//EXPAND_MEMBER(SET_NAME, DATA_TYPE, PIPELINE_TYPE_LOW, MACRO_TILE, WORKGROUP, WAVE_TILE, PIPELINE_TYPE, PERSISTENT) 
EXPAND_MEMBER_LAYOUTS(F16Set, f16, compv3, 256x256x32, 2x2x1, 32x32x16, CompV3, NonPersistent);
// using f16_rrr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rrr;
// using f16_rrc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rrc;
// using f16_rcr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rcr;
// using f16_rcc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::rcc;
// using f16_crr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::crr;
// using f16_crc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::crc;
// using f16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::ccr;
// using f16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename F16Set<CompV3, NonPersistent>::f16_256x256x32_2x2x1_32x32x16::ccc;

// clang-format on
