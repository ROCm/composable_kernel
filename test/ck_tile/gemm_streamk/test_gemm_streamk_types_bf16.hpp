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
struct BF16Layouts
{
    // clang-format off
    // For CDNA, we support [A, B, Acc, C] = [bf16, bf16, f32, bf16] and [bf16, bf16, f32, f32]:
    using bf16_bf16_f32_bf16 = Layouts<BF16, BF16, F32, BF16, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>;
    using bf16_bf16_f32_f32  = Layouts<BF16, BF16, F32,  F32, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>;
    using rrr = detail::combine_t<typename bf16_bf16_f32_bf16::rrr, typename bf16_bf16_f32_f32::rrr>;
    using rrc = detail::combine_t<typename bf16_bf16_f32_bf16::rrc, typename bf16_bf16_f32_f32::rrc>;
    using rcr = detail::combine_t<typename bf16_bf16_f32_bf16::rcr, typename bf16_bf16_f32_f32::rcr>;
    using rcc = detail::combine_t<typename bf16_bf16_f32_bf16::rcc, typename bf16_bf16_f32_f32::rcc>;
    using crr = detail::combine_t<typename bf16_bf16_f32_bf16::crr, typename bf16_bf16_f32_f32::crr>;
    using crc = detail::combine_t<typename bf16_bf16_f32_bf16::crc, typename bf16_bf16_f32_f32::crc>;
    using ccr = detail::combine_t<typename bf16_bf16_f32_bf16::ccr, typename bf16_bf16_f32_f32::ccr>;
    using ccc = detail::combine_t<typename bf16_bf16_f32_bf16::ccc, typename bf16_bf16_f32_f32::ccc>;
    // clang-format on
};

template<typename PipelineType,
         typename Persistent>
struct BF16Set
{
    // clang-format off
    // 32x32x16
    // 2x2x1
    using bf16_128x128x32_2x2x1_32x32x16 =  BF16Layouts<I128, I128,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x128x64_2x2x1_32x32x16 =  BF16Layouts<I128, I128,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x128x128_2x2x1_32x32x16 = BF16Layouts<I128, I128, I128, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x128x32_2x2x1_32x32x16 =  BF16Layouts<I256, I128,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x128x64_2x2x1_32x32x16 =  BF16Layouts<I256, I128,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x256x32_2x2x1_32x32x16 =  BF16Layouts<I128, I256,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x256x64_2x2x1_32x32x16 =  BF16Layouts<I128, I256,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x256x32_2x2x1_32x32x16 =  BF16Layouts<I256, I256,  I32, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x256x64_2x2x1_32x32x16 =  BF16Layouts<I256, I256,  I64, I2, I2, I1, I32, I32, I16, PipelineType, Persistent>;

    // 32x32x16
    // 4x1x1
    using bf16_128x128x32_4x1x1_32x32x16 =  BF16Layouts<I128, I128,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x128x64_4x1x1_32x32x16 =  BF16Layouts<I128, I128,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x128x128_4x1x1_32x32x16 = BF16Layouts<I128, I128, I128, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x128x32_4x1x1_32x32x16 =  BF16Layouts<I256, I128,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x128x64_4x1x1_32x32x16 =  BF16Layouts<I256, I128,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x256x32_4x1x1_32x32x16 =  BF16Layouts<I128, I256,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x256x64_4x1x1_32x32x16 =  BF16Layouts<I128, I256,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x256x32_4x1x1_32x32x16 =  BF16Layouts<I256, I256,  I32, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x256x64_4x1x1_32x32x16 =  BF16Layouts<I256, I256,  I64, I4, I1, I1, I32, I32, I16, PipelineType, Persistent>;

    // 32x32x16
    // 1x4x1
    using bf16_128x128x32_1x4x1_32x32x16 =  BF16Layouts<I128, I128,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x128x64_1x4x1_32x32x16 =  BF16Layouts<I128, I128,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x128x128_1x4x1_32x32x16 = BF16Layouts<I128, I128, I128, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x128x32_1x4x1_32x32x16 =  BF16Layouts<I256, I128,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x128x64_1x4x1_32x32x16 =  BF16Layouts<I256, I128,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x256x32_1x4x1_32x32x16 =  BF16Layouts<I128, I256,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_128x256x64_1x4x1_32x32x16 =  BF16Layouts<I128, I256,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x256x32_1x4x1_32x32x16 =  BF16Layouts<I256, I256,  I32, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;
    using bf16_256x256x64_1x4x1_32x32x16 =  BF16Layouts<I256, I256,  I64, I1, I4, I1, I32, I32, I16, PipelineType, Persistent>;

    // 16x16x16
    // 2x2x1
    using bf16_128x128x32_2x2x1_16x16x16 =  BF16Layouts<I128, I128,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x128x64_2x2x1_16x16x16 =  BF16Layouts<I128, I128,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x128x128_2x2x1_16x16x16 = BF16Layouts<I128, I128, I128, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x128x32_2x2x1_16x16x16 =  BF16Layouts<I256, I128,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x128x64_2x2x1_16x16x16 =  BF16Layouts<I256, I128,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x256x32_2x2x1_16x16x16 =  BF16Layouts<I128, I256,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x256x64_2x2x1_16x16x16 =  BF16Layouts<I128, I256,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x256x32_2x2x1_16x16x16 =  BF16Layouts<I256, I256,  I32, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x256x64_2x2x1_16x16x16 =  BF16Layouts<I256, I256,  I64, I2, I2, I1, I16, I16, I16, PipelineType, Persistent>;

    // 16x16x16
    // 4x1x1
    using bf16_128x128x32_4x1x1_16x16x16 =  BF16Layouts<I128, I128,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x128x64_4x1x1_16x16x16 =  BF16Layouts<I128, I128,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x128x128_4x1x1_16x16x16 = BF16Layouts<I128, I128, I128, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x128x32_4x1x1_16x16x16 =  BF16Layouts<I256, I128,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x128x64_4x1x1_16x16x16 =  BF16Layouts<I256, I128,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x256x32_4x1x1_16x16x16 =  BF16Layouts<I128, I256,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x256x64_4x1x1_16x16x16 =  BF16Layouts<I128, I256,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x256x32_4x1x1_16x16x16 =  BF16Layouts<I256, I256,  I32, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x256x64_4x1x1_16x16x16 =  BF16Layouts<I256, I256,  I64, I4, I1, I1, I16, I16, I16, PipelineType, Persistent>;

    // 16x16x16
    // 1x4x1
    using bf16_128x128x32_1x4x1_16x16x16 =  BF16Layouts<I128, I128,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x128x64_1x4x1_16x16x16 =  BF16Layouts<I128, I128,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x128x128_1x4x1_16x16x16 = BF16Layouts<I128, I128, I128, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x128x32_1x4x1_16x16x16 =  BF16Layouts<I256, I128,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x128x64_1x4x1_16x16x16 =  BF16Layouts<I256, I128,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x256x32_1x4x1_16x16x16 =  BF16Layouts<I128, I256,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_128x256x64_1x4x1_16x16x16 =  BF16Layouts<I128, I256,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x256x32_1x4x1_16x16x16 =  BF16Layouts<I256, I256,  I32, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    using bf16_256x256x64_1x4x1_16x16x16 =  BF16Layouts<I256, I256,  I64, I1, I4, I1, I16, I16, I16, PipelineType, Persistent>;
    // clang-format on
};

// clang-format off
// mem
// 32x32x16
// 2x2x1
using bf16_rrr_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rrr;
using bf16_rrc_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rrc;
using bf16_rcr_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rcr;
using bf16_rcc_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rcc;
using bf16_crr_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::crr;
using bf16_crc_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::crc;
using bf16_ccr_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::ccr;
using bf16_ccc_mem_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::ccc;

using bf16_rrr_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rrr;
using bf16_rrc_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rrc;
using bf16_rcr_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rcr;
using bf16_rcc_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rcc;
using bf16_crr_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::crr;
using bf16_crc_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::crc;
using bf16_ccr_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::ccr;
using bf16_ccc_mem_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<Mem, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::ccc;


// compv3
// 32x32x16
// 2x2x1
using bf16_rrr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rrr;
using bf16_rrc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rrc;
using bf16_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rcr;
using bf16_rcc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rcc;
using bf16_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::crr;
using bf16_crc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::crc;
using bf16_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::ccr;
using bf16_ccc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::ccc;

using bf16_rrr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rrr;
using bf16_rrc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rrc;
using bf16_rcr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rcr;
using bf16_rcc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rcc;
using bf16_crr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::crr;
using bf16_crc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::crc;
using bf16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::ccr;
using bf16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV3, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::ccc;


// compv4
// 32x32x16
// 2x2x1
using bf16_rrr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rrr;
using bf16_rrc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rrc;
using bf16_rcr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rcr;
using bf16_rcc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::rcc;
using bf16_crr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::crr;
using bf16_crc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::crc;
using bf16_ccr_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::ccr;
using bf16_ccc_compv4_128x128x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_128x128x32_2x2x1_32x32x16::ccc;

using bf16_rrr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rrr;
using bf16_rrc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rrc;
using bf16_rcr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rcr;
using bf16_rcc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::rcc;
using bf16_crr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::crr;
using bf16_crc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::crc;
using bf16_ccr_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::ccr;
using bf16_ccc_compv4_256x256x32_2x2x1_32x32x16_NonPersistent = typename BF16Set<CompV4, NonPersistent>::bf16_256x256x32_2x2x1_32x32x16::ccc;

// clang-format on
