// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "test_gemm_streamk_types.hpp"

template <typename M_MacroTile,
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
    using BF16_BF16_F32_BF16 = Layouts<BF16, BF16, F32, BF16, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>;
    using BF16_BF16_F32_F32 = Layouts<BF16, BF16, F32, F32, M_MacroTile, N_MacroTile, K_MacroTile, M_Warps, N_Warps, K_Warps, M_MmaTile, N_MmaTile, K_MmaTile, PipelineType, Persistent>;
    using RRR = detail::combine_t<typename BF16_BF16_F32_BF16::RRR, typename BF16_BF16_F32_F32::RRR>;
    using RRC = detail::combine_t<typename BF16_BF16_F32_BF16::RRC, typename BF16_BF16_F32_F32::RRC>;
    using RCR = detail::combine_t<typename BF16_BF16_F32_BF16::RCR, typename BF16_BF16_F32_F32::RCR>;
    using RCC = detail::combine_t<typename BF16_BF16_F32_BF16::RCC, typename BF16_BF16_F32_F32::RCC>;
    using CRR = detail::combine_t<typename BF16_BF16_F32_BF16::CRR, typename BF16_BF16_F32_F32::CRR>;
    using CRC = detail::combine_t<typename BF16_BF16_F32_BF16::CRC, typename BF16_BF16_F32_F32::CRC>;
    using CCR = detail::combine_t<typename BF16_BF16_F32_BF16::CCR, typename BF16_BF16_F32_F32::CCR>;
    using CCC = detail::combine_t<typename BF16_BF16_F32_BF16::CCC, typename BF16_BF16_F32_F32::CCC>;
    // clang-format on
};
// clang-format off

// Macro to declare all layout combinations for BF16 data type
#define DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT) \
    DECLARE_PARAMS_ALL_LAYOUTS(BF16Layouts, BF16, PIPELINE_TYPE, M_MACRO_TILE, N_MACRO_TILE, K_MACRO_TILE, M_WARPS, N_WARPS, K_WARPS, M_MMA_TILE, N_MMA_TILE, K_MMA_TILE, PERSISTENT)

// Macro to declare all layout combinations for BF16 data type and a variety of sizes
#define DECLARE_BF16_PARAMS_ALL_LAYOUTS_ALL_SIZES(PIPELINE_TYPE, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 128, 128, 32, 2, 2, 1, 32, 32, 16, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 128, 128, 64, 2, 2, 1, 32, 32, 16, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 128, 128, 128, 2, 2, 1, 32, 32, 16, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 256, 128, 32, 2, 2, 1, 32, 32, 16, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 256, 128, 64, 2, 2, 1, 32, 32, 16, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 128, 256, 32, 2, 2, 1, 32, 32, 16, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 128, 256, 64, 2, 2, 1, 32, 32, 16, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 256, 256, 32, 2, 2, 1, 32, 32, 16, PERSISTENT) \
    DECLARE_BF16_PARAMS_ALL_LAYOUTS(PIPELINE_TYPE, 256, 256, 64, 2, 2, 1, 32, 32, 16, PERSISTENT) 

// Declare all BF16 parameter sets for different pipeline types and persistence options
DECLARE_BF16_PARAMS_ALL_LAYOUTS_ALL_SIZES(Mem, NonPersistent)
DECLARE_BF16_PARAMS_ALL_LAYOUTS_ALL_SIZES(CompV3, NonPersistent)
DECLARE_BF16_PARAMS_ALL_LAYOUTS_ALL_SIZES(CompV4, NonPersistent)

// Here, we have a combination of parameter set symbols that we can use to compile into test cases
//        __________________________________________________  
//       |                Parameter Name                    |  
// using BF16_RRR_Mem_128x128x32_2x2x1_32x32x16_NonPersistent = ...
//        /   |     \         \      \      \       \
//     DATA LAYOUT  PIPELINE  MACRO  WARPS   MMA    PERSISTENT 
//     TYPE         TYPE      TILE   MxNxK   TILE   TYPE       
//                            MxNxK          MxNxK             
// 
// The options for each field are:
//  - DATA TYPE: BF16
//  - LAYOUT: RRR, RRC, RCR, RCC, CRR, CRC, CCR, CCC
//  - PIPELINE_TYPE: Mem, CompV3, CompV4
//  - Macro Tile: 128x128x32, 128x128x64, 128x128x128, 256x128x32, 256x128x64, 128x256x32, 128x256x64, 256x256x32, 256x256x64
//  - Warps: 2x2x1
//  - MMA Tile: 32x32x16
//  - PERSISTENT_TYPE: NonPersistent

// clang-format on
