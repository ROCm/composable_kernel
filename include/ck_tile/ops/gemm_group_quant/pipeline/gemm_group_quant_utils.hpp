// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

  template <typename Problem, typename DataType, index_t YPerTile, index_t XPerTile>
  CK_TILE_HOST_DEVICE static constexpr auto GetABQGlobalVectorLoadSize()
  {
      using I0 = number<0>;
      constexpr index_t MWarps = Problem::BlockGemmShape::BlockWarps::at(I0{});

      constexpr index_t BlockSize           = Problem::kBlockSize;

      // Data is replicated across warps along NWarps, so we divide BlockSize by MWarps
      constexpr index_t elements_per_thread = (YPerTile * XPerTile) / (BlockSize / MWarps);
      constexpr index_t PackedSize =
          ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

      // Assume DataType is even!
      if constexpr(XPerTile % (PackedSize * 32 / sizeof(DataType)) == 0 &&
                   elements_per_thread % (PackedSize * 32 / sizeof(DataType)) == 0 &&
                   PackedSize == 2)
      {
          return (PackedSize * 32 / sizeof(DataType));
      }
      else if constexpr(XPerTile % (PackedSize * 16 / sizeof(DataType)) == 0 &&
                        elements_per_thread % (PackedSize * 16 / sizeof(DataType)) == 0)
      {
          return (PackedSize * 16 / sizeof(DataType));
      }
      else if constexpr(XPerTile % (PackedSize * 8 / sizeof(DataType)) == 0 &&
                        elements_per_thread % (PackedSize * 8 / sizeof(DataType)) == 0)
      {
          return (PackedSize * 8 / sizeof(DataType));
      }
      else if constexpr(sizeof(DataType) >= PackedSize * 4 &&
                        XPerTile % (PackedSize * 4 / sizeof(DataType)) == 0 &&
                        elements_per_thread % (PackedSize * 4 / sizeof(DataType)) == 0)
      {
          return (PackedSize * 4 / sizeof(DataType));
      }
      else if constexpr(sizeof(DataType) >= PackedSize * 2 &&
                        XPerTile % (PackedSize * 2 / sizeof(DataType)) == 0 &&
                        elements_per_thread % (PackedSize * 2 / sizeof(DataType)) == 0)
      {
          return (PackedSize * 2 / sizeof(DataType));
      }
      else
      {
          return PackedSize;
      }
  }
  // template <typename ADataType_, typename BDataType_, typename QDataType>
  // std::string gemm_prec_str()
  // {
  //     std::string base_str = std::string(typeToStr<ADataType_>::name);
  //     if(!std::is_same_v<ADataType_, BDataType_>)
  //     {
  //         base_str += "_" + std::string(typeToStr<BDataType_>::name);
  //     }
  //     return base_str;
  // }
}
