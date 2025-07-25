// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmASmemBSmemCRegV1DefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
#if defined(__gfx950__)
        constexpr bool is_a_load_tr = std::is_same_v<remove_cvref_t<typename Problem::ALayout>,
                                                     tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_load_tr = std::is_same_v<remove_cvref_t<typename Problem::BLayout>,
                                                     tensor_layout::gemm::RowMajor>;
#else
        constexpr bool is_a_load_tr = false;
        constexpr bool is_b_load_tr = false;
#endif
        constexpr auto wg_attr_num_access = (is_a_load_tr || is_b_load_tr)
                                                ? WGAttrNumAccessEnum::Double
                                                : WGAttrNumAccessEnum::Single;

        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
#if 0
            constexpr index_t kBlockSize = Problem::kBlockSize;

            constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

            static_assert(kBlockSize % get_warp_size() == 0, "wrong!");

            constexpr index_t NumWarp = kBlockSize / get_warp_size();

            if constexpr(NumWarp == 4 && kMPerBlock % 128 == 0 &&
                         kNPerBlock % 128 == 0 % kKPerBlock % 16 == 0)
            {
                return make_tuple(WarpGemmMfmaF16F16F32M32N32K16<>{}, 2, 2);
            }
            else
            {
                return make_tuple(WarpGemmMfmaF16F16F32M32N32K16<>{}, 2, 2);
            }
#else
            using WG = WarpGemmMfmaDispatcher<ck_tile::half_t,
                                              ck_tile::half_t,
                                              float,
                                              32,
                                              32,
                                              16,
                                              true,
                                              false,
                                              false,
                                              wg_attr_num_access>;
            return make_tuple(WG{}, 4, 1);
#endif
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            using WG = WarpGemmMfmaDispatcher<ck_tile::bf16_t,
                                              ck_tile::bf16_t,
                                              float,
                                              32,
                                              32,
                                              16,
                                              true,
                                              false,
                                              false,
                                              wg_attr_num_access>;
            return make_tuple(WG{}, 4, 1);
        }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
