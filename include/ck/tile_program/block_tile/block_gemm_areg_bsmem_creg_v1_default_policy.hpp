// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Default policy for BlockGemmARegBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmARegBSmemCRegV1DefaultPolicy
{
    template <typename Problem>
    __host__ __device__ static constexpr auto GetWarpGemmMWarpNWarp()
    {
        using namespace ck::tile_program::warp;

        if constexpr(is_same_v<typename Problem::ADataType, half_t> &&
                     is_same_v<typename Problem::BDataType, half_t> &&
                     is_same_v<typename Problem::CDataType, float>)
        {
#if 0
            constexpr index_t kBlockSize = Problem::kBlockSize;

            constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

            static_assert(kBlockSize % get_warp_size() == 0, "wrong!");

            constexpr index_t NumWarp = kBlockSize / get_warp_size();

            // FIXME
            if constexpr(NumWarp == 4 && kMPerBlock % 128 == 0 &&
                         kNPerBlock % 128 == 0 % kKPerBlock % 16 == 0)
            {
                return make_tuple(WarpGemmMfmaF16F16F32M32N32K8{}, 4, 1);
            }
            else
            {
                return make_tuple(WarpGemmMfmaF16F16F32M32N32K8{}, 4, 1);
            }
#else
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{}, 4, 1);
#endif
        }
        else if constexpr(is_same_v<typename Problem::ADataType, bhalf_t> &&
                          is_same_v<typename Problem::BDataType, bhalf_t> &&
                          is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution{}, 4, 1);
        }
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
