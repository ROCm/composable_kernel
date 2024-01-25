// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm_dispatcher.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Default policy for BlockGemmASmemBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
template <typename AType_,
          typename BType_,
          typename CType_,
          typename BlockWarps_,
          typename WarpGemm_>
struct BlockGemmASmemBSmemCRegV1CustomPolicy
{
    using AType = remove_cvref_t<AType_>;
    using BType = remove_cvref_t<BType_>;
    using CType = remove_cvref_t<CType_>;

    using BlockWarps = remove_cvref_t<BlockWarps_>;

    static constexpr index_t kMWarps = BlockWarps::At(Number<0>{});
    static constexpr index_t kNWarps = BlockWarps::At(Number<1>{});
    static constexpr index_t kKWarps = BlockWarps::At(Number<2>{});

    using WarpGemm = remove_cvref_t<WarpGemm_>;

    template <typename Problem>
    __host__ __device__ static constexpr auto GetWarpGemmMWarpNWarp()
    {
        using namespace ck::tile_program::warp;
        return make_tuple(WarpGemm{}, kMWarps, kNWarps);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
