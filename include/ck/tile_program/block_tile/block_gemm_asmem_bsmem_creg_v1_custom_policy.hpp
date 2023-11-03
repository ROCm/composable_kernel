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
          typename WarpTile_,
          bool TranposeC_>
struct BlockGemmASmemBSmemCRegV1CustomPolicy
{
    using AType = remove_cvref_t<AType_>;
    using BType = remove_cvref_t<BType_>;
    using CType = remove_cvref_t<CType_>;

    using BlockWarps                     = remove_cvref_t<BlockWarps_>;
    using WarpTile                       = remove_cvref_t<WarpTile_>;
    static constexpr index_t BlockMWarps = BlockWarps::At(Number<0>{});
    static constexpr index_t BlockNWarps = BlockWarps::At(Number<1>{});
    static constexpr index_t BlockKWarps = BlockWarps::At(Number<2>{});

    static constexpr index_t MPerWarp = WarpTile::At(Number<0>{});
    static constexpr index_t NPerWarp = WarpTile::At(Number<1>{});
    static constexpr index_t KPerWarp = WarpTile::At(Number<2>{});

    static constexpr bool TranposeC = TranposeC_;

    using WarpGemm = ck::tile_program::warp::
        WarpGemmMfmaDispatcher<AType, BType, CType, MPerWarp, NPerWarp, KPerWarp, TranposeC>;

    template <typename Problem>
    __host__ __device__ static constexpr auto GetWarpGemmMWarpNWarp()
    {
        using namespace ck::tile_program::warp;
        return make_tuple(WarpGemm{}, BlockMWarps, BlockNWarps);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
