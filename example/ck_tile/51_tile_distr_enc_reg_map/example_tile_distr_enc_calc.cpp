// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <type_traits>
#include <tuple>
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/core/container/tuple.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace mma;
using I4        = pk_int4_t;
using I32       = int32_t;
using F16       = fp16_t;
using F32       = fp32_t;
using Target908 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX908>());
using Target950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());
using Target11  = decltype(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>());
using Target12  = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());

template <typename MmaOp>
int check_tile_distr_enc()
{
    using AEnc = typename TileDistrEncCalc<MmaOp>::AWarpDstrEncoding;
    using BEnc = typename TileDistrEncCalc<MmaOp>::BWarpDstrEncoding;
    using CEnc = typename TileDistrEncCalc<MmaOp>::CWarpDstrEncoding;

    TileDistrEncRegMap<AEnc>::print();
    TileDistrEncRegMap<BEnc>::print();
    TileDistrEncRegMap<CEnc>::print();

    // The only thing we check here is that CTranspose works as expected.
    using AEncTransp = typename TileDistrEncCalc<MmaOp, true>::AWarpDstrEncoding;
    using BEncTransp = typename TileDistrEncCalc<MmaOp, true>::BWarpDstrEncoding;
    using CEncTransp = typename TileDistrEncCalc<MmaOp, true>::CWarpDstrEncoding;

    // When using TransposeC, the A and B matrix layouts should be swapped.
    static_assert(std::is_same<AEncTransp, BEnc>());
    static_assert(std::is_same<BEncTransp, AEnc>());

    // Make sure the C matrix layout is transposed in the CTranspose case.
    int err = 0;
    for(index_t lane = 0; lane < TileDistrEncRegMap<CEnc>::num_lanes; lane++)
    {
        for(index_t vec = 0; vec < TileDistrEncRegMap<CEnc>::num_vector_items; vec++)
        {
            auto coords = TileDistrEncRegMap<CEnc>::calc_matrix_indices_from_lane_vector(lane, vec);
            auto coords_transp =
                TileDistrEncRegMap<CEncTransp>::calc_matrix_indices_from_lane_vector(lane, vec);

            if(coords[0] != coords_transp[1] || coords[1] != coords_transp[0])
            {
                err = 1;
                printf("\033[31mLane %2d vec %2d maps to C matrix coords %2d %2d and transposed C "
                       "matrix coords %2d %2d, inconsistent!\033[0m\n",
                       lane,
                       vec,
                       coords[0],
                       coords[1],
                       coords_transp[0],
                       coords_transp[1]);
            }
        }
    }

    return err;
}

// List of intrinsics to test.
// clang-format off
using Intrinsics = ck_tile::tuple<
    amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_16x16x16f16
    amdgcn_mma<F16, F16, F32, 64u, 32u, 4u,  DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_32x32x4f16
    amdgcn_mma<F16, F16, F32, 32u, 64u, 4u,  DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_32x32x4f16
    amdgcn_mma<F16, F16, F32, 64u, 4u,  4u,  DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_4x4x4f16
    amdgcn_mma<F16, F16, F32, 4u,  64u, 4u,  DefaultMfmaCtrlFlags,                Target908, MmaOpFamily::DENSE>, // mfma_f32_4x4x4f16
    amdgcn_mma<F16, F16, F32, 16u, 16u, 32u, DefaultMfmaCtrlFlags,                Target950, MmaOpFamily::DENSE>, // mfma_f32_16x16x32_f16
    amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,                Target11,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_f16_w32
    amdgcn_mma<I4,  I4,  I32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,                Target11,  MmaOpFamily::DENSE>, // wmma_i32_16x16x16_iu4_w32
    amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,                Target12,  MmaOpFamily::DENSE>, // wmma_f32_16x16x16_f16_w32_gfx12
    amdgcn_mma<I4,  I4,  I32, 16u, 16u, 16u, DefaultWmmaCtrlFlags,                Target12,  MmaOpFamily::DENSE>, // wmma_i32_16x16x16_iu4_w32_gfx12
    amdgcn_mma<I4,  I4,  I32, 16u, 16u, 32u, DefaultWmmaCtrlFlags,                Target12,  MmaOpFamily::DENSE>  // wmma_i32_16x16x32_iu4_w32_gfx12
>;
// clang-format on

int main()
{
    int err = 0;
    static_for<0, Intrinsics::size(), 1>{}([&](auto i) {
        using MmaOp = std::tuple_element_t<i.value, Intrinsics>;
        err |= check_tile_distr_enc<MmaOp>();
    });
    if(err)
        printf("\033[031mError in tile distribution enc calculator!\033[0m\n");
    return err;
}
