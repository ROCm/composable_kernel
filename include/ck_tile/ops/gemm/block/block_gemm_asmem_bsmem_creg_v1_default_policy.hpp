// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmASmemBSmemCRegV1DefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
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
                return make_tuple(WarpGemmMfmaF16F16F32M32N32K16{}, 2, 2);
            }
            else
            {
                return make_tuple(WarpGemmMfmaF16F16F32M32N32K16{}, 2, 2);
            }
#else
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution{}, 2, 2);
            // return make_tuple(WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution{}, 4, 1);
#endif
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution{}, 4, 1);
        }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
    
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALDSTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            static_assert(false, "Unsupported tensor_layout right now.");
        }
        else
        {
            //Number<krepeat>{}, Number<klane>{}, Number<Kpack>{}))), 
            constexpr index_t K2 = 16 / sizeof(ADataType);
            constexpr index_t K1 = 2;
            constexpr index_t K0 = KPerBlock / K1 / K2;
            //Number<mrepeat>{}, Number<mwaves>{}, Number<MPerXdl>{}))),
            constexpr index_t M2 = 32;  // MPERXDL
            constexpr index_t M1 = 2; //MWAVE
            // coalesce reading for each blocks
            if constexpr(get_warp_size() % (M2 * K0) == 0)
            {
                static_assert(M2 != 0, "M2 is zero, which will lead to a division by zero error.");
                static_assert(M1 != 0, "M1 is zero, which will lead to a division by zero error.");
                constexpr index_t M0 = MPerBlock / (M2 * M1);

                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<2>,
                                                tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
                                                tuple<sequence<1, 0>, sequence<2, 1>>,
                                                tuple<sequence<1, 0>, sequence<1, 2>>,
                                                sequence<1, 2, 2>,
                                                sequence<0, 0, 2>>{});
            }
            else
            {
                static_assert(false, "Unsupported shape right now.");
            }
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLDSTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using BLayout   = remove_cvref_t<typename Problem::BLayout>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            static_assert(false, "Unsupported tensor_layout right now.");
        }
        else
        {
            //Number<krepeat>{}, Number<klane>{}, Number<Kpack>{}))), 
            constexpr index_t K2 = 16 / sizeof(BDataType);
            constexpr index_t K1 = 2;
            constexpr index_t K0 = KPerBlock / K1 / K2;
            //Number<mrepeat>{}, Number<mwaves>{}, Number<MPerXdl>{}))),
            constexpr index_t N2 = 32;  // MPERXDL
            constexpr index_t N1 = 2; //MWAVE
            // coalesce reading for each blocks
            if constexpr(get_warp_size() % (N2 * K0) == 0)
            {
                static_assert(N2 != 0, "M2 is zero, which will lead to a division by zero error.");
                static_assert(N1 != 0, "M1 is zero, which will lead to a division by zero error.");
                constexpr index_t N0 = NPerBlock / (N2 * N1);

                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<2>,
                                                tuple<sequence<N0, N1, N2>, sequence<K0, K1, K2>>,
                                                tuple<sequence<0, 1>, sequence<2, 1>>,
                                                tuple<sequence<0, 1>, sequence<1, 2>>,
                                                sequence<1, 2, 2>,
                                                sequence<0, 0, 2>>{});
            }
            else
            {
                static_assert(false, "Unsupported shape right now.");
            }
        }
    }

};


} // namespace ck_tile
