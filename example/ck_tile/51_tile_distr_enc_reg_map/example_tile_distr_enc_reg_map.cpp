// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch::mma;

int main()
{
    // Define some tile distribution encodings and print register mappings.

    printf("Example RDNA3 V_WMMA_F32_16X16X16_F16 A Matrix (M, K)\nL{RM} V{K}\n");
    TileDistrEncRegMap<
        tile_distribution_encoding<sequence<2>, // R (= Repeat) Lanes 0-15 are duplicated at 16-31
                                   tuple<sequence<16>, sequence<16>>, // H (= Hidden dims = unmerged
                                                                      // dims) for M, K dimension
                                   tuple<sequence<0, 1>>, // P major (= Parallelism = lanes)
                                   tuple<sequence<0, 0>>, // P minor
                                   sequence<2>,           // Y major (= Yield = Vector items)
                                   sequence<0>            // Y minor
                                   >>::print();

    printf("\nExample RDNA3 V_WMMA_F32_16X16X16_F16 C Matrix (M, N)\nM{2, 1} L{M1N} V{M2M0} (dummy "
           "unmerge to be more similar to other layouts)\n");
    TileDistrEncRegMap<
        tile_distribution_encoding<sequence<>,                             // R (= Repeat)
                                   tuple<sequence<8, 2, 1>, sequence<16>>, // H (= Hidden dims =
                                                                           // unmerged dims) for M,
                                                                           // N dimension
                                   tuple<sequence<1, 2>>, // P major (= Parallelism = lanes)
                                   tuple<sequence<1, 0>>, // P minor
                                   sequence<1, 1>,        // Y major (= Yield = Vector items)
                                   sequence<0, 2>         // Y minor
                                   >>::print();

    printf("\nExample CDNA __builtin_amdgcn_mfma_f32_4x4x4f16 A Matrix (M, K) with 16x "
           "block-hiding in the M dimension\nL{BM} V{K}\n");
    TileDistrEncRegMap<
        tile_distribution_encoding<sequence<>,                          // R (= Repeat)
                                   tuple<sequence<16, 4>, sequence<4>>, // H (= Hidden dims =
                                                                        // unmerged dims) for M,
                                                                        // K dimension
                                   tuple<sequence<1, 1>>, // P major (= Parallelism = lanes)
                                   tuple<sequence<0, 1>>, // P minor
                                   sequence<2>,           // Y major (= Yield = Vector items)
                                   sequence<0>            // Y minor
                                   >>::print();

    printf("\nExample CDNA __builtin_amdgcn_mfma_f32_4x4x4f16 B Matrix (N, K) with 16x "
           "block-hiding in the M dimension\nL{BN} V{K}\n");
    TileDistrEncRegMap<
        tile_distribution_encoding<sequence<16>,                    // R (= Repeat)
                                   tuple<sequence<4>, sequence<4>>, // H (= Hidden dims =
                                                                    // unmerged dims) for N,
                                                                    // K dimension
                                   tuple<sequence<0, 1>>, // P major (= Parallelism = lanes)
                                   tuple<sequence<0, 0>>, // P minor
                                   sequence<2>,           // Y major (= Yield = Vector items)
                                   sequence<0>            // Y minor
                                   >>::print();

    printf("\nCustom example\n");
    TileDistrEncRegMap<
        tile_distribution_encoding<sequence<1>,                            // R (= Repeat)
                                   tuple<sequence<16>, sequence<1, 2, 8>>, // H (= Hidden dims =
                                                                           // unmerged dims)
                                   tuple<sequence<2, 1>>, // P major (= Parallelism = lanes)
                                   tuple<sequence<1, 0>>, // P minor
                                   sequence<2, 2>,        // Y major (= Yield = Vector items)
                                   sequence<0, 2>         // Y minor
                                   >>::print();

    return 0;
}
