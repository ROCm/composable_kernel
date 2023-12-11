// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "statically_indexed_array.hpp"
#include "data_type.hpp"

namespace ck {

#if 0 // debug
// S: scalar type
// NX: # of vector before transpose
// NY: # of vector after transpose
template <typename S,
          index_t NX,
          index_t NY,
          typename enable_if<is_scalar_type<S>::value, bool>::type = false>
struct transpose_vectors;

// transpose fp16 2x2
__device__ void transpose_fp16_2x2(const half2_t& x0, const half2_t& x1, half2_t& y0, half2_t& y1)
{
    constexpr int32_t m0 = 0x05040100;
    constexpr int32_t m1 = 0x07060302;

    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
    //                   -- -- -- --     -- -- -- --      -  -  -  -
    //             index  7  6  5  4      3  2  1  0     33 77 44 88
    // index is reversed because of little endianness (least significant bits first)
    y0 = bit_cast<half2_t>(__builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m0));
    y1 = bit_cast<half2_t>(__builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m1));
}

template <index_t NX, index_t NY>
struct transpose_vectors<half_t, NX, NY>
{
    // we got [NY * NX] amount of S data to be transposed
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S  = half_t;
    using VX = vector_type<half_t, s_per_x>;
    using VY = vector_type<half_t, s_per_y>;

    __device__ void operator()(const StaticallyIndexedArray<const VX&, NX>& vx_tuple,
                               StaticallyIndexedArray<VY&, NY>& vy_tuple)
    {
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};

        static_assert((NX % 2 == 0 && NY % 2 == 0), "wrong!");

        // loop over 2x2 tile and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 2>{}([&](auto iy) {
            static_for<0, NX, 2>{}([&](auto ix) {
                // reference to 2 half2_t data from vx_tuple
                const auto& x_s2_0 = vx_tuple[ix].template AsType<half2_t>()[iy / I2];
                const auto& x_s2_1 = vx_tuple[ix + I1].template AsType<half2_t>()[iy / I2];

                // reference to 2 half2_t data from vy_tuple
                auto& y_s2_0 = vy_tuple(iy).template AsType<half2_t>()(ix / I2);
                auto& y_s2_1 = vy_tuple(iy + I1).template AsType<half2_t>()(ix / I2);

                // transpose
                transpose_fp16_2x2(x_s2_0, x_s2_1, y_s2_0, y_s2_1);
            });
        });
    }

    // FIXME: duplicated code
    __device__ void operator()(const StaticallyIndexedArray<VX, NX>& vx_tuple,
                               StaticallyIndexedArray<VY, NY>& vy_tuple)
    {
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};

        static_assert((NX % 2 == 0 && NY % 2 == 0), "wrong!");

        // loop over 2x2 tile and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 2>{}([&](auto iy) {
            static_for<0, NX, 2>{}([&](auto ix) {
                // reference to 2 half2_t data from vx_tuple
                const auto& x_s2_0 = vx_tuple[ix].template AsType<half2_t>()[iy / I2];
                const auto& x_s2_1 = vx_tuple[ix + I1].template AsType<half2_t>()[iy / I2];

                // reference to 2 half2_t data from vy_tuple
                auto& y_s2_0 = vy_tuple(iy).template AsType<half2_t>()(ix / I2);
                auto& y_s2_1 = vy_tuple(iy + I1).template AsType<half2_t>()(ix / I2);

                // transpose
                transpose_fp16_2x2(x_s2_0, x_s2_1, y_s2_0, y_s2_1);
            });
        });
    }
};

// transpose int8 4x4
__device__ void transpose_int8_4x4(const int8x4_t& x0,
                                   const int8x4_t& x1,
                                   const int8x4_t& x2,
                                   const int8x4_t& x3,
                                   int8x4_t& y0,
                                   int8x4_t& y1,
                                   int8x4_t& y2,
                                   int8x4_t& y3)
{
    int32_t t0, t1;
    int32_t z0, z1, z2, z3;
    constexpr int32_t m0 = 0x05010400;
    constexpr int32_t m1 = 0x05040100;
    constexpr int32_t m2 = 0x07060302;
    constexpr int32_t m3 = 0x07030602;

    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
    //                   -- -- -- --     -- -- -- --      -  -  -  -
    //             index  7  6  5  4      3  2  1  0     33 77 44 88
    // index is reversed because of little endianness (least significant bits first)
    t0 = __builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m0);
    t1 = __builtin_amdgcn_perm(bit_cast<int32_t>(x3), bit_cast<int32_t>(x2), m0);
    z0 = __builtin_amdgcn_perm(bit_cast<int32_t>(t1), bit_cast<int32_t>(t0), m1);
    z1 = __builtin_amdgcn_perm(bit_cast<int32_t>(t1), bit_cast<int32_t>(t0), m2);
    t0 = __builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m3);
    t1 = __builtin_amdgcn_perm(bit_cast<int32_t>(x3), bit_cast<int32_t>(x2), m3);
    z2 = __builtin_amdgcn_perm(bit_cast<int32_t>(t1), bit_cast<int32_t>(t0), m1);
    z3 = __builtin_amdgcn_perm(bit_cast<int32_t>(t1), bit_cast<int32_t>(t0), m2);

    y0 = bit_cast<int8x4_t>(z0);
    y1 = bit_cast<int8x4_t>(z1);
    y2 = bit_cast<int8x4_t>(z2);
    y3 = bit_cast<int8x4_t>(z3);
}


template <index_t NX, index_t NY>
struct transpose_vectors<int8_t, NX, NY>
{
    // we got [NY * NX] amount of S data to be transposed
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S  = int8_t;
    using VX = vector_type<int8_t, s_per_x>;
    using VY = vector_type<int8_t, s_per_y>;

    __device__ void operator()(const StaticallyIndexedArray<const VX&, NX>& vx_tuple,
                               StaticallyIndexedArray<VY&, NY>& vy_tuple)
    {
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};
        static constexpr auto I3 = Number<3>{};
        static constexpr auto I4 = Number<4>{};

        static_assert((NX % 4 == 0 && NY % 4 == 0), "wrong!");

        // loop over 4x4 tile and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 4>{}([&](auto iy) {
            static_for<0, NX, 4>{}([&](auto ix) {
                // reference to 4 int8 data from vx_tuple
                const auto& x_s4_0 = vx_tuple[ix].template AsType<int8x4_t>()[iy / I4];
                const auto& x_s4_1 = vx_tuple[ix + I1].template AsType<int8x4_t>()[iy / I4];
                const auto& x_s4_2 = vx_tuple[ix + I2].template AsType<int8x4_t>()[iy / I4];
                const auto& x_s4_3 = vx_tuple[ix + I3].template AsType<int8x4_t>()[iy / I4];

                // reference to 4 int8 data from vy_tuple
                auto& y_s4_0 = vy_tuple(iy).template AsType<int8x4_t>()(ix / I4);
                auto& y_s4_1 = vy_tuple(iy + I1).template AsType<int8x4_t>()(ix / I4);
                auto& y_s4_2 = vy_tuple(iy + I2).template AsType<int8x4_t>()(ix / I4);
                auto& y_s4_3 = vy_tuple(iy + I3).template AsType<int8x4_t>()(ix / I4);

                // transpose
                transpose_int8_4x4(x_s4_0, x_s4_1, x_s4_2, x_s4_3, y_s4_0, y_s4_1, y_s4_2, y_s4_3);
            });
        });
    }
};
#else

// S: scalar type
// NX: # of vector before transpose
// NY: # of vector after transpose
// we got [NX, NY] amount of S data to be transposed into [NY, NX] amount of S data
template <typename S_,
          index_t NX,
          index_t NY,
          typename enable_if<is_scalar_type<S_>::value, bool>::type = false>
struct transpose_vectors
{
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S = remove_cvref_t<S_>;

    using VX = vector_type<S, s_per_x>;
    using VY = vector_type<S, s_per_y>;

    __device__ void operator()(const StaticallyIndexedArray<VX, NX>& vx_tuple,
                               StaticallyIndexedArray<VY, NY>& vy_tuple)
    {
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        if constexpr(is_same_v<S, half_t> || is_same_v<S, bhalf_t>)
        {
            static_assert((NX % 2 == 0 && NY % 2 == 0), "wrong!");

            using S2 = typename vector_type<S, 2>::type;

            // loop over 2x2 tile and transpose data from vx_tuple into vy_tuple
            static_for<0, NY, 2>{}([&](auto iy) {
                static_for<0, NX, 2>{}([&](auto ix) {
                    // 2 16bitx2 data from vx_tuple to be transposed
                    const int32_t x_s2_0 =
                        bit_cast<int32_t>(vx_tuple[ix].template AsType<S2>()[iy / I2]);
                    const int32_t x_s2_1 =
                        bit_cast<int32_t>(vx_tuple[ix + I1].template AsType<S2>()[iy / I2]);

                    constexpr int32_t m0 = 0x05040100;
                    constexpr int32_t m1 = 0x07060302;

                    // transpose 2x2 16bit
                    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
                    //                   -- -- -- --     -- -- -- --      -  -  -  -
                    //             index  7  6  5  4      3  2  1  0     33 77 44 88
                    // index is reversed because of little endianness (least significant bits first)
                    const int32_t y_s2_0 = __builtin_amdgcn_perm(x_s2_1, x_s2_0, m0);
                    const int32_t y_s2_1 = __builtin_amdgcn_perm(x_s2_1, x_s2_0, m1);

                    // 2 16bitx2 data after transposed
                    vy_tuple(iy).template AsType<S2>()(ix / I2)      = bit_cast<S2>(y_s2_0);
                    vy_tuple(iy + I1).template AsType<S2>()(ix / I2) = bit_cast<S2>(y_s2_1);
                });
            });
        }
        else if constexpr(is_same_v<S, int8_t> || is_same_v<S, f8_t> || is_same_v<S, bf8_t>)
        {
            static_assert((NX % 4 == 0 && NY % 4 == 0), "wrong!");

            using S4 = typename vector_type<S, 4>::type;

            // loop over 4x4 tile and transpose data from vx_tuple into vy_tuple
            static_for<0, NY, 4>{}([&](auto iy) {
                static_for<0, NX, 4>{}([&](auto ix) {
                    // 4 int8x4 data from vx_tuple
                    const int32_t x_s4_0 =
                        bit_cast<int32_t>(vx_tuple[ix].template AsType<S4>()[iy / I4]);
                    const int32_t x_s4_1 =
                        bit_cast<int32_t>(vx_tuple[ix + I1].template AsType<S4>()[iy / I4]);
                    const int32_t x_s4_2 =
                        bit_cast<int32_t>(vx_tuple[ix + I2].template AsType<S4>()[iy / I4]);
                    const int32_t x_s4_3 =
                        bit_cast<int32_t>(vx_tuple[ix + I3].template AsType<S4>()[iy / I4]);

                    // transpose
                    int32_t t_s4_0, t_s4_1;
                    int32_t y_s4_0, y_s4_1, y_s4_2, y_s4_3;

                    constexpr int32_t m0 = 0x05010400;
                    constexpr int32_t m1 = 0x05040100;
                    constexpr int32_t m2 = 0x07060302;
                    constexpr int32_t m3 = 0x07030602;

                    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
                    //                   -- -- -- --     -- -- -- --      -  -  -  -
                    //             index  7  6  5  4      3  2  1  0     33 77 44 88
                    // index is reversed because of little endianness (least significant bits first)
                    t_s4_0 = __builtin_amdgcn_perm(x_s4_1, x_s4_0, m0);
                    t_s4_1 = __builtin_amdgcn_perm(x_s4_3, x_s4_2, m0);
                    y_s4_0 = __builtin_amdgcn_perm(t_s4_1, t_s4_0, m1);
                    y_s4_1 = __builtin_amdgcn_perm(t_s4_1, t_s4_0, m2);
                    t_s4_0 = __builtin_amdgcn_perm(x_s4_1, x_s4_0, m3);
                    t_s4_1 = __builtin_amdgcn_perm(x_s4_3, x_s4_2, m3);
                    y_s4_2 = __builtin_amdgcn_perm(t_s4_1, t_s4_0, m1);
                    y_s4_3 = __builtin_amdgcn_perm(t_s4_1, t_s4_0, m2);

                    // 4 int8x4 data from vy_tuple
                    vy_tuple(iy).template AsType<S4>()(ix / I4)      = bit_cast<S4>(y_s4_0);
                    vy_tuple(iy + I1).template AsType<S4>()(ix / I4) = bit_cast<S4>(y_s4_1);
                    vy_tuple(iy + I2).template AsType<S4>()(ix / I4) = bit_cast<S4>(y_s4_2);
                    vy_tuple(iy + I3).template AsType<S4>()(ix / I4) = bit_cast<S4>(y_s4_3);
                });
            });
        }
    }
};
#endif

} // namespace ck
