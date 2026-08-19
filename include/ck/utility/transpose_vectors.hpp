// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "statically_indexed_array.hpp"
#include "data_type.hpp"

namespace ck {

template <typename S, index_t NX, index_t NY, typename S2 = void>
struct transpose_vectors;

// transpose b16 (bf16/fp16) 2x2
template <typename T>
__device__ void transpose_b16_2x2(const T& x0, const T& x1, T& y0, T& y1)
{
    constexpr int32_t m0 = 0x05040100;
    constexpr int32_t m1 = 0x07060302;

    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
    //                   -- -- -- --     -- -- -- --      -  -  -  -
    //             index  7  6  5  4      3  2  1  0     33 77 44 88
    // index is reversed because of little endianness (least significant bits first)
    y0 = bit_cast<T>(__builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m0));
    y1 = bit_cast<T>(__builtin_amdgcn_perm(bit_cast<int32_t>(x1), bit_cast<int32_t>(x0), m1));
}

template <typename T, index_t NX, index_t NY>
struct transpose_vectors<
    T,
    NX,
    NY,
    typename enable_if<is_same<T, bhalf_t>::value || is_same<T, half_t>::value, void>::type>
{
    // we got [NY * NX] amount of S data to be transposed
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S  = T;
    using VX = vector_type<T, s_per_x>;
    using VY = vector_type<T, s_per_y>;

    __device__ void operator()(const StaticallyIndexedArray<const VX&, NX>& vx_tuple,
                               StaticallyIndexedArray<VY&, NY>& vy_tuple)
    {
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};

        using VTx2 = typename vector_type<T, 2>::type;

        static_assert((NX % 2 == 0 && NY % 2 == 0), "wrong!");

        // loop over 2x2 tile and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 2>{}([&](auto iy) {
            static_for<0, NX, 2>{}([&](auto ix) {
                // reference to 2 b16x2 data from vx_tuple
                const auto& x_s2_0 = vx_tuple[ix].template AsType<VTx2>()[iy / I2];
                const auto& x_s2_1 = vx_tuple[ix + I1].template AsType<VTx2>()[iy / I2];

                // reference to 2 b16x2 data from vy_tuple
                auto& y_s2_0 = vy_tuple(iy).template AsType<VTx2>()(ix / I2);
                auto& y_s2_1 = vy_tuple(iy + I1).template AsType<VTx2>()(ix / I2);

                // transpose
                transpose_b16_2x2(x_s2_0, x_s2_1, y_s2_0, y_s2_1);
            });
        });
    }
};

// transpose b8 4x4
template <typename T>
__device__ void
transpose_b8_4x4(const T& x0, const T& x1, const T& x2, const T& x3, T& y0, T& y1, T& y2, T& y3)
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

    y0 = bit_cast<T>(z0);
    y1 = bit_cast<T>(z1);
    y2 = bit_cast<T>(z2);
    y3 = bit_cast<T>(z3);
}

template <typename T, index_t NX, index_t NY>
struct transpose_vectors<
    T,
    NX,
    NY,
    typename enable_if<is_same<T, int8_t>::value || is_same<T, f8_t>::value, void>::type>
{
    // we got [NY * NX] amount of S data to be transposed
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S  = T;
    using VX = vector_type<T, s_per_x>;
    using VY = vector_type<T, s_per_y>;

    __device__ void operator()(const StaticallyIndexedArray<const VX&, NX>& vx_tuple,
                               StaticallyIndexedArray<VY&, NY>& vy_tuple)
    {
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};
        static constexpr auto I3 = Number<3>{};
        static constexpr auto I4 = Number<4>{};

        using VTx4 = typename vector_type<T, 4>::type;

        static_assert((NX % 4 == 0 && NY % 4 == 0), "wrong!");

        // loop over 4x4 tile and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 4>{}([&](auto iy) {
            static_for<0, NX, 4>{}([&](auto ix) {
                // reference to 4 b8 data from vx_tuple
                const auto& x_s4_0 = vx_tuple[ix].template AsType<VTx4>()[iy / I4];
                const auto& x_s4_1 = vx_tuple[ix + I1].template AsType<VTx4>()[iy / I4];
                const auto& x_s4_2 = vx_tuple[ix + I2].template AsType<VTx4>()[iy / I4];
                const auto& x_s4_3 = vx_tuple[ix + I3].template AsType<VTx4>()[iy / I4];

                // reference to 4 b8 data from vy_tuple
                auto& y_s4_0 = vy_tuple(iy).template AsType<VTx4>()(ix / I4);
                auto& y_s4_1 = vy_tuple(iy + I1).template AsType<VTx4>()(ix / I4);
                auto& y_s4_2 = vy_tuple(iy + I2).template AsType<VTx4>()(ix / I4);
                auto& y_s4_3 = vy_tuple(iy + I3).template AsType<VTx4>()(ix / I4);

                // transpose
                transpose_b8_4x4(x_s4_0, x_s4_1, x_s4_2, x_s4_3, y_s4_0, y_s4_1, y_s4_2, y_s4_3);
            });
        });
    }
};

} // namespace ck
