#ifndef CK_TRANSPOSE_VECTORS_AMD_HPP
#define CK_TRANSPOSE_VECTORS_AMD_HPP

#include "config.hpp"
#include "statically_indexed_array.hpp"
#include "data_type.hpp"

namespace ck {

template <typename S,
          index_t NX,
          index_t NY,
          typename enable_if<is_scalar_type<S>::value, bool>::type = false>
struct transpose_vectors;

// transpose fp16 2x2
__device__ void transpose_fp16_2x2(const half2_t& x0, const half2_t& x1, half2_t& y0, half2_t& y1)
{
#if 0 // debug
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    const vector_type<half_t, 2> vx0{x0}, vx1{x1};
    vector_type<half_t, 2> vy0, vy1;

    vy0.template AsType<half_t>()(I0) = vx0.template AsType<half_t>()[I0];
    vy0.template AsType<half_t>()(I1) = vx1.template AsType<half_t>()[I0];

    vy1.template AsType<half_t>()(I0) = vx0.template AsType<half_t>()[I1];
    vy1.template AsType<half_t>()(I1) = vx1.template AsType<half_t>()[I1];

    y0 = vy0.template AsType<half2_t>()[I0];
    y1 = vy1.template AsType<half2_t>()[I0];
#else
    asm volatile("\n \
            v_pack_b32_f16 %0, %2, %3 \n \
            v_pack_b32_f16 %1, %2, %3, op_sel:[1, 1] \n \
            "
                 : "=v"(y0), "=v"(y1)
                 : "v"(x0), "v"(x1), "0"(y0), "1"(y1));
#endif
}

template <index_t NX, index_t NY>
struct transpose_vectors<half_t, NX, NY>
{
    // we got [NY * NX] ammount of S data to be transposed
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S = half_t;
    using X = typename vector_type<half_t, s_per_x>::type;
    using Y = typename vector_type<half_t, s_per_y>::type;

    __device__ void operator()(const StaticallyIndexedArray<X, NX>& x_tuple,
                               StaticallyIndexedArray<Y, NY>& y_tuple)
    {
        static constexpr auto I0 = Number<0>{};
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};

#if 0
        static_assert((NX == 2 && NY == 2), "wrong!");

        if constexpr(NX == 2 && NY == 2)
        {
            transpose_fp16_2x2(x_tuple[I0], x_tuple[I1], y_tuple(I0), y_tuple(I1));
        }
#else
        static_assert((NX % 2 == 0 && NY % 2 == 0), "wrong!");

        // create tuple of vector_type for holding data from x_tuple
        const auto vx_tuple = generate_tuple(
            [&](auto i) { return vector_type<S, s_per_x>{x_tuple[i]}; }, Number<NX>{});

        // create tuple of vector_type to hold intermediate data for y_tuple
        auto vy_tuple =
            generate_tuple([&](auto) { return vector_type<S, s_per_y>{}; }, Number<NY>{});

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

        // copy data from vy_tuple into y_tuple
        static_for<0, NY, 1>{}([&](auto i) { y_tuple(i) = vy_tuple[i].template AsType<Y>()[I0]; });
#endif
    }
};

} // namespace ck
#endif
