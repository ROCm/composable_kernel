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
#if 0
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
            v_pack_b32_f16 %0, %1, %2 \n \
            "
                 : "=v"(y0)
                 : "v"(x0), "v"(x1));

    asm volatile("\n \
            v_pack_b32_f16 %0, %1, %2, op_sel:[1, 1] \n \
            "
                 : "=v"(y1)
                 : "v"(x0), "v"(x1));
#endif
}

template <index_t NX, index_t NY>
struct transpose_vectors<half_t, NX, NY>
{
    // we got [NY * NX] ammount of S data to be transposed
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
};

} // namespace ck
#endif
