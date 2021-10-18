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
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using X = typename vector_type<half_t, s_per_x>::type;
    using Y = typename vector_type<half_t, s_per_y>::type;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __device__ void operator()(const StaticallyIndexedArray<X, NX>& xs,
                               StaticallyIndexedArray<Y, NY>& ys)
    {
        // TODO make this generic for any NX, NY
        static_assert((NX == 2 && NY == 2), "wrong!");

        if constexpr(NX == 2 && NY == 2)
        {
            transpose_fp16_2x2(xs[I0], xs[I1], ys(I0), ys(I1));
        }
    }
};

} // namespace ck
#endif
