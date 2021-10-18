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
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

#if 1 // debug
    const vector_type<half_t, 2> vx0{x0}, vx1{x1};
    vector_type<half_t, 2> vy0, vy1;

    vy0.template AsType<half_t>()(I0) = vx0.template AsType<half_t>()[I0];
    vy0.template AsType<half_t>()(I1) = vx1.template AsType<half_t>()[I0];

    vy1.template AsType<half_t>()(I0) = vx0.template AsType<half_t>()[I1];
    vy1.template AsType<half_t>()(I1) = vx1.template AsType<half_t>()[I1];

    y0 = vy0.template AsType<half2_t>()[I0];
    y1 = vy1.template AsType<half2_t>()[I0];
#endif
}

template <index_t NX, index_t NY>
struct transpose_vectors<half_t, NX, NY>
{
    using X = typename vector_type<half_t, NY>::type;
    using Y = typename vector_type<half_t, NX>::type;

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
