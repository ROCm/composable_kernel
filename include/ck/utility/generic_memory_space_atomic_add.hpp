#pragma once
#include "data_type.hpp"

namespace ck {

template <typename X>
__device__ X atomic_add(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_add<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ uint32_t atomic_add<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float atomic_add<float>(float* p_dst, const float& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float2_t atomic_add<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

} // namespace ck
