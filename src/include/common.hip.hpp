#pragma once
#include "data_type.hip.hpp"
#include "constant_integral.hip.hpp"
#include "Sequence.hip.hpp"
#include "Array.hip.hpp"
#include "functional.hip.hpp"

extern "C" __attribute__((address_space(3))) void* __to_local(void* p)[[hc]];

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

template <class T1, class T2>
struct is_same
{
    static const bool value = false;
};

template <class T>
struct is_same<T, T>
{
    static const bool value = true;
};

#if DEVICE_BACKEND_CUDA
template <typename T>
__host__ __device__ constexpr T max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__host__ __device__ constexpr T min(T a, T b)
{
    return a < b ? a : b;
}
#endif

__host__ __device__ constexpr index_t integer_divide_ceil(index_t a, index_t b)
{
    return (a + b - 1) / b;
}
