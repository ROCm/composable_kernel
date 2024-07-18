// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

template <typename T>
__host__ __device__ inline void print(T t)
{
    t.Print();
}

template <>
__host__ __device__ inline void print(bool v)
{
    printf("%d", static_cast<int32_t>(v));
}

template <>
__host__ __device__ inline void print(int32_t v)
{
    printf("%d", v);
}

template <>
__host__ __device__ inline void print(int64_t v)
{
    printf("%ld", v);
}

template <>
__host__ __device__ inline void print(float v)
{
    printf("%f", v);
}

template <>
__host__ __device__ inline void print(_Float16 v)
{
    printf("%f", static_cast<float>(v));
}

} // namespace ck
