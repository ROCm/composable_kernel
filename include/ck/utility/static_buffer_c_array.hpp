// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "data_type.hpp"

namespace ck {

// clang-format off
template<typename T_, index_t N_>
struct static_buffer_c {
    using type = T_;
    static constexpr index_t N = N_;

    type data[N];
    __host__ __device__ static constexpr auto size() { return N; }
    __host__ __device__ auto & get() {return data; }
    __host__ __device__ const auto & get() const {return data; }
    __host__ __device__ auto & get(index_t i) {return data[i]; }
    __host__ __device__ const auto & get(index_t i) const {return data[i]; }

#define SB_COMMON_AS() \
            static_assert(sizeof(type) * N % sizeof(Tx) == 0); \
            constexpr int vx = sizeof(type) * N / sizeof(Tx)

    template<typename Tx>
    __host__ __device__ auto & get_as() {SB_COMMON_AS();
            return reinterpret_cast<static_buffer_c<Tx, vx>&>(data);}
    template<typename Tx>
    __host__ __device__ const auto & get_as() const {SB_COMMON_AS();
            return reinterpret_cast<const static_buffer_c<Tx, vx>&>(data);}
    template<typename Tx>
    __host__ __device__ auto & get_as(index_t i) {SB_COMMON_AS();
            return reinterpret_cast<static_buffer_c<Tx, vx>&>(data).get(i);}
    template<typename Tx>
    __host__ __device__ const auto & get_as(index_t i) const {SB_COMMON_AS();
            return reinterpret_cast<const static_buffer_c<Tx, vx>&>(data).get(i);}
#undef SB_COMMON_AS
};
// clang-format on
} // namespace ck
