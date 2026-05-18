// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <concepts>
#include <utility>
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

/**
 * @file conv_algorithm_limits.hpp
 * @brief Compile-time validation concepts and helpers for convolution algorithm configurations
 *
 * This file provides C++20 concepts and compile-time validation functions for validating
 * block transfer configurations, memory access patterns, and hardware instruction constraints
 * in convolution algorithms.
 *
 * Key features:
 * - Vector transfer size validation for VMEM and LDS operations
 * - Access order permutation validation
 * - Thread cluster dimension validation
 * - Tile coverage validation for block transfers
 */

namespace ck_tile::builder {

template <auto Value>
concept InputVectorTransferLimits = requires {
    requires Value.src_vector_dim > 0 && Value.src_scalar_per_vector > 0 &&
                     Value.lds_dst_scalar_per_vector > 0;
};

template <auto Value>
concept TileInputOutputVectorTransferLimits =
    requires { requires Value.a > 0 && Value.b > 0 && Value.c > 0; };

template <auto Value>
concept OutputVectorTransferLimits = requires {
    requires Value.scalar_per_vector > 0 && Value.m_xdl_per_wave_per_shuffle > 0 &&
                     Value.n_xdl_per_wave_per_shuffle > 0;
};

// Limits for access order. Must be a permutation of {0, 1, 2}.
template <auto Value>
concept AccessOrderLimits3D = requires {
    requires((Value[0] != Value[1]) && (Value[0] != Value[2]) && (Value[1] != Value[2]) &&
             (Value[0] >= 0 && Value[0] < 3) && (Value[1] >= 0 && Value[1] < 3) &&
             (Value[2] >= 0 && Value[2] < 3) && (Value.Size() == 3));
};

// Limits for access order. Must be a permutation of {0, 1, 2, 3}.
template <auto Value>
concept AccessOrderLimits4D = requires {
    requires((Value[0] != Value[1]) && (Value[0] != Value[2]) && (Value[0] != Value[3]) &&
             (Value[1] != Value[2]) && (Value[1] != Value[3]) && (Value[2] != Value[3]) &&
             (Value[0] >= 0 && Value[0] < 4) && (Value[1] >= 0 && Value[1] < 4) &&
             (Value[2] >= 0 && Value[2] < 4) && (Value[3] >= 0 && Value[3] < 4) &&
             (Value.Size() == 4));
};

namespace detail {

// Helper to check if access order is a valid permutation
template <auto Value>
constexpr bool is_valid_permutation()
{
    constexpr auto size = Value.Size();

    // Check all values are in range [0, size)
    for(size_t i = 0; i < size; ++i)
    {
        if(Value[i] < 0 || Value[i] >= static_cast<decltype(Value[0])>(size))
            return false;
    }

    // Check all values are unique (valid permutation)
    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = i + 1; j < size; ++j)
        {
            if(Value[i] == Value[j])
                return false;
        }
    }

    return true;
}

} // namespace detail

// Generic access order limits. Must be a valid permutation of {0, 1, ..., Dims-1}.
// Works with both 3D and 4D (or any dimensionality) access orders.
template <auto Value, size_t Dims>
concept AccessOrderLimits = requires {
    requires Value.Size() == Dims;
    requires detail::is_valid_permutation<Value>();
};

namespace detail {

// Helper trait to get compile-time size from ck::Array
template <typename T>
concept HasStaticSize = requires {
    { T::Size() } -> std::convertible_to<size_t>;
};

// Helper trait to get compile-time size from std::array and similar
template <typename T>
concept HasTupleSize = requires {
    { std::tuple_size<T>::value } -> std::convertible_to<size_t>;
};

// Helper for dependent static_assert
template <typename>
constexpr bool always_false = false;

// Get compile-time size of a range
template <typename Range>
constexpr size_t get_range_size()
{
    if constexpr(HasStaticSize<Range>)
    {
        return Range::Size();
    }
    else if constexpr(HasTupleSize<Range>)
    {
        return std::tuple_size_v<Range>;
    }
    else
    {
        static_assert(always_false<Range>, "Unsupported type of range object.");
    }
}

// Fold expression implementation for product calculation
template <typename Range, size_t... Is>
constexpr auto get_cluster_size_impl(const Range& range, std::index_sequence<Is...>)
{
    using value_type = std::remove_cvref_t<decltype(range[0])>;
    return ((range[Is]) * ... * value_type{1});
}

// Generic function that calculates the product of all elements in a range
// Works with any indexable range with compile-time size (ck::Array, std::array, etc.)
template <typename Range>
    requires requires(Range r) {
        r[0];                    // Must be indexable
        get_range_size<Range>(); // Must have compile-time size
    }
constexpr auto get_cluster_size(const Range& range)
{
    return get_cluster_size_impl(range, std::make_index_sequence<get_range_size<Range>()>{});
}

// Calculate K dimension coverage (k0 * k1, with vectorization if applicable)
template <auto BlockTransfer>
constexpr auto get_k_coverage()
{
    auto k0      = BlockTransfer.thread_cluster_dims[0];
    auto k1      = BlockTransfer.thread_cluster_dims[2];
    auto k_total = k0 * k1;

    // If vectorization is on k0 (dim 0) or k1 (dim 2), multiply by vector size
    if constexpr(BlockTransfer.src_vector_dim == 0 || BlockTransfer.src_vector_dim == 2)
    {
        k_total *= BlockTransfer.src_scalar_per_vector;
    }

    return k_total;
}

// Calculate M/N dimension coverage (m_n, with vectorization if applicable)
template <auto BlockTransfer>
constexpr auto get_mn_coverage()
{
    auto mn = BlockTransfer.thread_cluster_dims[1];

    // If vectorization is on m_n (dim 1), multiply by vector size
    if constexpr(BlockTransfer.src_vector_dim == 1)
    {
        mn *= BlockTransfer.src_scalar_per_vector;
    }

    return mn;
}

template <size_t N, DataType Type>
constexpr bool IsVmemVectorSizeValid()
{
    using enum builder::DataType;
    // We have following type & VectorSize pair constraints.
    //-----------------------------------------------------------------------------------
    // (std::is_same_v<T, double> && (N == 1 || N == 2 || N == 4 || N == 8)) ||
    // (std::is_same_v<T, float> && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
    // (std::is_same_v<T, fp16_t> &&
    //     (N == 1 || N == 2 || N == 4 || N == 6 || N == 8 || N == 16 || N == 32)) ||
    // (std::is_same_v<T, bf16_t> &&
    //     (N == 1 || N == 2 || N == 4 || N == 6 || N == 8 || N == 16 || N == 32)) ||
    // (std::is_same_v<T, int32_t> &&
    //     (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
    // (std::is_same_v<T, fp8_t> && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
    // (std::is_same_v<T, bf8_t> && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
    // (std::is_same_v<T, int8_t> && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
    // (std::is_same_v<T, e8m0_t> && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
    // (std::is_same_v<T, pk_int4_t> &&
    //     (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
    // (std::is_same_v<T, pk_fp4_raw_t> &&
    //     (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
    // (std::is_same_v<T, pk_fp4_t> && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16))
    //-----------------------------------------------------------------------------------
    // explicitly not using switch statement since we do not handle all possible data types
    // in DataType structure yet, so that I could cover all of them in `else` branch.
    if constexpr(Type == FP64)
    {
        return N == 1 || N == 2 || N == 4 || N == 8;
    }
    else if constexpr(Type == FP32)
    {
        return N == 1 || N == 2 || N == 4 || N == 8 || N == 16;
    }
    else if constexpr(Type == I32)
    {
        return N == 1 || N == 2 || N == 4 || N == 8 || N == 16;
    }
    else if constexpr(Type == FP16 || Type == BF16)
    {
        return N == 1 || N == 2 || N == 4 || N == 6 || N == 8 || N == 16 || N == 32;
    }
    else if constexpr(Type == FP8 || Type == BF8)
    {
        return N == 1 || N == 2 || N == 4 || N == 8 || N == 16;
    }
    else if constexpr(Type == I8)
    {
        return N == 1 || N == 2 || N == 4 || N == 8 || N == 16;
    }
    else
    {
        static_assert(always_false<void>, "Unsupported memory instruction data type!");
    }
}

// Valid LDS instruction bit sizes based on supported DS_READ/DS_WRITE operations
// DS_READ_{B32,B64,B96,B128,U8,I8,U16,I16}
// DS_WRITE_{B32,B64,B96,B128,B8,B16}
template <size_t N, size_t DataTypeSize>
constexpr bool IsLDSVectorSizeValid()
{
    constexpr size_t bits = N * DataTypeSize * 8;
    return ck_tile::is_any_value_of(bits, 8, 16, 32, 64, 96, 128);
}

} // namespace detail

// product of thread cluster lengths must be <= workgroup size
template <auto BlockTransfer, size_t BlockSize>
concept ValidBlockTransferClusterSize =
    requires { requires detail::get_cluster_size(BlockTransfer.thread_cluster_dims) <= BlockSize; };

// Check that thread cluster covers the K and M dimensions for A transfer
template <auto ABlockTransfer, auto TileSize>
concept ThreadsCoverATile = requires {
    // K dimension: k0 * k1 * (vectorization) must divide K
    requires TileSize.k % detail::get_k_coverage<ABlockTransfer>() == 0;
    // M dimension: m_n * (vectorization) must divide M
    requires TileSize.m % detail::get_mn_coverage<ABlockTransfer>() == 0;
};

// Check that thread cluster covers the K and N dimensions for B transfer
template <auto BBlockTransfer, auto TileSize>
concept ThreadsCoverBTile = requires {
    // K dimension: k0 * k1 * (vectorization) must divide K
    requires TileSize.k % detail::get_k_coverage<BBlockTransfer>() == 0;
    // N dimension: m_n * (vectorization) must divide N
    requires TileSize.n % detail::get_mn_coverage<BBlockTransfer>() == 0;
};

template <auto CBlockTransfer, auto TileSize>
concept ThreadsCoverCTile = requires {
    // M dimension: m_wave_per_xdl must divide M
    requires TileSize.m % CBlockTransfer.thread_cluster_dims[1] == 0;
    // N dimension: n_wave_per_xdl * (vectorization) must divide N
    requires TileSize.n % (CBlockTransfer.thread_cluster_dims[3] *
                           CBlockTransfer.scalar_per_vector) == 0;
};

template <size_t N, DataType Type>
concept IsVmemVectorSizeValid = detail::IsVmemVectorSizeValid<N, Type>();

template <size_t N, size_t DataTypeSize>
concept IsLDSVectorSizeValid = detail::IsLDSVectorSizeValid<N, DataTypeSize>();

// Composite concept for input block transfer validation (A)
// Includes all validations: vector transfer limits, access order, cluster size,
// vector size validity, and tile coverage
template <auto A_BlockTransfer,
          DataType Type,
          size_t TypeSize,
          size_t BlockSize,
          auto TileSize,
          size_t ThreadClusterRank = 3>
concept ValidABlockTransfer =
    InputVectorTransferLimits<A_BlockTransfer> &&
    AccessOrderLimits<A_BlockTransfer.thread_cluster_order, ThreadClusterRank> &&
    AccessOrderLimits<A_BlockTransfer.src_access_order, ThreadClusterRank> &&
    ValidBlockTransferClusterSize<A_BlockTransfer, BlockSize> &&
    IsVmemVectorSizeValid<A_BlockTransfer.src_scalar_per_vector, Type> &&
    IsLDSVectorSizeValid<A_BlockTransfer.lds_dst_scalar_per_vector, TypeSize> &&
    ThreadsCoverATile<A_BlockTransfer, TileSize>;

// Composite concept for input block transfer validation (B)
template <auto B_BlockTransfer,
          DataType Type,
          size_t TypeSize,
          size_t BlockSize,
          auto TileSize,
          size_t ThreadClusterRank = 3>
concept ValidBBlockTransfer =
    InputVectorTransferLimits<B_BlockTransfer> &&
    AccessOrderLimits<B_BlockTransfer.thread_cluster_order, ThreadClusterRank> &&
    AccessOrderLimits<B_BlockTransfer.src_access_order, ThreadClusterRank> &&
    ValidBlockTransferClusterSize<B_BlockTransfer, BlockSize> &&
    IsVmemVectorSizeValid<B_BlockTransfer.src_scalar_per_vector, Type> &&
    IsLDSVectorSizeValid<B_BlockTransfer.lds_dst_scalar_per_vector, TypeSize> &&
    ThreadsCoverBTile<B_BlockTransfer, TileSize>;

// Composite concept for output block transfer validation (C)
template <auto C_BlockTransfer, DataType Type, size_t BlockSize, auto TileSize>
concept ValidCBlockTransfer = OutputVectorTransferLimits<C_BlockTransfer> &&
                              ValidBlockTransferClusterSize<C_BlockTransfer, BlockSize> &&
                              IsVmemVectorSizeValid<C_BlockTransfer.scalar_per_vector, Type> &&
                              ThreadsCoverCTile<C_BlockTransfer, TileSize>;

// Usage: IsValidLayout<ACTUAL_LAYOUT, VALID_LAYOUT_1, VALID_LAYOUT_2, ...>
template <auto ACTUAL_LAYOUT, auto... VALID_LAYOUTS>
concept IsValidLayout = ck_tile::is_any_value_of(ACTUAL_LAYOUT, VALID_LAYOUTS...);

} // namespace ck_tile::builder
