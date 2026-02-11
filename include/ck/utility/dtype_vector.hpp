// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck/utility/data_type.hpp"
#include "ck/utility/math.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
namespace ck {

__device__ int static err = 0;

template <typename T, index_t N, typename Enable = void>
struct non_native_vector_base;

template <typename T, index_t N>
struct non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>>
{
    using data_t = typename scalar_type<T>::type; // select data_t based on the size of T
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    using data_v = NativeVectorT<data_t, N>;
    using type   = non_native_vector_base<T, N>;

    union alignas(math::next_power_of_two<N * sizeof(T)>())
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray_v2<data_t, N> dxN;
        StaticallyIndexedArray_v2<T, N> dTxN;
        StaticallyIndexedArray_v2<data_v, 1> dNx1;
    } data_;

    __host__ __device__ constexpr non_native_vector_base(data_t a) : data_{data_v(a)} {}
    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }
    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};
    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }
    __host__ __device__ constexpr operator data_t() const
    {
        if constexpr(N == 1)
        {
            return data_.dxN[Number<0>{}];
        }
        else
        {
            return data_.dxN; // XXX this should cause an error
        }
    }
    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return data_.dTxN; // XXX this should cause an error
        }
    }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const [[clang::lifetimebound]]
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }
};

// implementation for f6x16 and f6x32
template <typename T, index_t N>
struct non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 12 || sizeof(T) == 16 || sizeof(T) == 24 || sizeof(T) == 32>>
{
    using data_t    = typename scalar_type<T>::type; // select data_t based on declared base type
    using element_t = typename T::element_type; // select element_t based on declared element type
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    static constexpr size_t size_factor = sizeof(data_t) / sizeof(element_t);
    using data_v                        = NativeVectorT<element_t, N * size_factor>;
    using type                          = non_native_vector_base<T, N>;

    union alignas(math::next_power_of_two<N * sizeof(T)>())
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray_v2<data_t, N> dxN;
        StaticallyIndexedArray_v2<T, N> dTxN;
        StaticallyIndexedArray_v2<data_v, 1> dNx1;
    } data_;

    // Broadcast single value to vector
    __host__ __device__ constexpr non_native_vector_base(data_t a) : data_{}
    {
        // TODO: consider removing initialization similar to vector_type<T, 256>

        ck::static_for<0, N, 1>{}([&](auto i) {
            data_.dxN(i) = a; // broadcast value to all elements
        });
    }

    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }

    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};

    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr non_native_vector_base(element_t v) : data_{data_v(v)} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }

    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return err; // XXX this should cause an error
        }
    }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const [[clang::lifetimebound]]
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, data_v> || is_same_v<X, T>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, data_v> || is_same_v<X, T>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else
        {
            return err;
        }
    }
};

template <typename T, index_t N>
struct scalar_type<non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>>>
{
    using type                           = typename non_native_vector_base<T, N>::data_t;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct scalar_type<non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 12 || sizeof(T) == 16 || sizeof(T) == 24 || sizeof(T) == 32>>>
{
    using type                           = typename non_native_vector_base<T, N>::element_t;
    static constexpr index_t vector_size = N * non_native_vector_base<T, N>::size_factor;
};

/**
 * @brief Helper struct to determine the storage type for vector_type
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 * @tparam Enable SFINAE helper
 */
template <typename T, index_t Rank, typename Enable = void>
struct vector_type_storage;

/**
 * @brief Vector storage type for native scalar types.
 * @tparam T The element type of the vector
 * @note For Rank = 1 and native types, the storage type is simply T itself (scalar)
 */
template <typename T>
struct vector_type_storage<T, 1, enable_if_t<is_native_type<T>()>>
{
    using type = T;
};

/**
 * @brief Vector storage type for native vector types.
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 *
 * Assigns a native vector type based on the element type and rank.
 * For boolean types, uses a C-style array `T[Rank]`, otherwise uses
 * the `NativeVectorT<T, Rank>` template specialization.
 *
 * @note Special handling note:
 * Sub-byte sizes such as bool have different sizes in ext_vector_type (via NativeVectorT) vs array
 * types due to packing. Builtin vector types pack bool elements, while C++ arrays use 1 byte per
 * bool as a standard (minimum write size = 1 byte). e.g., ext_vector_type(bool, 4) is packed as
 * minimum 1 byte, while bool[4] is 4 bytes. vector_type::AsType, aliases with
 * StaticallyIndexedArray_v2 which is C-style array under the hood, so we need to avoid using
 * ext_vector_type with bool due to potential for data slicing errors.
 */
template <typename T, index_t Rank>
struct vector_type_storage<T, Rank, enable_if_t<is_native_type<T>() && (Rank > 1)>>
{
    using type = std::conditional_t<std::is_same_v<bool, T>, T[Rank], NativeVectorT<T, Rank>>;
};

/**
 * @brief Vector storage type for non-native vector types.
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 * @note For non-native types, the storage type is non_native_vector_base<T, Rank>
 */
template <typename T, index_t Rank>
struct vector_type_storage<T, Rank, enable_if_t<!is_native_type<T>()>>
{
    using type = non_native_vector_base<T, Rank>;
};

/**
 * @brief Convenience wrapper for vector_type_storage
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 */
template <typename T, index_t Rank>
using vector_type_storage_t = typename vector_type_storage<T, Rank>::type;

/**
 * @brief Trait to check whether one vector storage class is the same as another (e.g., same scalar,
 * or same vector class).
 * @tparam Lhs The source storage type
 * @tparam Rhs The comparator storage type
 *
 * Same storage classes are:
 * - Same type
 * - Same template vector types with matching base type (may have different ranks)
 * - C-style arrays of same base type (may have different ranks)
 */
template <typename Lhs, typename Rhs>
struct is_same_vector_storage_class : public false_type
{
};

/**
 * @brief Template native vector types of same base type with different ranks
 * @tparam T The base element type
 * @tparam LhsRank The rank of the source type
 * @tparam RhsRank The rank of the comparator type
 */
template <typename T, index_t LhsRank, index_t RhsRank>
struct is_same_vector_storage_class<NativeVectorT<T, LhsRank>, NativeVectorT<T, RhsRank>>
    : true_type
{
};

/**
 * @brief Template non-native vector types of same base type with different ranks
 * @tparam T The base element type
 * @tparam LhsRank The rank of the source type
 * @tparam RhsRank The rank of the comparator type
 * @tparam Enable SFINAE helper
 */
template <typename T, index_t LhsRank, index_t RhsRank, typename Enable>
struct is_same_vector_storage_class<non_native_vector_base<T, LhsRank, Enable>,
                                    non_native_vector_base<T, RhsRank, Enable>> : true_type
{
};

/**
 * @brief C-style arrays of same base type with different ranks
 * @tparam T The base element type
 * @tparam LhsRank The rank of the source type
 * @tparam RhsRank The rank of the comparator type
 */
template <typename T, index_t LhsRank, index_t RhsRank>
struct is_same_vector_storage_class<T[LhsRank], T[RhsRank]> : true_type
{
};

/**
 * @brief Convenience evaluator for is_same_vector_storage_class
 * @tparam Lhs The source storage type
 * @tparam Rhs The comparator storage type
 */
template <typename Lhs, typename Rhs>
static constexpr bool is_same_vector_storage_class_v =
    is_same_vector_storage_class<Lhs, Rhs>::value;

// Fwd declaration
template <typename T, index_t Rank>
struct vector_type;

/**
 * @brief Trait to extract element type and rank from vector_type and related types
 * @tparam T The vector type
 */
template <typename T>
struct vector_type_traits
{
    using element_type            = T;
    static constexpr index_t Rank = 1;
};

/**
 * @brief Specialization of vector_type_traits for vector_type
 * @tparam T The element type of the vector
 * @tparam Rank_ The number of elements in the vector
 */
template <typename T, index_t Rank_>
struct vector_type_traits<vector_type<T, Rank_>>
{
    using element_type            = T;
    static constexpr index_t Rank = Rank_;
};

/**
 * @brief Specialization of vector_type_traits for non_native_vector_base
 * @tparam T The element type of the vector
 * @tparam Rank_ The number of elements in the vector
 */
template <typename T, index_t Rank_>
struct vector_type_traits<non_native_vector_base<T, Rank_>>
{
    using element_type            = T;
    static constexpr index_t Rank = Rank_;
};

/**
 * @brief Specialization of vector_type_traits for NativeVectorT
 * @tparam T The element type of the vector
 * @tparam Rank_ The number of elements in the vector
 */
template <typename T, index_t Rank_>
struct vector_type_traits<NativeVectorT<T, Rank_>>
{
    using element_type            = T;
    static constexpr index_t Rank = Rank_;
};

/**
 * @brief Vector type wrapper
 * @tparam T The element type of the vector
 * @tparam Rank The number of elements in the vector
 */
template <typename T, index_t Rank>
struct vector_type
{
    /// @brief Internal storage type for vector_type.
    using StorageT = vector_type_storage_t<T, Rank>;
    using type     = StorageT;
    StorageT data_;

    /// @brief Default constructor for vector_type
    __host__ __device__ constexpr vector_type() : data_{} {}

    /// @brief Constructor for native vector initialization
    __host__ __device__ constexpr vector_type(StorageT v) : data_{v} {}

    /**
     * @brief Validates whether a type can be used in an AsType cast operation for vector_type
     * class.
     *
     * This function checks if a given type X can be legally used as an alias to either reinterpret
     * or slice (iterate) through the local storage type StorageT. The validation ensures type
     * safety and structural compatibility between the source and target vector types.
     *
     * @tparam X The target type to validate for AsType casting.
     *
     * @return constexpr bool True if the type is valid for AsType casting, false otherwise
     *
     * @note Requirements for a valid AsType<X> cast on vector_type<T, Rank>:
     *       1. The value type of X must match the storage value type (T)
     *       2. X must be either:
     *          a) A scalar type (T) where RankX == 1, OR
     *          b) A vector class that matches the storage vector class (e.g., both are
     * NativeVectorT or non_native_vector_base) where:
     *             - RankX is a power of 2, OR
     *               RankX == 3, OR
     *               RankX == Storage Rank
     *             - RankX must be <= Storage Rank
     * @example
     * auto srcVec = vector_type<float, 8>{};  // T = float, Rank = 8, native vector storage
     * auto result = srcVec.AsType<X>();       // Where datatype X could be:
     * X = NativeVectorT<float, 4>;            // OK: native vector T, RankX = 4 (power of 2)
     * X = float;                              // OK: scalar T, RankX = 1
     * X = NativeVectorT<float, 5>;            // ERROR: RankX not a power of 2, ==3, or ==Rank
     * X = int;                                // ERROR: Invalid scalar cast, T != int
     * X = float[4];                           // ERROR: Invalid type, storage vector class doesn't
     *                                         // match (native vector != C-array)
     */
    template <typename X>
    static constexpr bool is_as_type_cast_valid()
    {
        using TraitsX = vector_type_traits<X>;

        // Checks storage classes match, with same base type (may have different ranks)
        constexpr bool is_valid_cast =
            is_same_vector_storage_class_v<StorageT, X> || // Matching vector storage
            is_same_v<X, T>;                               // Matching scalar type

        // Validate vector ranks
        constexpr bool is_valid_rank = (math::is_power_of_two_integer(TraitsX::Rank) ||
                                        (TraitsX::Rank == 3) || (TraitsX::Rank == Rank)) &&
                                       (TraitsX::Rank <= Rank);

        return is_valid_cast && is_valid_rank;
    }

    /**
     * @brief Allows casting the vector_type to another type X via aliasing or slicing.
     * Use cases are to expose the internal storage type, or to slice the vector into smaller
     * vectors for iteration purposes.
     * @tparam X The target type to validate for AsType casting.
     * @returns a reference to the reinterpreted data as StaticallyIndexedArray_v2<X, Rank / RankX>.
     * Rigid control of allowable casts is enforced via static_assert to ensure type safety.
     * See is_as_type_cast_valid() for requirements.
     */
    template <typename X>
    __host__ __device__ constexpr auto const& AsType() const [[clang::lifetimebound]]
    {
        // Make this a hard error if the datatype X is not a valid cast.
        static_assert(is_as_type_cast_valid<X>(), "Datatype X is not a valid AsType cast");

        using TraitsX = vector_type_traits<X>;

        // Calculate the new rank after slicing.
        // Note: We might end up with incomplete quantization from slicing
        // when Rank % TraitsX::Rank != 0, so take the floor division.
        constexpr index_t newRank = Rank / TraitsX::Rank;

        // Determine the cast type:
        // - Scalar T if slicing to scalar or vector size of 1,
        // - X otherwise.
        using CastT   = conditional_t<TraitsX::Rank == 1, T, X>;
        using ResultT = StaticallyIndexedArray_v2<CastT, newRank>;

        // As a rule, the aliasing type should not be larger than the original type.
        static_assert(sizeof(ResultT) <= sizeof(vector_type),
                      "Resulting aliasing cannot be larger than original type");

        // Re-cast as vectorized type.
        return *(bit_cast<ResultT const*>(this));
    }

    /**
     * @brief Allows casting the vector_type to another type X via aliasing or slicing.
     * Use cases are to expose the internal storage type, or to slice the vector into smaller
     * vectors for iteration purposes.
     * @tparam X The target type to validate for AsType casting.
     * @returns a reference to the reinterpreted data as StaticallyIndexedArray_v2<X, Rank / RankX>.
     * Rigid control of allowable casts is enforced via static_assert to ensure type safety.
     * See is_as_type_cast_valid() for requirements.
     */
    template <typename X>
    __host__ __device__ constexpr auto& AsType() [[clang::lifetimebound]]
    {
        // Make this a hard error if the datatype X is not a valid cast.
        static_assert(is_as_type_cast_valid<X>(), "Datatype X is not a valid AsType cast");

        using TraitsX = vector_type_traits<X>;

        // Calculate the new rank after slicing.
        // Note: We might end up with incomplete quantization from slicing
        // when Rank % TraitsX::Rank != 0, so take the floor division.
        constexpr index_t newRank = Rank / TraitsX::Rank;

        // Determine the cast type:
        // - Scalar T if slicing to scalar or vector size of 1,
        // - X otherwise.
        using CastT   = conditional_t<TraitsX::Rank == 1, T, X>;
        using ResultT = StaticallyIndexedArray_v2<CastT, newRank>;

        // As a rule, the aliasing type should not be larger than the original type.
        static_assert(sizeof(ResultT) <= sizeof(vector_type),
                      "Resulting aliasing cannot be larger than original type");

        // Re-cast as vectorized type.
        return *(bit_cast<ResultT*>(this));
    }
};

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<NativeVectorT<T, V>, N>;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<vector_type<T, V>, N>;

/**
 * @brief scalar_type trait override for vector_type
 * @tparam T The vector type
 * @tparam N The number of elements in the vector
 */
template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
{
    using type                           = typename scalar_type<T>::type;
    static constexpr index_t vector_size = N;
};

// vector_type_maker
// This is the right way to handle "vector of vectors": making a bigger vector instead
template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<NativeVectorT<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
__host__ __device__ constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;

// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x6_t  = typename vector_type<int32_t, 6>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

// f8
using f8x2_fnuz_t  = typename vector_type<f8_fnuz_t, 2>::type;
using f8x4_fnuz_t  = typename vector_type<f8_fnuz_t, 4>::type;
using f8x8_fnuz_t  = typename vector_type<f8_fnuz_t, 8>::type;
using f8x16_fnuz_t = typename vector_type<f8_fnuz_t, 16>::type;
using f8x32_fnuz_t = typename vector_type<f8_fnuz_t, 32>::type;
using f8x64_fnuz_t = typename vector_type<f8_fnuz_t, 64>::type;

// bf8
using bf8x2_fnuz_t  = typename vector_type<bf8_fnuz_t, 2>::type;
using bf8x4_fnuz_t  = typename vector_type<bf8_fnuz_t, 4>::type;
using bf8x8_fnuz_t  = typename vector_type<bf8_fnuz_t, 8>::type;
using bf8x16_fnuz_t = typename vector_type<bf8_fnuz_t, 16>::type;
using bf8x32_fnuz_t = typename vector_type<bf8_fnuz_t, 32>::type;
using bf8x64_fnuz_t = typename vector_type<bf8_fnuz_t, 64>::type;

// f8
using f8x2_ocp_t  = typename vector_type<f8_ocp_t, 2>::type;
using f8x4_ocp_t  = typename vector_type<f8_ocp_t, 4>::type;
using f8x8_ocp_t  = typename vector_type<f8_ocp_t, 8>::type;
using f8x16_ocp_t = typename vector_type<f8_ocp_t, 16>::type;
using f8x32_ocp_t = typename vector_type<f8_ocp_t, 32>::type;
using f8x64_ocp_t = typename vector_type<f8_ocp_t, 64>::type;

// bf8
using bf8x2_ocp_t  = typename vector_type<bf8_ocp_t, 2>::type;
using bf8x4_ocp_t  = typename vector_type<bf8_ocp_t, 4>::type;
using bf8x8_ocp_t  = typename vector_type<bf8_ocp_t, 8>::type;
using bf8x16_ocp_t = typename vector_type<bf8_ocp_t, 16>::type;
using bf8x32_ocp_t = typename vector_type<bf8_ocp_t, 32>::type;
using bf8x64_ocp_t = typename vector_type<bf8_ocp_t, 64>::type;

#if CK_FP8_TYPE_OCP
// f8
using f8x2_t  = f8x2_ocp_t;
using f8x4_t  = f8x4_ocp_t;
using f8x8_t  = f8x8_ocp_t;
using f8x16_t = f8x16_ocp_t;
using f8x32_t = f8x32_ocp_t;
using f8x64_t = f8x64_ocp_t;

// bf8
using bf8x2_t  = bf8x2_ocp_t;
using bf8x4_t  = bf8x4_ocp_t;
using bf8x8_t  = bf8x8_ocp_t;
using bf8x16_t = bf8x16_ocp_t;
using bf8x32_t = bf8x32_ocp_t;
using bf8x64_t = bf8x64_ocp_t;
#elif CK_FP8_TYPE_FNUZ
// f8
using f8x2_t  = f8x2_fnuz_t;
using f8x4_t  = f8x4_fnuz_t;
using f8x8_t  = f8x8_fnuz_t;
using f8x16_t = f8x16_fnuz_t;
using f8x32_t = f8x32_fnuz_t;
using f8x64_t = f8x64_fnuz_t;

// bf8
using bf8x2_t  = bf8x2_fnuz_t;
using bf8x4_t  = bf8x4_fnuz_t;
using bf8x8_t  = bf8x8_fnuz_t;
using bf8x16_t = bf8x16_fnuz_t;
using bf8x32_t = bf8x32_fnuz_t;
using bf8x64_t = bf8x64_fnuz_t;
#endif

// u8
using uint8x2_t  = typename vector_type<uint8_t, 2>::type;
using uint8x4_t  = typename vector_type<uint8_t, 4>::type;
using uint8x8_t  = typename vector_type<uint8_t, 8>::type;
using uint8x16_t = typename vector_type<uint8_t, 16>::type;
using uint8x32_t = typename vector_type<uint8_t, 32>::type;
using uint8x64_t = typename vector_type<uint8_t, 64>::type;

// f4
using f4x2_t  = typename vector_type<f4x2_pk_t, 1>::type;
using f4x4_t  = typename vector_type<f4x2_pk_t, 2>::type;
using f4x8_t  = typename vector_type<f4x2_pk_t, 4>::type;
using f4x16_t = typename vector_type<f4x2_pk_t, 8>::type;
using f4x32_t = typename vector_type<f4x2_pk_t, 16>::type;
using f4x64_t = typename vector_type<f4x2_pk_t, 32>::type;

// f6
using f6x16_t   = typename vector_type<f6x16_pk_t, 1>::type;
using f6x16x2_t = typename vector_type<f6x16_pk_t, 2>::type;
using f6x32_t   = typename vector_type<f6x32_pk_t, 1>::type;

// bf6
using bf6x16_t   = typename vector_type<bf6x16_pk_t, 1>::type;
using bf6x16x2_t = typename vector_type<bf6x16_pk_t, 2>::type;
using bf6x32_t   = typename vector_type<bf6x32_pk_t, 1>::type;

#ifndef CK_CODE_GEN_RTC
// e8m0
using e8m0x4_bexp_t = typename vector_type<e8m0_bexp_t, 4>::type;
#endif

// pack int4
using pk_i4x2_t = typename vector_type<pk_i4_t, 2>::type;
using pk_i4x4_t = typename vector_type<pk_i4_t, 4>::type;
using pk_i4x8_t = typename vector_type<pk_i4_t, 8>::type;

} // namespace ck
#pragma clang diagnostic pop
