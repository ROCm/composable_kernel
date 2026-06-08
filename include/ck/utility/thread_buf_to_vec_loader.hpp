// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/index_expression.hpp"

namespace ck {

/**
 * @brief Invokes multiple functors based on an index parameter
 * @tparam Funcs Parameter pack of functor types
 * @details Stores a tuple of functors and provides an operator() that invokes all of them
 * with the same index parameter. Uses static_for to iterate through the functors.
 */
template <typename... Funcs>
struct FunctorInvoker
{
    ck::Tuple<Funcs...> funcs;

    __host__ __device__ constexpr FunctorInvoker(Funcs... fs) : funcs(ck::forward<Funcs>(fs)...) {}

    /**
     * @brief Invokes all functors with the given index
     * @tparam I The index to pass to each functor
     * @param i Number wrapper containing the index value
     */
    template <index_t I>
    __host__ __device__ constexpr void operator()(ck::Number<I> i) const
    {
        invoke(i, std::index_sequence_for<Funcs...>{});
    }

    private:
    template <index_t I, std::size_t... Is>
    __host__ __device__ constexpr void invoke(ck::Number<I> i, std::index_sequence<Is...>) const
    {
        (funcs[ck::Number<static_cast<index_t>(Is)>{}](i), ...);
    }
};

// required for CTAD to work with __host__ __device__ qualifiers
template <typename... Fs>
__host__ __device__ constexpr auto MakeFunctorInvoker(Fs&&... fs)
{
    return FunctorInvoker<Fs...>{ck::forward<Fs&&>(fs)...};
}

/**
 * @brief Helper struct for evaluating compile-time index expressions
 * @tparam T The expression type to evaluate
 * @tparam ik The index variable value
 * @details Provides a value member that evaluates the index expression T using
 * the index_expression::eval_v
 */
template <typename T, index_t ik>
struct IndexEval;

template <typename T, index_t ik>
struct IndexEval<const T, ik> : IndexEval<T, ik>
{
};

template <index_t v, index_t ik>
struct IndexEval<Number<v>, ik>
{
    static constexpr index_t value = v;
};

template <index_t ik>
struct IndexEval<index_expression::Ik, ik>
{
    static constexpr index_t value = ik;
};

template <typename L, typename R, index_t ik>
struct IndexEval<index_expression::Add<L, R>, ik>
{
    static constexpr index_t value = IndexEval<L, ik>::value + IndexEval<R, ik>::value;
};

template <typename L, typename R, index_t ik>
struct IndexEval<index_expression::Mult<L, R>, ik>
{
    static constexpr index_t value = IndexEval<L, ik>::value * IndexEval<R, ik>::value;
};

template <typename L, typename R, index_t ik>
struct IndexEval<index_expression::Div<L, R>, ik>
{
    static constexpr index_t divisor = IndexEval<R, ik>::value;
    static_assert(divisor != 0,
                  "ck::index_expression::Div: division by zero in compile-time index expression");
    static constexpr index_t value = IndexEval<L, ik>::value / divisor;
};

template <typename L, typename R, index_t ik>
struct IndexEval<index_expression::Mod<L, R>, ik>
{
    static constexpr index_t divisor = IndexEval<R, ik>::value;
    static_assert(divisor != 0,
                  "ck::index_expression::Mod: modulo by zero in compile-time index expression");
    static constexpr index_t value = IndexEval<L, ik>::value % divisor;
};

/**
 * @brief Loads thread elements from buffer to vector using compile-time index expressions
 * @tparam ThreadVec The vector type to load into
 * @tparam ThreadBuf The buffer type to load from
 * @tparam ThreadDesc The descriptor for thread memory layout
 * @tparam ComputeType The computation type for the result
 * @tparam IdxExpr Parameter pack of compile-time index expressions
 * @details Uses index expressions to compute offsets in ThreadBuf and loads the values
 * into the ThreadVec. The operator() accepts a compile-time index and evaluates all
 * index expressions for that particular index value.
 *
 * Example:
 * @code
 *   // Load from buffer using index expressions Ik (the loop index) and Number<5>
 *   using Loader = thread_buf_to_vec_loader<VecType, BufType, DescType, float,
 *                                            index_expression::Ik, index_expression::Number<5>>;
 *   Loader loader{thread_vec, thread_buf};
 *   loader(Number<3>{});  // Loads at offset computed by evaluating expressions with ik=3
 * @endcode
 */
template <typename ThreadVec,
          typename ThreadBuf,
          typename ThreadDesc,
          typename ComputeType,
          typename... IdxExpr>
struct thread_buf_to_vec_loader
{
    ThreadVec& thread_vec;
    ThreadBuf& thread_buf;

    __host__ __device__ constexpr thread_buf_to_vec_loader(ThreadVec& tv, ThreadBuf& tb)
        : thread_vec(tv), thread_buf(tb)
    {
    }

    /**
     * @brief Loads a single element from buffer to vector for the given index
     * @tparam ik The index value for which to evaluate the index expressions
     */
    template <index_t ik>
    __host__ __device__ constexpr void operator()(Number<ik>) const
    {
        // TODO c++20: ThreadDesc could be an auto parameter, but clang doesn't support auto
        // non-type template parameters yet
        constexpr auto thread_desc = ThreadDesc{};
        constexpr auto idx_tuple   = ck::make_tuple(Number<IndexEval<IdxExpr, ik>::value>{}...);
        constexpr auto offset      = thread_desc.CalculateOffset(idx_tuple);

        auto& target = thread_vec.template AsType<ComputeType>()(Number<ik>{});
        target       = thread_buf[Number<offset>{}];
    }
};
} // namespace ck
