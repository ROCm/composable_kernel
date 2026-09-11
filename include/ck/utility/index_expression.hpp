// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"

/**
 * @file index_expression.hpp
 * @brief Compile-time index expression evaluation utilities
 * @details Provides types and operations for building and evaluating index expressions at compile
 * time. Supports arithmetic operations (Add, Mult, Div, Mod) on compile-time constants and index
 * variables. Expressions are built using template types and evaluated via the eval() function
 * family.
 */

namespace ck::index_expression {

/**
 * @brief Represents an index variable in a compile-time expression
 * @details Used as a placeholder for index values that are determined at evaluation time.
 * When an expression containing Ik is evaluated with eval<ik>(), the Ik type evaluates to the
 * provided ik value.
 */
struct Ik
{
};

/**
 * @brief Binary addition operation for compile-time index expressions
 * @tparam L Left operand type (must be evaluable to index_t)
 * @tparam R Right operand type (must be evaluable to index_t)
 */
template <typename L, typename R>
struct Add
{
};

/**
 * @brief Binary multiplication operation for compile-time index expressions
 * @tparam L Left operand type
 * @tparam R Right operand type
 */
template <typename L, typename R>
struct Mult
{
};

/**
 * @brief Binary division operation for compile-time index expressions
 * @tparam L Left operand type
 * @tparam R Right operand type
 */
template <typename L, typename R>
struct Div
{
};

/**
 * @brief Binary modulo operation for compile-time index expressions
 * @tparam L Left operand type
 * @tparam R Right operand type
 * @note Both operands must evaluate to integer types
 */
template <typename L, typename R>
struct Mod
{
};

template <index_t ik, index_t v>
/**
 * @brief Evaluates a literal Number value
 * @tparam ik The index variable value (unused for literal expressions)
 * @tparam v The literal constant value
 * @return The constant value v
 */
constexpr index_t eval(Number<v>)
{
    return v;
}

/**
 * @brief Evaluates an index variable to its provided value
 * @tparam ik The value to substitute for the index variable
 * @return The index value ik
 */
template <index_t ik>
constexpr index_t eval(Ik)
{
    return ik;
}

/**
 * @brief Evaluates an addition expression
 * @tparam ik The index variable value
 * @tparam L Type of left operand
 * @tparam R Type of right operand
 * @return Sum of eval(L) + eval(R)
 */
template <index_t ik, typename L, typename R>
constexpr index_t eval(Add<L, R>)
{
    return eval<ik>(L{}) + eval<ik>(R{});
}

/**
 * @brief Evaluates a multiplication expression
 * @tparam ik The index variable value
 * @tparam L Type of left operand
 * @tparam R Type of right operand
 * @return Product of eval(L) * eval(R)
 */
template <index_t ik, typename L, typename R>
constexpr index_t eval(Mult<L, R>)
{
    return eval<ik>(L{}) * eval<ik>(R{});
}

/**
 * @brief Evaluates a division expression
 * @tparam ik The index variable value
 * @tparam L Type of left operand
 * @tparam R Type of right operand (must evaluate to non-zero)
 * @return Quotient of eval(L) / eval(R)
 */
template <index_t ik, typename L, typename R>
constexpr index_t eval(Div<L, R>)
{
    constexpr index_t d = eval<ik>(R{});
    static_assert(d != 0,
                  "ck::index_expression::Div: division by zero in compile-time index expression");
    return eval<ik>(L{}) / d;
}

/**
 * @brief Evaluates a modulo expression
 * @tparam ik The index variable value
 * @tparam L Type of left operand
 * @tparam R Type of right operand
 * @return Remainder of eval(L) % eval(R)
 * @details Both operands must evaluate to integer index types
 */
template <index_t ik, typename L, typename R>
constexpr index_t eval(Mod<L, R>)
{
    using lhs_t = decltype(eval<ik>(L{}));
    using rhs_t = decltype(eval<ik>(R{}));
    static_assert(std::is_integral_v<lhs_t> && std::is_integral_v<rhs_t>,
                  "ck::index_expression::Mod: modulo requires integer operands");
    return eval<ik>(L{}) % eval<ik>(R{});
}

/**
 * @brief Compile-time evaluation helper for index expressions
 * @tparam Expr The expression to evaluate
 * @tparam ik The index variable value
 * @details Helper variable template that evaluates the given expression at compile time
 * for the provided index value.
 * @example
 *   using Expr = Add<Ik, Number<5>>;
 *   static_assert(eval_v<Expr, 3> == 8);  // Evaluates to 3 + 5 = 8
 */
template <typename Expr, index_t ik>
inline constexpr index_t eval_v = eval<ik>(Expr{});

} // namespace ck::index_expression
