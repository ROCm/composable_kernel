// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck/utility/index_expression.hpp"

using namespace ck;

/**
 * Test basic evaluation of literal values and index variables
 * - Number<7> should evaluate to the literal constant 7 regardless of index value
 * - Ik (index variable) should evaluate to the provided index value
 */
TEST(IndexExpression, EvalLiteralAndIk)
{
    EXPECT_EQ((index_expression::eval_v<Number<7>, 3>), 7);
    EXPECT_EQ((index_expression::eval_v<Number<7>, 5>), 7);

    EXPECT_EQ((index_expression::eval_v<index_expression::Ik, 3>), 3);
    EXPECT_EQ((index_expression::eval_v<index_expression::Ik, 7>), 7);
}

/**
 * Test arithmetic operations with index expressions
 */
TEST(IndexExpression, EvalAddMultDivMod)
{

    using ExprAdd  = index_expression::Add<index_expression::Ik, Number<5>>;
    using ExprMult = index_expression::Mult<ExprAdd, Number<2>>;
    using ExprDiv  = index_expression::Div<ExprMult, Number<4>>;
    using ExprMod  = index_expression::Mod<ExprMult, Number<3>>;

    EXPECT_EQ((index_expression::eval_v<ExprAdd, 3>), 8);
    EXPECT_EQ((index_expression::eval_v<ExprMult, 3>), 16);
    EXPECT_EQ((index_expression::eval_v<ExprDiv, 3>), 4);
    EXPECT_EQ((index_expression::eval_v<ExprMod, 3>), 1);
}

/**
 * Test nested compound expressions to verify proper precedence and composition
 */
TEST(IndexExpression, EvalNestedExpression)
{
    // Build nested expression: (ik + (2 * 5)) / 2
    using InnerMult = index_expression::Mult<Number<2>, Number<5>>;
    using InnerAdd  = index_expression::Add<index_expression::Ik, InnerMult>;
    using Expr      = index_expression::Div<InnerAdd, Number<2>>;

    // With ik=6: ((6 + (2*5)) / 2) = 8
    EXPECT_EQ((index_expression::eval_v<Expr, 6>), 8);
}
