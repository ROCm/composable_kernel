// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Extended unit tests for Problem - covers dimension inference, validation, edge cases

#include "ck_tile/dispatcher/problem.hpp"
#include <gtest/gtest.h>
#include <limits>

using namespace ck_tile::dispatcher;

// =============================================================================
// Dimension Inference Tests
// =============================================================================

class ProblemDimensionInferenceTest : public ::testing::Test
{
};

TEST_F(ProblemDimensionInferenceTest, FromAB_Basic)
{
    // A: MxK (1024x512), B: KxN (512x2048)
    auto problem = Problem::from_ab(1024, 512, 512, 2048);

    EXPECT_EQ(problem.M, 1024);
    EXPECT_EQ(problem.N, 2048);
    EXPECT_EQ(problem.K, 512);
    EXPECT_TRUE(problem.is_valid());
}

TEST_F(ProblemDimensionInferenceTest, FromDimensions_Valid)
{
    // A: 1024x512, B: 512x2048, C: 1024x2048
    auto problem = Problem::from_dimensions(1024, 512, 512, 2048, 1024, 2048);

    EXPECT_EQ(problem.M, 1024);
    EXPECT_EQ(problem.N, 2048);
    EXPECT_EQ(problem.K, 512);
    EXPECT_TRUE(problem.is_valid());
}

TEST_F(ProblemDimensionInferenceTest, FromShapes_WithC)
{
    TensorShape A{1024, 512, false};
    TensorShape B{512, 2048, false};
    TensorShape C{1024, 2048, false};

    auto problem = Problem::from_shapes(A, B, C);

    EXPECT_EQ(problem.M, 1024);
    EXPECT_EQ(problem.N, 2048);
    EXPECT_EQ(problem.K, 512);
    EXPECT_TRUE(problem.is_valid());
}

TEST_F(ProblemDimensionInferenceTest, FromShapes_TransposedA)
{
    // A stored as KxM (transposed)
    TensorShape A{512, 1024, true};
    TensorShape B{512, 2048, false};
    TensorShape C{1024, 2048, false};

    auto problem = Problem::from_shapes(A, B, C);

    EXPECT_EQ(problem.M, 1024);
    EXPECT_EQ(problem.N, 2048);
    EXPECT_EQ(problem.K, 512);
}

TEST_F(ProblemDimensionInferenceTest, FromShapes_TransposedB)
{
    TensorShape A{1024, 512, false};
    // B stored as NxK (transposed)
    TensorShape B{2048, 512, true};
    TensorShape C{1024, 2048, false};

    auto problem = Problem::from_shapes(A, B, C);

    EXPECT_EQ(problem.M, 1024);
    EXPECT_EQ(problem.N, 2048);
    EXPECT_EQ(problem.K, 512);
}

// =============================================================================
// Validation Tests
// =============================================================================

class ProblemValidationTest : public ::testing::Test
{
};

TEST_F(ProblemValidationTest, ValidProblem)
{
    Problem p(1024, 1024, 1024);
    EXPECT_TRUE(p.is_valid());
}

TEST_F(ProblemValidationTest, ZeroM)
{
    Problem p(0, 1024, 1024);
    EXPECT_FALSE(p.is_valid());
}

TEST_F(ProblemValidationTest, ZeroN)
{
    Problem p(1024, 0, 1024);
    EXPECT_FALSE(p.is_valid());
}

TEST_F(ProblemValidationTest, ZeroK)
{
    Problem p(1024, 1024, 0);
    EXPECT_FALSE(p.is_valid());
}

TEST_F(ProblemValidationTest, NegativeM)
{
    Problem p;
    p.M = -1;
    p.N = 1024;
    p.K = 1024;
    EXPECT_FALSE(p.is_valid());
}

TEST_F(ProblemValidationTest, ZeroKBatch)
{
    Problem p(1024, 1024, 1024);
    p.k_batch = 0;
    EXPECT_FALSE(p.is_valid());
}

TEST_F(ProblemValidationTest, ValidKBatch)
{
    Problem p(1024, 1024, 1024);
    p.k_batch = 4;
    EXPECT_TRUE(p.is_valid());
}

// =============================================================================
// num_ops Tests
// =============================================================================

class ProblemNumOpsTest : public ::testing::Test
{
};

TEST_F(ProblemNumOpsTest, SmallProblem)
{
    Problem p(10, 20, 30);
    // 2 * M * N * K = 2 * 10 * 20 * 30 = 12000
    EXPECT_EQ(p.num_ops(), 12000);
}

TEST_F(ProblemNumOpsTest, SymmetricProblem)
{
    Problem p(1024, 1024, 1024);
    // 2 * 1024^3 = 2,147,483,648
    EXPECT_EQ(p.num_ops(), 2LL * 1024 * 1024 * 1024);
}

TEST_F(ProblemNumOpsTest, AsymmetricProblem)
{
    Problem p(512, 2048, 256);
    EXPECT_EQ(p.num_ops(), 2LL * 512 * 2048 * 256);
}

TEST_F(ProblemNumOpsTest, LargeProblem)
{
    Problem p(4096, 4096, 4096);
    std::int64_t expected = 2LL * 4096 * 4096 * 4096;
    EXPECT_EQ(p.num_ops(), expected);
    EXPECT_GT(p.num_ops(), 0); // No overflow
}

// =============================================================================
// Edge Cases
// =============================================================================

class ProblemEdgeCasesTest : public ::testing::Test
{
};

TEST_F(ProblemEdgeCasesTest, MinimumValidSize)
{
    Problem p(1, 1, 1);
    EXPECT_TRUE(p.is_valid());
    EXPECT_EQ(p.num_ops(), 2);
}

TEST_F(ProblemEdgeCasesTest, NonSquare_TallMatrix)
{
    Problem p(8192, 64, 1024);
    EXPECT_TRUE(p.is_valid());
}

TEST_F(ProblemEdgeCasesTest, NonSquare_WideMatrix)
{
    Problem p(64, 8192, 1024);
    EXPECT_TRUE(p.is_valid());
}

TEST_F(ProblemEdgeCasesTest, NonSquare_DeepK)
{
    Problem p(1024, 1024, 8192);
    EXPECT_TRUE(p.is_valid());
}

TEST_F(ProblemEdgeCasesTest, SmallK)
{
    Problem p(1024, 1024, 16);
    EXPECT_TRUE(p.is_valid());
}

TEST_F(ProblemEdgeCasesTest, NonPowerOf2Dimensions)
{
    Problem p(1000, 2000, 300);
    EXPECT_TRUE(p.is_valid());
    EXPECT_EQ(p.num_ops(), 2LL * 1000 * 2000 * 300);
}

TEST_F(ProblemEdgeCasesTest, PrimeDimensions)
{
    Problem p(997, 1009, 1013); // All prime numbers
    EXPECT_TRUE(p.is_valid());
}

// =============================================================================
// Configuration Tests
// =============================================================================

class ProblemConfigurationTest : public ::testing::Test
{
};

TEST_F(ProblemConfigurationTest, DefaultConfiguration)
{
    Problem p(1024, 1024, 1024);

    EXPECT_FALSE(p.prefer_persistent);
    EXPECT_FALSE(p.enable_validation);
    EXPECT_EQ(p.smem_budget, 0);
    EXPECT_EQ(p.k_batch, 1);
}

TEST_F(ProblemConfigurationTest, SetPersistentPreference)
{
    Problem p(1024, 1024, 1024);
    p.prefer_persistent = true;

    EXPECT_TRUE(p.prefer_persistent);
    EXPECT_TRUE(p.is_valid());
}

TEST_F(ProblemConfigurationTest, SetSmemBudget)
{
    Problem p(1024, 1024, 1024);
    p.smem_budget = 65536; // 64KB

    EXPECT_EQ(p.smem_budget, 65536);
    EXPECT_TRUE(p.is_valid());
}

TEST_F(ProblemConfigurationTest, SetKBatch)
{
    Problem p(1024, 1024, 1024);

    for(int kb : {1, 2, 4, 8, 16})
    {
        p.k_batch = kb;
        EXPECT_EQ(p.k_batch, kb);
        EXPECT_TRUE(p.is_valid());
    }
}

// =============================================================================
// Copy and Assignment Tests
// =============================================================================

class ProblemCopyTest : public ::testing::Test
{
};

TEST_F(ProblemCopyTest, CopyConstruction)
{
    Problem p1(1024, 2048, 512);
    p1.prefer_persistent = true;
    p1.k_batch           = 4;

    Problem p2(p1);

    EXPECT_EQ(p2.M, 1024);
    EXPECT_EQ(p2.N, 2048);
    EXPECT_EQ(p2.K, 512);
    EXPECT_TRUE(p2.prefer_persistent);
    EXPECT_EQ(p2.k_batch, 4);
}

TEST_F(ProblemCopyTest, Assignment)
{
    Problem p1(1024, 2048, 512);
    Problem p2(256, 256, 256);

    p2 = p1;

    EXPECT_EQ(p2.M, 1024);
    EXPECT_EQ(p2.N, 2048);
    EXPECT_EQ(p2.K, 512);
}

// =============================================================================
// Builder Tests
// =============================================================================

class ProblemBuilderTest : public ::testing::Test
{
};

TEST_F(ProblemBuilderTest, BasicBuild)
{
    auto problem = ProblemBuilder().dimensions(1024, 2048, 512).build();

    EXPECT_EQ(problem.M, 1024);
    EXPECT_EQ(problem.N, 2048);
    EXPECT_EQ(problem.K, 512);
    EXPECT_TRUE(problem.is_valid());
}

TEST_F(ProblemBuilderTest, WithSplitK)
{
    auto problem = ProblemBuilder().dimensions(1024, 1024, 1024).split_k(4).build();

    EXPECT_EQ(problem.k_batch, 4);
}

TEST_F(ProblemBuilderTest, WithPersistent)
{
    auto problem = ProblemBuilder().dimensions(1024, 1024, 1024).persistent(true).build();

    EXPECT_TRUE(problem.prefer_persistent);
}

TEST_F(ProblemBuilderTest, WithSmemBudget)
{
    auto problem = ProblemBuilder().dimensions(1024, 1024, 1024).smem_budget(65536).build();

    EXPECT_EQ(problem.smem_budget, 65536);
}

TEST_F(ProblemBuilderTest, ChainedConfiguration)
{
    auto problem = ProblemBuilder()
                       .dimensions(2048, 2048, 1024)
                       .split_k(2)
                       .persistent(true)
                       .smem_budget(32768)
                       .validate(true)
                       .build();

    EXPECT_EQ(problem.M, 2048);
    EXPECT_EQ(problem.N, 2048);
    EXPECT_EQ(problem.K, 1024);
    EXPECT_EQ(problem.k_batch, 2);
    EXPECT_TRUE(problem.prefer_persistent);
    EXPECT_EQ(problem.smem_budget, 32768);
    EXPECT_TRUE(problem.enable_validation);
}

TEST_F(ProblemBuilderTest, FromAB)
{
    auto problem = ProblemBuilder().from_ab(1024, 512, 512, 2048).build();

    EXPECT_EQ(problem.M, 1024);
    EXPECT_EQ(problem.N, 2048);
    EXPECT_EQ(problem.K, 512);
}

// =============================================================================
// Dimension Mismatch Error Tests
// =============================================================================

class ProblemDimensionErrorTest : public ::testing::Test
{
};

TEST_F(ProblemDimensionErrorTest, KMismatchThrows)
{
    EXPECT_THROW((void)Problem::from_ab(1024, 512, 256, 2048), // K mismatch: 512 vs 256
                 std::invalid_argument);
}

TEST_F(ProblemDimensionErrorTest, MDimensionMismatchThrows)
{
    TensorShape A{1024, 512, false};
    TensorShape B{512, 2048, false};
    TensorShape C{512, 2048, false}; // M mismatch: A says M=1024, C says M=512

    EXPECT_THROW((void)Problem::from_shapes(A, B, C), std::invalid_argument);
}

TEST_F(ProblemDimensionErrorTest, NDimensionMismatchThrows)
{
    TensorShape A{1024, 512, false};
    TensorShape B{512, 2048, false};
    TensorShape C{1024, 1024, false}; // N mismatch: B says N=2048, C says N=1024

    EXPECT_THROW((void)Problem::from_shapes(A, B, C), std::invalid_argument);
}

// =============================================================================
// Validate Sizes Tests
// =============================================================================

class ProblemValidateSizesTest : public ::testing::Test
{
};

TEST_F(ProblemValidateSizesTest, CorrectSizes)
{
    Problem p(1024, 2048, 512);

    // This should not throw
    EXPECT_NO_THROW(p.validate_sizes(1024 * 512, // A size
                                     512 * 2048, // B size
                                     1024 * 2048 // C size
                                     ));
}

TEST_F(ProblemValidateSizesTest, WrongASizeThrows)
{
    Problem p(1024, 2048, 512);

    EXPECT_THROW(p.validate_sizes(1024 * 256, // Wrong A size
                                  512 * 2048,
                                  1024 * 2048),
                 std::invalid_argument);
}

TEST_F(ProblemValidateSizesTest, WrongBSizeThrows)
{
    Problem p(1024, 2048, 512);

    EXPECT_THROW(p.validate_sizes(1024 * 512,
                                  256 * 2048, // Wrong B size
                                  1024 * 2048),
                 std::invalid_argument);
}

TEST_F(ProblemValidateSizesTest, WrongCSizeThrows)
{
    Problem p(1024, 2048, 512);

    EXPECT_THROW(p.validate_sizes(1024 * 512,
                                  512 * 2048,
                                  512 * 1024 // Wrong C size
                                  ),
                 std::invalid_argument);
}
