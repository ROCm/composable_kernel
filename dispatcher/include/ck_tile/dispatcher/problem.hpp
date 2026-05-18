// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// Tensor Information for Automatic MNK Inference
// =============================================================================

/// TensorShape: Describes tensor dimensions for automatic MNK inference
struct TensorShape
{
    std::int64_t rows;  // First dimension
    std::int64_t cols;  // Second dimension
    bool is_transposed; // Whether the tensor is transposed (column-major)

    TensorShape() : rows(0), cols(0), is_transposed(false) {}
    TensorShape(std::int64_t r, std::int64_t c, bool trans = false)
        : rows(r), cols(c), is_transposed(trans)
    {
    }

    /// Get logical M (rows when not transposed)
    [[nodiscard]] std::int64_t logical_rows() const { return is_transposed ? cols : rows; }

    /// Get logical N (cols when not transposed)
    [[nodiscard]] std::int64_t logical_cols() const { return is_transposed ? rows : cols; }
};

// =============================================================================
// Problem: Runtime Parameters
// =============================================================================

/// Problem: Runtime parameters for kernel invocation
/// Captures problem dimensions and resource constraints that vary between invocations
/// even when using the same kernel
struct Problem
{
    // Problem dimensions
    std::int64_t M; // Number of rows in A and C
    std::int64_t N; // Number of columns in B and C
    std::int64_t K; // Shared dimension (columns of A, rows of B)

    // Batch configuration
    std::int32_t k_batch; // Number of K-dimension splits for split-K GEMM

    // Resource preferences
    std::int32_t smem_budget; // Shared memory budget in bytes (0 = no constraint)
    bool prefer_persistent;   // Prefer persistent kernel variants

    // Validation control
    bool enable_validation; // Enable output validation against reference

    /// Default constructor with sensible defaults
    Problem()
        : M(0),
          N(0),
          K(0),
          k_batch(1),
          smem_budget(0),
          prefer_persistent(false),
          enable_validation(false)
    {
    }

    /// Constructor with problem dimensions
    Problem(std::int64_t m, std::int64_t n, std::int64_t k)
        : M(m),
          N(n),
          K(k),
          k_batch(1),
          smem_budget(0),
          prefer_persistent(false),
          enable_validation(false)
    {
    }

    /// Check if problem dimensions are valid
    [[nodiscard]] bool is_valid() const { return M > 0 && N > 0 && K > 0 && k_batch > 0; }

    /// Get total number of operations (for performance metrics)
    [[nodiscard]] std::int64_t num_ops() const
    {
        return 2 * M * N * K; // Multiply-add counts as 2 ops
    }

    // =========================================================================
    // Factory Methods for Automatic MNK Inference
    // =========================================================================

    /**
     * Create Problem by inferring MNK from tensor shapes.
     *
     * For GEMM: C[M,N] = A[M,K] x B[K,N]
     *
     * @param a_shape Shape of matrix A (M x K, or K x M if transposed)
     * @param b_shape Shape of matrix B (K x N, or N x K if transposed)
     * @param c_shape Shape of matrix C (M x N) - used for validation
     * @throws std::invalid_argument if dimensions are inconsistent
     *
     * Example:
     *   // A is 512x256, B is 256x1024, C is 512x1024
     *   auto problem = Problem::from_shapes({512, 256}, {256, 1024}, {512, 1024});
     *   // Infers: M=512, N=1024, K=256
     */
    [[nodiscard]] static Problem
    from_shapes(TensorShape a_shape, TensorShape b_shape, TensorShape c_shape)
    {
        // For C = A x B:
        // A: [M, K] (or [K, M] if transposed)
        // B: [K, N] (or [N, K] if transposed)
        // C: [M, N]

        std::int64_t M_from_A = a_shape.logical_rows();
        std::int64_t K_from_A = a_shape.logical_cols();
        std::int64_t K_from_B = b_shape.logical_rows();
        std::int64_t N_from_B = b_shape.logical_cols();
        std::int64_t M_from_C = c_shape.logical_rows();
        std::int64_t N_from_C = c_shape.logical_cols();

        // Validate K dimension matches between A and B
        if(K_from_A != K_from_B)
        {
            throw std::invalid_argument(
                "K dimension mismatch: A has K=" + std::to_string(K_from_A) +
                ", B has K=" + std::to_string(K_from_B));
        }

        // Validate M dimension matches between A and C
        if(M_from_A != M_from_C)
        {
            throw std::invalid_argument(
                "M dimension mismatch: A has M=" + std::to_string(M_from_A) +
                ", C has M=" + std::to_string(M_from_C));
        }

        // Validate N dimension matches between B and C
        if(N_from_B != N_from_C)
        {
            throw std::invalid_argument(
                "N dimension mismatch: B has N=" + std::to_string(N_from_B) +
                ", C has N=" + std::to_string(N_from_C));
        }

        return Problem(M_from_A, N_from_B, K_from_A);
    }

    /**
     * Create Problem from tensor dimensions (simple version without transpose).
     *
     * @param a_rows Rows of matrix A (= M)
     * @param a_cols Columns of matrix A (= K)
     * @param b_rows Rows of matrix B (= K)
     * @param b_cols Columns of matrix B (= N)
     * @param c_rows Rows of matrix C (= M) - for validation
     * @param c_cols Columns of matrix C (= N) - for validation
     * @throws std::invalid_argument if dimensions are inconsistent
     *
     * Example:
     *   // A[512,256] x B[256,1024] = C[512,1024]
     *   auto problem = Problem::from_dimensions(512, 256, 256, 1024, 512, 1024);
     */
    [[nodiscard]] static Problem from_dimensions(std::int64_t a_rows,
                                                 std::int64_t a_cols,
                                                 std::int64_t b_rows,
                                                 std::int64_t b_cols,
                                                 std::int64_t c_rows,
                                                 std::int64_t c_cols)
    {
        return from_shapes(
            TensorShape(a_rows, a_cols), TensorShape(b_rows, b_cols), TensorShape(c_rows, c_cols));
    }

    /**
     * Create Problem from A and B dimensions only (C is inferred).
     *
     * @param a_rows Rows of matrix A (= M)
     * @param a_cols Columns of matrix A (= K)
     * @param b_rows Rows of matrix B (= K) - validated
     * @param b_cols Columns of matrix B (= N)
     * @throws std::invalid_argument if K dimensions don't match
     *
     * Example:
     *   // A[512,256] x B[256,1024] = C[512,1024]
     *   auto problem = Problem::from_ab(512, 256, 256, 1024);
     */
    [[nodiscard]] static Problem
    from_ab(std::int64_t a_rows, std::int64_t a_cols, std::int64_t b_rows, std::int64_t b_cols)
    {
        if(a_cols != b_rows)
        {
            throw std::invalid_argument("K dimension mismatch: A.cols=" + std::to_string(a_cols) +
                                        ", B.rows=" + std::to_string(b_rows));
        }
        return Problem(a_rows, b_cols, a_cols);
    }

    /**
     * Validate that tensor pointers have consistent sizes.
     * Call this before kernel execution to catch dimension errors early.
     *
     * @param a_size Total elements in A tensor
     * @param b_size Total elements in B tensor
     * @param c_size Total elements in C tensor
     * @throws std::invalid_argument if sizes don't match expected dimensions
     */
    void validate_sizes(std::int64_t a_size, std::int64_t b_size, std::int64_t c_size) const
    {
        std::int64_t expected_a = M * K;
        std::int64_t expected_b = K * N;
        std::int64_t expected_c = M * N;

        if(a_size != expected_a)
        {
            throw std::invalid_argument("A tensor size mismatch: got " + std::to_string(a_size) +
                                        ", expected " + std::to_string(expected_a) + " (M*K = " +
                                        std::to_string(M) + "*" + std::to_string(K) + ")");
        }
        if(b_size != expected_b)
        {
            throw std::invalid_argument("B tensor size mismatch: got " + std::to_string(b_size) +
                                        ", expected " + std::to_string(expected_b) + " (K*N = " +
                                        std::to_string(K) + "*" + std::to_string(N) + ")");
        }
        if(c_size != expected_c)
        {
            throw std::invalid_argument("C tensor size mismatch: got " + std::to_string(c_size) +
                                        ", expected " + std::to_string(expected_c) + " (M*N = " +
                                        std::to_string(M) + "*" + std::to_string(N) + ")");
        }
    }
};

// =============================================================================
// Convenience Builders
// =============================================================================

/// Builder pattern for Problem configuration
class ProblemBuilder
{
    public:
    ProblemBuilder() = default;

    /// Set dimensions from A and B shapes
    ProblemBuilder&
    from_ab(std::int64_t a_rows, std::int64_t a_cols, std::int64_t b_rows, std::int64_t b_cols)
    {
        problem_ = Problem::from_ab(a_rows, a_cols, b_rows, b_cols);
        return *this;
    }

    /// Set MNK directly
    ProblemBuilder& dimensions(std::int64_t m, std::int64_t n, std::int64_t k)
    {
        problem_.M = m;
        problem_.N = n;
        problem_.K = k;
        return *this;
    }

    /// Set split-K batch count
    ProblemBuilder& split_k(std::int32_t k_batch)
    {
        problem_.k_batch = k_batch;
        return *this;
    }

    /// Set shared memory budget
    ProblemBuilder& smem_budget(std::int32_t budget)
    {
        problem_.smem_budget = budget;
        return *this;
    }

    /// Prefer persistent kernels
    ProblemBuilder& persistent(bool prefer = true)
    {
        problem_.prefer_persistent = prefer;
        return *this;
    }

    /// Enable validation
    ProblemBuilder& validate(bool enable = true)
    {
        problem_.enable_validation = enable;
        return *this;
    }

    /// Build the Problem
    [[nodiscard]] Problem build() const
    {
        if(!problem_.is_valid())
        {
            throw std::invalid_argument("Invalid problem dimensions");
        }
        return problem_;
    }

    private:
    Problem problem_;
};

} // namespace dispatcher
} // namespace ck_tile
