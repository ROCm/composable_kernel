// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_contraction/pipeline/batched_contraction_problem.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"

/**
 * @file batched_contraction_kernel.hpp
 * @brief Batched Tensor Contraction Operations
 *
 * @section batched_contraction_overview What is Batched Tensor Contraction with Multiple D?
 *
 * Tensor contraction is a fundamental operation that generalizes matrix multiplication to
 * multi-dimensional tensors. It performs element-wise multiplication and summation over
 * shared dimensions
 *
 * **Beyond pure contraction, this kernel supports multiple auxiliary input tensors (D tensors)**
 * that are fused with the contraction result through configurable epilogue operations, enabling
 * efficient computation of complex tensor expressions in a single kernel launch.
 *
 * @subsection mathematical_formulation Mathematical Formulation
 *
 * For tensors A and B with arbitrary dimensionalities, the complete operation computes:
 *
 * **E[G₀,G₁,...,M₀,M₁,...,N₀,N₁,...] = epilogue_op(C, D₀, D₁, D₂, ...)**
 *
 * Where:
 * **C[G₀,G₁,...,M₀,M₁,...,N₀,N₁,...] = Σ_{K₀,K₁,...} A[G₀,G₁,...,M₀,M₁,...,K₀,K₁,...] ×
 * B[G₀,G₁,...,N₀,N₁,...,K₀,K₁,...]**
 *
 * Where:
 * - **G dimensions**: Batch dimensions (shared across A, B, and output E)
 * - **M dimensions**: Row dimensions of the output matrix (from tensor A)
 * - **N dimensions**: Column dimensions of the output matrix (from tensor B)
 * - **K dimensions**: Contraction dimensions (summed over, present in both A and B)
 *
 * @subsection why_gemm_implementation Why Tensor Contraction Can Be Implemented Using GEMM
 *
 * **Mathematical Equivalence**: Tensor contraction is fundamentally equivalent to matrix
 * multiplication when dimensions are appropriately flattened. The key insight is that the summation
 * operation over shared dimensions (K dimensions) in tensor contraction is mathematically identical
 * to the dot product computation in matrix multiplication.
 *
 * **Dimension Flattening Strategy**:
 * - **M dimensions** (from tensor A) → Flattened into matrix rows (M_total)
 * - **N dimensions** (from tensor B) → Flattened into matrix columns (N_total)
 * - **K dimensions** (contraction dims) → Flattened into inner dimension (K_total)
 * - **G dimensions** (batch dims) → Handled through batch processing
 *
 * **Mathematical Transformation**:
 * ```
 * Original: E[g,m₀,m₁,n₀,n₁] = Σ_{k₀,k₁} A[g,m₀,m₁,k₀,k₁] × B[g,n₀,n₁,k₀,k₁]
 * Flattened: E[g,M,N] = Σ_K A[g,M,K] × B[g,N,K]  (where M=m₀×m₁, N=n₀×n₁, K=k₀×k₁)
 * GEMM Form: E = A × Bᵀ
 *
 * **Why This Approach Is Optimal**:
 * Rather than implementing tensor contraction from scratch, this kernel leverages the highly
 * optimized `UniversalGemmKernel` as its computational backend.
 *
 * @subsection current_limitations Current Kernel Limitations
 *
 * **Layout Restrictions:**
 * - **Row-Major Only**: All tensors must use row-major memory layout
 * - **Packed Tensors**: Only contiguous/packed tensor layouts supported
 * - **Hardcoded Strides**: stride_A = K_total, stride_B = K_total, stride_E = N_total
 * - **D Tensor Layout**: All D tensors must match E tensor layout (stride_Ds = N_total)
 *
 * **Implementation Constraints:**
 * - **Fixed Stride Calculation**: Strides are automatically calculated and cannot be customized
 * - **No Column-Major**: Column-major or custom stride patterns not supported
 * - **No Strided Access**: Non-contiguous tensor slicing not supported
 *
 * **Future Enhancements:**
 * - Support for arbitrary stride patterns
 * - Column-major and mixed layout support
 * - Non-contiguous tensor operation support
 */

namespace ck_tile {

/// @brief Host arguments for batched tensor contraction operations.
///
/// @par Overview
///     This structure encapsulates all host-side arguments required for batched tensor contraction.
///     It supports arbitrary number of batch dimensions (G), M dimensions, N dimensions, and K
///     dimensions.
///
/// @par Tensor Layout Assumptions
///     - A tensor: [G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
///     - B tensor: [G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
///     - D tensors: [G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...] (auxiliary input tensors)
///     - E tensor: [G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...] (output tensor)
///
/// @tparam NumDTensor Number of D (auxiliary input) tensors. Default is 0.
template <ck_tile::index_t NumDTensor = 0>
struct BatchedContractionHostArgs
{
    /// @brief Constructor for batched contraction host arguments.
    ///
    /// @param a_ptr_ Pointer to input tensor A
    /// @param b_ptr_ Pointer to input tensor B
    /// @param ds_ptr_ Array of pointers to auxiliary input tensors D
    /// @param e_ptr_ Pointer to output tensor E
    /// @param k_batch_ Number of k-splits for split-K batching
    /// @param A_dims_ Dimension vector for tensor A: [G0, G1, ..., M0, M1, ..., K0, K1, ...]
    /// @param B_dims_ Dimension vector for tensor B: [G0, G1, ..., N0, N1, ..., K0, K1, ...]
    /// @param Ds_dims_ Dimension vectors for D tensors: [G0, G1, ..., M0, M1, ..., N0, N1, ...]
    /// @param E_dims_ Dimension vector for tensor E: [G0, G1, ..., M0, M1, ..., N0, N1, ...]
    /// @param A_strides_ Stride vector for tensor A: [G0, G1, ..., M0, M1, ..., K0, K1, ...]
    /// @param B_strides_ Stride vector for tensor B: [G0, G1, ..., N0, N1, ..., K0, K1, ...]
    /// @param Ds_strides_ Stride vectors for D tensors: [G0, G1, ..., M0, M1, ..., N0, N1, ...]
    /// @param E_strides_ Stride vector for tensor E: [G0, G1, ..., M0, M1, ..., N0, N1, ...]
    CK_TILE_HOST
    BatchedContractionHostArgs(
        const void* a_ptr_,
        const void* b_ptr_,
        const std::array<const void*, NumDTensor>& ds_ptr_,
        void* e_ptr_,
        ck_tile::index_t k_batch_,
        const std::vector<ck_tile::index_t>& A_dims_, // [G0, G1, ..., M0, M1, ... , K0, K1, ...]
        const std::vector<ck_tile::index_t>& B_dims_, // [G0, G1, ..., N0, N1, ... , K0, K1, ...]
        const std::array<std::vector<ck_tile::index_t>, NumDTensor>&
            Ds_dims_, // [G0, G1, ..., M0, M1, ... , N0, N1, ...][NumDTensor]
        const std::vector<ck_tile::index_t>& E_dims_, // [G0, G1, ..., M0, M1, ... , N0, N1, ...]

        const std::vector<ck_tile::index_t>& A_strides_, // [G0, G1, ..., M0, M1, ...,K0, K1, ...]
        const std::vector<ck_tile::index_t>& B_strides_, // [G0, G1, ..., N0, N1, ...,K0, K1, ...]
        const std::array<std::vector<ck_tile::index_t>, NumDTensor>&
            Ds_strides_, // [G0, G1, ..., M0, M1, ...,N0, N1, ...]
        const std::vector<ck_tile::index_t>&
            E_strides_) // [G0, G1, ..., M0, M1, ...,N0, N1, ...][NumDTensor]

        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          k_batch(k_batch_),
          A_dims(A_dims_),
          B_dims(B_dims_),
          Ds_dims(Ds_dims_),
          E_dims(E_dims_),
          A_strides(A_strides_),
          B_strides(B_strides_),
          Ds_strides(Ds_strides_),
          E_strides(E_strides_)
    {
    }

    const void* a_ptr;                          ///< Pointer to input tensor A
    const void* b_ptr;                          ///< Pointer to input tensor B
    std::array<const void*, NumDTensor> ds_ptr; ///< Array of pointers to auxiliary input tensors D
    void* e_ptr;                                ///< Pointer to output tensor E
    ck_tile::index_t k_batch;                   ///< Number of k-splits for split-K batching
    const std::vector<ck_tile::index_t>
        A_dims; ///< Dimension vector for tensor A: [G0, G1, ..., M0, M1, ..., K0, K1, ...]
    const std::vector<ck_tile::index_t>
        B_dims; ///< Dimension vector for tensor B: [G0, G1, ..., N0, N1, ..., K0, K1, ...]
    const std::array<std::vector<ck_tile::index_t>, NumDTensor>
        Ds_dims; ///< Dimension vectors for D tensors: [G0, G1, ..., M0, M1, ..., N0, N1, ...]
    const std::vector<ck_tile::index_t>
        E_dims; ///< Dimension vector for tensor E: [G0, G1, ..., M0, M1, ..., N0, N1, ...]
    const std::vector<ck_tile::index_t>
        A_strides; ///< Stride vector for tensor A: [G0, G1, ..., M0, M1, ..., K0, K1, ...]
    const std::vector<ck_tile::index_t>
        B_strides; ///< Stride vector for tensor B: [G0, G1, ..., N0, N1, ..., K0, K1, ...]
    const std::array<std::vector<ck_tile::index_t>, NumDTensor>
        Ds_strides; ///< Stride vectors for D tensors: [G0, G1, ..., M0, M1, ..., N0, N1, ...]
    const std::vector<ck_tile::index_t>
        E_strides; ///< Stride vector for tensor E: [G0, G1, ..., M0, M1, ..., N0, N1, ...]
};

/// @brief Kernel arguments for batched tensor contraction operations.
///
/// @tparam NumDimG Number of batch dimensions
/// @tparam NumDimM Number of M (output row) dimensions
/// @tparam NumDimN Number of N (output column) dimensions
/// @tparam NumDimK Number of K (contraction) dimensions
/// @tparam NumDTensor Number of auxiliary input D tensors. Default is 0.

template <ck_tile::index_t NumDimG,
          ck_tile::index_t NumDimM,
          ck_tile::index_t NumDimN,
          ck_tile::index_t NumDimK,
          ck_tile::index_t NumDTensor = 0>
struct BatchedContractionKernelArgs
{
    const void* a_ptr;                          ///< Pointer to input tensor A
    const void* b_ptr;                          ///< Pointer to input tensor B
    std::array<const void*, NumDTensor> ds_ptr; ///< Array of pointers to auxiliary input tensors D
    void* e_ptr;                                ///< Pointer to output tensor E
    ck_tile::index_t k_batch;                   ///< Number of k-splits for split-K batching

    ck_tile::index_t M_dims[NumDimM]; ///< M dimension sizes: [M0, M1, M2, ..., M_{NumDimM-1}]
    ck_tile::index_t N_dims[NumDimN]; ///< N dimension sizes: [N0, N1, N2, ..., N_{NumDimN-1}]
    ck_tile::index_t K_dims[NumDimK]; ///< K dimension sizes: [K0, K1, K2, ..., K_{NumDimK-1}]
    ck_tile::index_t
        G_dims[NumDimG]; ///< G (batch) dimension sizes: [G0, G1, G2, ..., G_{NumDimG-1}]

    // Batch strides for efficient offset calculation
    ck_tile::index_t batch_stride_A;                          ///< Batch stride for tensor A
    ck_tile::index_t batch_stride_B;                          ///< Batch stride for tensor B
    ck_tile::index_t batch_stride_E;                          ///< Batch stride for tensor E
    std::array<ck_tile::index_t, NumDTensor> batch_stride_Ds; ///< Batch strides for D tensors

    ck_tile::index_t G_total; ///< Total batch size: G0 * G1 * ... * G_{NumDimG-1}
    ck_tile::index_t M_total; ///< Total M dimension: M0 * M1 * ... * M_{NumDimM-1}
    ck_tile::index_t N_total; ///< Total N dimension: N0 * N1 * ... * N_{NumDimN-1}
    ck_tile::index_t K_total; ///< Total K dimension: K0 * K1 * ... * K_{NumDimK-1}

    ck_tile::index_t stride_A; ///< Leading dimension stride for tensor A (row-major: K_total)
    ck_tile::index_t stride_B; ///< Leading dimension stride for tensor B (row-major: K_total)
    std::array<ck_tile::index_t, NumDTensor>
        stride_Ds;             ///< Leading dimension strides for D tensors (row-major: N_total)
    ck_tile::index_t stride_E; ///< Leading dimension stride for tensor E (row-major: N_total)
};

/// @brief GPU kernel for batched tensor contraction operations.
///
/// @par Overview
///     This kernel performs batched tensor contraction operations using the underlying
///     UniversalGemmKernel. It supports arbitrary tensor dimensionalities (G, M, N, K) and
///     processes multiple batch instances in parallel. Each batch performs: E =
///     epilogue_op(contraction(A, B), D0, D1, ...).
///
/// @tparam Problem_ Tensor contraction problem specification defining data types and dimensions
/// @tparam TilePartitioner_ Tile partitioning strategy for workload distribution
/// @tparam GemmPipeline_ GEMM computation pipeline for core matrix operations
/// @tparam EpiloguePipeline_ Epilogue pipeline for post-GEMM operations and tensor fusion

template <typename Problem_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct BatchedContractionKernel
{
    // Type aliases for cleaner code and better readability
    using Problem = ck_tile::remove_cvref_t<Problem_>; ///< Tensor contraction problem specification
    using ADataType =
        ck_tile::remove_cvref_t<typename Problem::ADataType>; ///< Data type for input tensor A
    using BDataType =
        ck_tile::remove_cvref_t<typename Problem::BDataType>; ///< Data type for input tensor B
    using DsDataType =
        ck_tile::remove_cvref_t<typename Problem::DsDataType>; ///< Data types for auxiliary input
                                                               ///< tensors D
    using EDataType =
        ck_tile::remove_cvref_t<typename Problem::EDataType>; ///< Data type for output tensor E

    // Compile-time dimension constants extracted from problem specification
    static constexpr ck_tile::index_t NumDimG = Problem::NumDimG; ///< Number of batch dimensions
    static constexpr ck_tile::index_t NumDimM =
        Problem::NumDimM; ///< Number of M (output row) dimensions
    static constexpr ck_tile::index_t NumDimN =
        Problem::NumDimN; ///< Number of N (output column) dimensions
    static constexpr ck_tile::index_t NumDimK =
        Problem::NumDimK; ///< Number of K (contraction) dimensions
    static constexpr ck_tile::index_t NumDTensor =
        Problem::NumDTensor; ///< Number of auxiliary input D tensors

    // Pipeline and partitioning strategy types
    using TilePartitioner =
        ck_tile::remove_cvref_t<TilePartitioner_>; ///< Tile partitioning strategy for workload
                                                   ///< distribution
    using GemmPipeline = ck_tile::remove_cvref_t<GemmPipeline_>; ///< GEMM computation pipeline
    using EpiloguePipeline =
        ck_tile::remove_cvref_t<EpiloguePipeline_>; ///< Epilogue pipeline for post-GEMM operations

    // Underlying GEMM kernel that performs the actual computation
    using UniversalGemmKernel =
        ck_tile::UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    static constexpr ck_tile::index_t kBlockSize =
        UniversalGemmKernel::kBlockSize; ///< GPU block size inherited from GEMM kernel

    using KernelArgs =
        BatchedContractionKernelArgs<NumDimG, NumDimM, NumDimN, NumDimK, NumDTensor>; ///< Kernel
                                                                                      ///< argument
                                                                                      ///< structure

    /// @brief Returns the kernel name for debugging and profiling purposes.
    /// @return Constant string identifier for this kernel
    CK_TILE_HOST static constexpr auto GetKernelName() { return "batched_contraction_kernel"; }

    /// @brief Validates whether the given kernel arguments are supported.
    /// @param kargs Kernel arguments to validate
    /// @return True if arguments are supported, false otherwise
    /// @details Checks underlying GEMM kernel support and ensures valid batch dimensions
    CK_TILE_HOST static constexpr bool IsSupportedArguments(const KernelArgs& kargs)
    {
        typename UniversalGemmKernel::KernelArgs gemm_kargs{{kargs.a_ptr},
                                                            {kargs.b_ptr},
                                                            kargs.ds_ptr,
                                                            kargs.e_ptr,
                                                            kargs.M_total,
                                                            kargs.N_total,
                                                            kargs.K_total,
                                                            {kargs.stride_A},
                                                            {kargs.stride_B},
                                                            kargs.stride_Ds,
                                                            kargs.stride_E,
                                                            kargs.k_batch};

        return UniversalGemmKernel::IsSupportedArgument(gemm_kargs) && kargs.G_total > 0;
    }

    /// @brief Returns the shared memory size required by the kernel.
    /// @return Shared memory size in bytes
    /// @details Delegates to underlying GEMM kernel's shared memory requirements
    CK_TILE_HOST static constexpr ck_tile::index_t GetSmemSize()
    {
        return UniversalGemmKernel::GetSmemSize();
    }

    /// @brief Returns the GPU block size for kernel launch.
    /// @return 3D block dimensions for GPU kernel execution
    CK_TILE_HOST static constexpr auto GetBlockSize()
    {
        return dim3(UniversalGemmKernel::kBlockSize);
    }

    CK_TILE_HOST static constexpr auto GridSize(const KernelArgs& kargs)
    {
        return dim3(
            TilePartitioner::GridSize(kargs.M_total, kargs.N_total), kargs.G_total, kargs.k_batch);
    }

    CK_TILE_HOST static constexpr KernelArgs
    MakeKernelArgs(const BatchedContractionHostArgs<NumDTensor>& host_args)
    {
        const auto expected_A_dims = NumDimG + NumDimM + NumDimK;
        const auto expected_B_dims = NumDimG + NumDimN + NumDimK;
        const auto expected_E_dims = NumDimG + NumDimM + NumDimN;

        if(host_args.A_dims.size() != expected_A_dims ||
           host_args.A_strides.size() != expected_A_dims)
        {
            throw std::invalid_argument("A dimension size mismatch");
        }
        if(host_args.B_dims.size() != expected_B_dims ||
           host_args.B_strides.size() != expected_B_dims)
        {
            throw std::invalid_argument("B dimension size mismatch");
        }
        if(host_args.E_dims.size() != expected_E_dims ||
           host_args.E_strides.size() != expected_E_dims)
        {
            throw std::invalid_argument("E dimension size mismatch");
        }

        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            if(host_args.Ds_dims[d].size() != expected_E_dims ||
               host_args.Ds_strides[d].size() != expected_E_dims)
            {
                throw std::invalid_argument("D dimension size mismatch");
            }
        }

        KernelArgs kargs;
        kargs.a_ptr   = host_args.a_ptr;
        kargs.b_ptr   = host_args.b_ptr;
        kargs.ds_ptr  = host_args.ds_ptr;
        kargs.e_ptr   = host_args.e_ptr;
        kargs.k_batch = host_args.k_batch;

        // Validate and set G dimensions (must be identical across all tensors)
        for(ck_tile::index_t i = 0; i < NumDimG; ++i)
        {
            // All tensors must have same G dimensions for valid contraction
            if(host_args.A_dims[i] != host_args.B_dims[i] ||
               host_args.A_dims[i] != host_args.E_dims[i])
            {
                throw std::invalid_argument(
                    "All tensors must have identical G dimensions for valid contraction");
            }

            // Store G dimensions (same for all tensors)
            kargs.G_dims[i] = host_args.A_dims[i];
        }

        // Set batch strides from the stride of last G dimension
        kargs.batch_stride_A = host_args.A_strides[NumDimG - 1];
        kargs.batch_stride_B = host_args.B_strides[NumDimG - 1];
        kargs.batch_stride_E = host_args.E_strides[NumDimG - 1];

        for(ck_tile::index_t i = 0; i < NumDimM; ++i)
        {
            kargs.M_dims[i] = host_args.A_dims[NumDimG + i];
            if(kargs.M_dims[i] != host_args.E_dims[NumDimG + i])
            {
                throw std::invalid_argument("M dimension mismatch between A and E tensors");
            }
        }
        for(ck_tile::index_t i = 0; i < NumDimN; ++i)
        {
            kargs.N_dims[i] = host_args.B_dims[NumDimG + i];
            if(kargs.N_dims[i] != host_args.E_dims[NumDimG + NumDimM + i])
            {
                throw std::invalid_argument("N dimension mismatch between B and E tensors");
            }
        }
        for(ck_tile::index_t i = 0; i < NumDimK; ++i)
        {
            kargs.K_dims[i] = host_args.A_dims[NumDimG + NumDimM + i];
            if(kargs.K_dims[i] != host_args.B_dims[NumDimG + NumDimN + i])
            {
                throw std::invalid_argument("K dimension mismatch between A and B tensors");
            }
        }

        // Calculate total dimensions from individual dimension arrays
        kargs.G_total = 1;
        for(ck_tile::index_t i = 0; i < NumDimG; ++i)
        {
            kargs.G_total *= kargs.G_dims[i];
        }

        kargs.M_total = 1;
        for(ck_tile::index_t i = 0; i < NumDimM; ++i)
        {
            kargs.M_total *= kargs.M_dims[i];
        }

        kargs.N_total = 1;
        for(ck_tile::index_t i = 0; i < NumDimN; ++i)
        {
            kargs.N_total *= kargs.N_dims[i];
        }

        kargs.K_total = 1;
        for(ck_tile::index_t i = 0; i < NumDimK; ++i)
        {
            kargs.K_total *= kargs.K_dims[i];
        }

        kargs.stride_A = kargs.K_total;
        kargs.stride_B = kargs.K_total;
        kargs.stride_E = kargs.N_total;

        // Validate D tensors have same G dimensions and set their batch strides
        for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
        {
            for(ck_tile::index_t i = 0; i < NumDimG; ++i)
            {
                if(host_args.Ds_dims[d][i] != host_args.A_dims[i])
                {
                    throw std::invalid_argument(
                        "D tensor G dimensions must match A/B/E tensor G dimensions");
                }
            }
            // Set batch stride for D tensor
            kargs.batch_stride_Ds[d] = host_args.Ds_strides[d][NumDimG - 1];
            kargs.stride_Ds[d]       = kargs.N_total; // D tensors same shape as E
        }

        return kargs;
    }

    CK_TILE_DEVICE void operator()(const KernelArgs& kargs) const
    {

        const auto [iM, iN] =
            TilePartitioner{kargs.M_total, kargs.N_total}.GetOutputTileIndex(blockIdx.x);
        const ck_tile::index_t i_m =
            __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const ck_tile::index_t i_n =
            __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const auto i_batch_flat = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const auto i_splitk     = __builtin_amdgcn_readfirstlane(blockIdx.z);

        // Calculate batch offsets for each tensor
        const auto batch_offset_A = i_batch_flat * kargs.batch_stride_A;
        const auto batch_offset_B = i_batch_flat * kargs.batch_stride_B;
        const auto batch_offset_E = i_batch_flat * kargs.batch_stride_E;

        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.a_ptr) + batch_offset_A;
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.b_ptr) + batch_offset_B;
        EDataType* e_ptr       = static_cast<EDataType*>(kargs.e_ptr) + batch_offset_E;

        std::array<const void*, NumDTensor> ds_batch_ptr;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType           = typename std::tuple_element<i.value, DsDataType>::type;
            const auto batch_offset_D = i_batch_flat * kargs.batch_stride_Ds[i];
            ds_batch_ptr[i] = static_cast<const DDataType*>(kargs.ds_ptr[i]) + batch_offset_D;
        });

        typename UniversalGemmKernel::KernelArgs gemm_kargs{{a_ptr},
                                                            {b_ptr},
                                                            ds_batch_ptr,
                                                            e_ptr,
                                                            kargs.M_total,
                                                            kargs.N_total,
                                                            kargs.K_total,
                                                            {kargs.stride_A},
                                                            {kargs.stride_B},
                                                            kargs.stride_Ds,
                                                            kargs.stride_E,
                                                            kargs.k_batch};

        const typename UniversalGemmKernel::SplitKBatchOffset splitk_batch_offset(gemm_kargs,
                                                                                  i_splitk);

        const ADataType* a_ptr_final = a_ptr + splitk_batch_offset.as_k_split_offset[0];
        const BDataType* b_ptr_final = b_ptr + splitk_batch_offset.bs_k_split_offset[0];
        __shared__ char smem_ptr[GetSmemSize()];

        UniversalGemmKernel::RunGemm({a_ptr_final},
                                     {b_ptr_final},
                                     ds_batch_ptr,
                                     e_ptr,
                                     smem_ptr,
                                     gemm_kargs,
                                     splitk_batch_offset,
                                     i_m,
                                     i_n);
    }
};

} // namespace ck_tile
