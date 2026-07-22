// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_contraction/pipeline/batched_contraction_problem.hpp"

namespace ck_tile {

namespace detail {

template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename EDataType_,
          ck_tile::index_t NumDimG_,
          ck_tile::index_t NumDimM_,
          ck_tile::index_t NumDimN_,
          ck_tile::index_t NumDimK_>
using BatchedContractionMultiABDBase = BatchedContractionProblem<
    ck_tile::remove_cvref_t<std::tuple_element_t<0, ck_tile::remove_cvref_t<AsDataType_>>>,
    ck_tile::remove_cvref_t<std::tuple_element_t<0, ck_tile::remove_cvref_t<BsDataType_>>>,
    DsDataType_,
    EDataType_,
    NumDimG_,
    NumDimM_,
    NumDimN_,
    NumDimK_,
    ck_tile::remove_cvref_t<DsDataType_>::size()>;

} // namespace detail

/// @brief Problem specification for batched tensor contraction with multiple A, B, and D tensors.
///
/// @par Overview
///     Extends BatchedContractionProblem to support multiple input A and B tensors (tuples)
///     that are fused via user-provided elementwise operations during the load phase.
///     Multiple D tensors are fused with the contraction result via the epilogue.
///
/// @par Mathematical Formulation
///     fused_A[G,M,K] = a_element_op(A0[G,M,K], A1[G,M,K], ...)
///     fused_B[G,N,K] = b_element_op(B0[G,N,K], B1[G,N,K], ...)
///     C[G,M,N] = sum_K fused_A[G,M,K] * fused_B[G,N,K]
///     E[G,M,N] = cde_element_op(C[G,M,N], D0[G,M,N], D1[G,M,N], ...)
///
/// @tparam AsDataType_ Tuple of data types for A tensors, e.g. tuple<half_t, float>
/// @tparam BsDataType_ Tuple of data types for B tensors, e.g. tuple<half_t>
/// @tparam DsDataType_ Tuple of data types for D tensors, e.g. tuple<half_t>
/// @tparam EDataType_  Data type for output tensor E
/// @tparam NumDimG_    Number of batch (G) dimensions
/// @tparam NumDimM_    Number of M dimensions
/// @tparam NumDimN_    Number of N dimensions
/// @tparam NumDimK_    Number of K (contraction) dimensions
template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename EDataType_,
          ck_tile::index_t NumDimG_,
          ck_tile::index_t NumDimM_,
          ck_tile::index_t NumDimN_,
          ck_tile::index_t NumDimK_>
struct BatchedContractionMultiABDProblem : detail::BatchedContractionMultiABDBase<AsDataType_,
                                                                                  BsDataType_,
                                                                                  DsDataType_,
                                                                                  EDataType_,
                                                                                  NumDimG_,
                                                                                  NumDimM_,
                                                                                  NumDimN_,
                                                                                  NumDimK_>
{
    using Base = detail::BatchedContractionMultiABDBase<AsDataType_,
                                                        BsDataType_,
                                                        DsDataType_,
                                                        EDataType_,
                                                        NumDimG_,
                                                        NumDimM_,
                                                        NumDimN_,
                                                        NumDimK_>;

    using AsDataType = ck_tile::remove_cvref_t<AsDataType_>;
    using BsDataType = ck_tile::remove_cvref_t<BsDataType_>;
    using DsDataType = ck_tile::remove_cvref_t<DsDataType_>;
    using EDataType  = ck_tile::remove_cvref_t<EDataType_>;

    static_assert(is_detected<is_tuple, AsDataType>::value,
                  "AsDataType must be a tuple of data types");
    static_assert(is_detected<is_tuple, BsDataType>::value,
                  "BsDataType must be a tuple of data types");
    static_assert(is_detected<is_tuple, DsDataType>::value,
                  "DsDataType must be a tuple of data types");

    static constexpr ck_tile::index_t NumATensor = AsDataType::size();
    static constexpr ck_tile::index_t NumBTensor = BsDataType::size();
    static constexpr ck_tile::index_t NumDTensor = DsDataType::size();

    static_assert(NumATensor >= 1, "At least one A tensor is required");
    static_assert(NumBTensor >= 1, "At least one B tensor is required");
};

} // namespace ck_tile
