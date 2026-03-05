// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::core::arch::mma {

/**
 * @enum SparseCompressionIndex
 * @brief Indicates which set of sparse-indices within a VGPR starting at srcC
 * containing 8-bits (for 16-bit source data) or 16-bits (for 8-bit source data)
 * of index information for a lane. \see DefaultSparseMfmaCtrlFlags
 */
enum struct SparseCompressionIndex : int
{
    FIRST  = 0, // Uses bits  [7:0] or [15..0], for 16 and 8 bit data respectively
    SECOND = 1, // Uses bits [15:8] or [31:16], for 16 and 8 bit data respectively
    THIRD  = 2, // Uses bits [23:16]
    FOURTH = 3, // Uses bits [31:24]
};

namespace sparse::detail {

/**
 * @struct BuiltinParams
 * @brief Translates the SparseCompressionIndex to the correct CBSZ and ABID pairs for sparse
 * builtins. The actual behavior of the builtin depends on the input data type: 16-bit source data:
 * If CBSZ=0, ABID selects one of four 8-bit sets of sparse-indices within a VGPR starting at srcC
 * containing 8-bits of index information for a lane. If CBSZ!=0 the very first is selected
 * (VGPR[srcC][7..0]).
 *
 * 8-bit source data:
 * If CBSZ=0, ABID selects one of two 16-bit sets of sparse-indices within a VGPR starting at srcC
 * containing 16-bits of index information for a lane. If CBSZ!=0; the very first is selected
 * (VGPR[srcC][15..0]).
 */
struct BuiltinParams
{
    int UseFirstIndex;       // CBSZ
    int ByteIndexToOverride; // ABID
};

template <SparseCompressionIndex Idx>
static constexpr BuiltinParams getBuiltinParams()
{
    BuiltinParams params;
    if constexpr(Idx == SparseCompressionIndex::FIRST)
    {
        params.UseFirstIndex       = 1;
        params.ByteIndexToOverride = 0;
    }
    else
    {
        params.UseFirstIndex       = 0;
        params.ByteIndexToOverride = static_cast<int>(Idx);
    }
    return params;
}

} // namespace sparse::detail

} // namespace ck_tile::core::arch::mma

// Include sparse MFMA traits and architecture-specific implementations
#include "ck_tile/core/arch/mma/sparse/mfma/sparse_gfx9.hpp"
#include "ck_tile/core/arch/mma/sparse/wmma/sparse_gfx12.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_transforms.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_selector.hpp"
