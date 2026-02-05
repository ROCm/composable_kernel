// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_shape.hpp"

namespace ck_tile {

template <typename XDataType_, typename ComputeDataType_, typename YDataType_, typename BlockShape_>
struct MHCProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;

    // PhiDataType is the same as XDataType for the weight matrix
    using PhiDataType = XDataType;

    // BlockGemm compatibility - map our types to BlockGemm's expected types
    using ADataType = XDataType;       // Input matrix A
    using BDataType = PhiDataType;     // Weight matrix B (phi)
    using CDataType = ComputeDataType; // Output/accumulator matrix C

    // BlockGemmShape with kM, kN, kK members for BlockGemm
    // Use supported warp gemm configuration for float32: 32x32x8
    // We'll use 2 warps in M and 1 warp in N to get 64x32 block
    using BlockGemmShape =
        TileGemmShape<sequence<64, 32, 8>,  // BlockTile (M, N, K)
                      sequence<2, 1, 1>,    // BlockWarps (2 warps in M, 1 in N, 1 in K)
                      sequence<32, 32, 8>>; // WarpTile (matches available float32 MFMA)

    // Layout types for BlockGemm
    using ALayout = ck_tile::tensor_layout::gemm::RowMajor; // x is row-major [B, nC]
    using BLayout =
        ck_tile::tensor_layout::gemm::ColumnMajor; // phi treated as column-major for V1 pipeline
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor; // output is row-major

    // For GEMM pipeline compatibility
    using AsDataTypeTuple = tuple<ADataType>;
    using BsDataTypeTuple = tuple<BDataType>;
    using AsLayoutTuple   = tuple<ALayout>;
    using BsLayoutTuple   = tuple<BLayout>;

    using AElementWise = identity;
    using BElementWise = identity;

    static constexpr bool TransposeC = false;
    static constexpr bool kPadM      = true; // Enable padding to help with boundary conditions
    static constexpr bool kPadN      = true; // Enable padding
    static constexpr bool kPadK      = true; // Enable padding
    static constexpr bool Preshuffle = false;

    static constexpr auto Scheduler        = GemmPipelineScheduler::Intrawave;
    static constexpr index_t NumWaveGroups = 1;

    static constexpr index_t VectorLoadSize = 16;
    static constexpr index_t VectorSizeA    = 4;
    static constexpr index_t VectorSizeB    = 4;

    // kBlockSize for BlockGemm compatibility
    static constexpr index_t kBlockSize = BlockShape::BlockSize;

    // Additional traits required by v3 pipeline
    static constexpr bool DoubleSmemBuffer      = true; // Enable double buffering for multi-block
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool FixedVectorSize       = false;

    struct Traits
    {
        static constexpr bool UsePersistentKernel = false;
    };

    CK_TILE_HOST static const std::string GetName() { return "MHCProblem"; }
};

} // namespace ck_tile
