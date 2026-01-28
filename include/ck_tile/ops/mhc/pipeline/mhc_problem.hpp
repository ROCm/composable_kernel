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
    // BlockGemm expects windows to match exactly: A[kM, kK], B[kK, kN]
    // Our windows: x[16, 256], phi[256, 16]
    // Try matching to warp gemm size: kM=16, kN=16, kK=16
    // We'll need to iterate over K dimension
    using BlockGemmShape = MHCGemmShape<16, 16, 16>;

    // Keep original BlockShape for other uses
    // using BlockShape is already defined above

    // Layout types for BlockGemm
    using ALayout = ck_tile::tensor_layout::gemm::RowMajor; // x is row-major [1, nC]
    using BLayout = ck_tile::tensor_layout::gemm::RowMajor; // phi is row-major [nC, n]
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor; // output is row-major

    // kBlockSize for BlockGemm compatibility
    static constexpr index_t kBlockSize = BlockShape::BlockSize;
};

} // namespace ck_tile
