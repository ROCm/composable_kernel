// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/ck.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Bilinear = ck::tensor_operation::element_wise::Bilinear;
using Scale    = ck::tensor_operation::element_wise::Scale;

enum struct ContractionMatrixLayout
{
    MK_KN_MN_MN, // 0
    MK_NK_MN_MN, // 1
    KM_KN_MN_MN, // 2
    KM_NK_MN_MN, // 3
};

enum struct ContractionDataType
{
    F32_F32_F32_F32, // 0
    F64_F64_F64_F64, // 1
};

inline void collect_index_params(char* argv[],
                                 std::vector<ck::index_t>& params,
                                 const ck::index_t from,
                                 const ck::index_t num)
{
    for(ck::index_t p = from; p < from + num; p++)
        params.push_back(std::stoi(argv[p]));
}

// Defualt strides for row-major: {Dim1 * Dim2 * Dim3, Dim2 * Dim3, Dim3, 1}
// Defualt strides for column-major: {Dim1, 1, Dim0 * Dim1 * Dim3, Dim0 * Dim1}
inline void
assign_default_strides(Row, std::vector<ck::index_t>& strides, std::vector<ck::index_t> dims)
{
    strides = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

inline void
assign_default_strides(Col, std::vector<ck::index_t>& strides, std::vector<ck::index_t> dims)
{
    strides = {dims[1], 1, dims[0] * dims[1] * dims[3], dims[0] * dims[1]};
}
