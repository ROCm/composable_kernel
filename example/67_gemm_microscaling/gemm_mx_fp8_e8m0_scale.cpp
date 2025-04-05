// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_mx_common.hpp"

using ADataType = ck::f8_t;
using BDataType = ck::f8_t;

using XDataType = ck::e8m0_bexp_t;

using CDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = CDataType;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough; // elementwise transformation for A matrix
using BElementOp = PassThrough; // elementwise transformation for B matrix
using CElementOp = PassThrough; // elementwise transformation for C matrix

constexpr ck::index_t mx_vector_size = 32; // scaling block size

int main(int argc, char* argv[])
{
    return run_mx_gemm_example<ADataType,
                               BDataType,
                               XDataType,
                               CDataType,
                               ALayout,
                               BLayout,
                               CLayout,
                               AElementOp,
                               BElementOp,
                               CElementOp,
                               AccDataType,
                               CShuffleDataType,
                               mx_vector_size>(argc, argv)
               ? 0
               : -1;
}
