// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_mx_common.hpp"

using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
#if 1
// XXX: MX-native GEMM kernel will work with e8m0_bexp_t scale type
using XDataType = float;
#else
using XDataType = ck::e8m0_bexp_t;
#endif
using AccDataType      = float;
using CShuffleDataType = float;
using CDataType        = float;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using CElementOp = PassThrough; // elementwise transformation for C matrix

constexpr ck::index_t mx_vector_size = 128; // scaling block size

int main(int argc, char* argv[])
{
    return run_mx_gemm_example<ADataType,
                               BDataType,
                               XDataType,
                               CDataType,
                               ALayout,
                               BLayout,
                               CLayout,
                               CElementOp,
                               AccDataType,
                               CShuffleDataType,
                               mx_vector_size>(argc, argv)
               ? 0
               : -1;
}
