// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {
enum struct MatrixALayout
{
    MK, // Row-major layout for matrix A (default)
    KM  // Column-major layout for matrix A
};

enum struct MatrixBLayout
{
    NK, // Row-major layout for matrix B (default)
    KN  // Column-major layout for matrix B
};

enum struct MatrixCLayout
{
    MN, // Row-major layout for matrix C (default)
    NM  // Column-major layout for matrix C
};
} // namespace ck_tile
