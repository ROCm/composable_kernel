// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_universal_mem_instance_common.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
template float gemm_<gemm_traits_<ck_tile::half_t, ck_tile::half_t, float, ck_tile::half_t, Row, Col, Row, 128, 32, 64, 4, 1, 1, 32, 32, 8, false, false, false>>(const A&, const S&);
// clang-format on
