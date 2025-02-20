// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_universal_comp_v3_instance_common.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
template float gemm_<gemm_traits_<ck_tile::bf8_t, ck_tile::bf8_t, float, ck_tile::half_t, Col, Row, Row, 256, 256, 64, 2, 2, 1, 32, 32, 16, false, false, false>>(const A&, const S&);
// clang-format on
