// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_universal_comp_instance_common.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
template float gemm_<gemm_traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, ck_tile::bf16_t, Row, Col, Row, 256, 256, 32, 2, 2, 1, 32, 32, 16, false, false, false>>(const A&, const S&);
// clang-format on
