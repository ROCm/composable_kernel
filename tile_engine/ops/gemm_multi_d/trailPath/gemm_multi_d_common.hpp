// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

// Data types
using ADataType = ck_tile::half_t;
using BDataType = ck_tile::half_t;
using AccDataType = float;
using D0DataType = ck_tile::half_t;
using D1DataType = ck_tile::half_t;
using DsDataType = ck_tile::tuple<D0DataType, D1DataType>;
using EDataType = ck_tile::half_t;


// Layout configurations
using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using D0Layout = ck_tile::tensor_layout::gemm::RowMajor;
using D1Layout = ck_tile::tensor_layout::gemm::RowMajor;
using DsLayout = ck_tile::tuple<D0Layout, D1Layout>;
using ELayout = ck_tile::tensor_layout::gemm::RowMajor;

