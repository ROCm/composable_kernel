// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_universal_mem_instance_common.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;

template float gemm_<trait_<ck_tile::bf16_t,
                            ck_tile::bf16_t,
                            float,
                            ck_tile::bf16_t,
                            Row,
                            Row,
                            Row,
                            128,
                            32,
                            64,
                            4,
                            1,
                            1,
                            32,
                            32,
                            8,
                            false,
                            false,
                            false>>(const A&, const S&);
