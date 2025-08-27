// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "test_topk_softmax.hpp"

int main() { return run_gemm_combinations<ck_tile::bf16_t>("bf16"); }
