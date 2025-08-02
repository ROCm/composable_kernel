
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.


#include "gemm_multi_d_mem_cshuffle_intrawave_false_false_false.hpp" 


template struct mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 32>;
template struct mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 16>;
template struct mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 16, 16, 16>;
template struct mem_cshuffle_intrawave_false_false_false::GemmKernel<256, 128, 32, 1, 4, 1, 32, 32, 8>;
