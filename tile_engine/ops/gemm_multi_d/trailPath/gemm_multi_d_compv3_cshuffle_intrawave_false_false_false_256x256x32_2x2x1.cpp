
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.


#include "gemm_multi_d_compv3_cshuffle_intrawave_false_false_false.hpp" 


template struct compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 16>;
template struct compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 32, 32, 8>;
template struct compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 16>;
template struct compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 16, 16, 32>;
template struct compv3_cshuffle_intrawave_false_false_false::GemmKernel<256, 256, 32, 2, 2, 1, 4, 64, 16>;
