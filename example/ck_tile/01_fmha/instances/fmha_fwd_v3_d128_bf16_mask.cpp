// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"

using kernel_traits = ck_tile::fmha_fwd_v3_kernel_traits<FmhaFwdBf16, false, true>;

INST_FMHA_FWD_V3_DISPATCH(kernel_traits)
