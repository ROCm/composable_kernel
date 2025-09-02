// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"

namespace ck_tile {

using kernel_traits =
    fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::fp16, false, false>;

INST_FMHA_FWD_V3_DISPATCH(kernel_traits)

} // namespace ck_tile
