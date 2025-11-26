// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"

namespace ck_tile {

using kernel_traits =
    fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, true>;

INST_FMHA_FWD_V3_DISPATCH(kernel_traits)

} // namespace ck_tile
