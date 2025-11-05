// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"
#include "mask.hpp"

namespace ck_tile {

float fmha_fwd_v3(fmha_fwd_traits traits, fmha_fwd_args args, const ck_tile::stream_config& config)
{
    if(traits.data_type.compare("fp16") == 0)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits = fmha_fwd_v3_kernel_traits<FmhaFwdFp16, false, false>;

            return fmha_fwd_<kernel_traits, ck_tile::gfx950_t>(config, args);
        }
        else
        {
            using kernel_traits = fmha_fwd_v3_kernel_traits<FmhaFwdFp16, false, true>;

            return fmha_fwd_<kernel_traits, ck_tile::gfx950_t>(config, args);
        }
    }
    else if(traits.data_type.compare("bf16") == 0)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits = fmha_fwd_v3_kernel_traits<FmhaFwdBf16, false, false>;

            return fmha_fwd_<kernel_traits, ck_tile::gfx950_t>(config, args);
        }
        else
        {
            using kernel_traits = fmha_fwd_v3_kernel_traits<FmhaFwdBf16, false, true>;

            return fmha_fwd_<kernel_traits, ck_tile::gfx950_t>(config, args);
        }
    }

    return -1.;
}

} // namespace ck_tile
