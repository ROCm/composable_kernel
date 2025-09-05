// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"
#include "mask.hpp"

namespace ck_tile {

std::ostream& operator<<(std::ostream& stream, const fmha_fwd_v3_args::data_type_enum& data_type)
{
    switch(data_type)
    {
    case fmha_fwd_v3_args::data_type_enum::fp16: return stream << "fp16";
    case fmha_fwd_v3_args::data_type_enum::bf16: return stream << "bf16";
    default: return stream << "unknown";
    }
}

std::pair<bool, float> fmha_fwd_v3(const fmha_fwd_v3_args& args, const stream_config& config)
{
    if(args.data_type == fmha_fwd_v3_args::data_type_enum::fp16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::fp16, false, false>;

            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
        }
        else
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::fp16, false, true>;

            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
        }
    }
    else if(args.data_type == fmha_fwd_v3_args::data_type_enum::bf16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, false>;

            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
        }
        else
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, true>;

            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
        }
    }

    return std::make_pair(false, -1.f);
}

} // namespace ck_tile
