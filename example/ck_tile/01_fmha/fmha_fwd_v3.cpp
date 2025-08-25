// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"
#include "mask.hpp"

#define DEBUG_DTYPE_FP16 0
#define DEBUG_DTYPE_BF16 1
#define DEBUG_MASK_NONE 0
#define DEBUG_MASK_CAUSAL 1

#define DEBUG_SINGLE_INST 0
#define DEBUG_SINGLE_INST_DTYPE DEBUG_DTYPE_BF16
#define DEBUG_SINGLE_INST_MASK DEBUG_MASK_NONE

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
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_FP16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_NONE)
            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
#endif
        }
        else
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::fp16, false, true>;
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_FP16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_CAUSAL)
            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
#endif
        }
    }
    else if(args.data_type == fmha_fwd_v3_args::data_type_enum::bf16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, false>;
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_BF16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_NONE)
            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
#endif
        }
        else
        {
            using kernel_traits =
                fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, true>;

#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_BF16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_CAUSAL)
            return fmha_fwd_v3_kernel_dispatch<kernel_traits>(args, config);
#endif
        }
    }

    return std::make_pair(false, -1.f);
}

} // namespace ck_tile
