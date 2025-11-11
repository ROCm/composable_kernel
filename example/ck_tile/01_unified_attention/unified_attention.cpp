// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"
#include "mask.hpp"

namespace ck_tile {

std::ostream& operator<<(std::ostream& stream,
                         const unified_attention_args::data_type_enum& data_type)
{
    switch(data_type)
    {
    case unified_attention_args::data_type_enum::fp16: return stream << "fp16";
    case unified_attention_args::data_type_enum::bf16: return stream << "bf16";
    default: return stream << "unknown";
    }
}

std::pair<bool, float> unified_attention(const unified_attention_args& args,
                                         const stream_config& config)
{
    if(args.data_type == unified_attention_args::data_type_enum::fp16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits =
                unified_attention_kernel_traits<unified_attention_args::data_type_enum::fp16,
                                                false>;

            return unified_attention_kernel_dispatch<kernel_traits>(args, config);
        }
        else
        {
            using kernel_traits =
                unified_attention_kernel_traits<unified_attention_args::data_type_enum::fp16, true>;

            return unified_attention_kernel_dispatch<kernel_traits>(args, config);
        }
    }
    else if(args.data_type == unified_attention_args::data_type_enum::bf16)
    {
        if(args.mask_type == static_cast<int>(mask_enum::no_mask))
        {
            using kernel_traits =
                unified_attention_kernel_traits<unified_attention_args::data_type_enum::bf16,
                                                false>;

            return unified_attention_kernel_dispatch<kernel_traits>(args, config);
        }
        else
        {
            using kernel_traits =
                unified_attention_kernel_traits<unified_attention_args::data_type_enum::bf16, true>;

            return unified_attention_kernel_dispatch<kernel_traits>(args, config);
        }
    }

    return std::make_pair(false, -1.f);
}

} // namespace ck_tile
