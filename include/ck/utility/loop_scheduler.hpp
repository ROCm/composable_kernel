// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#ifndef __HIPCC_RTC__
#include <ostream>
#endif

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

enum struct LoopScheduler
{
    Default,
    Interwave,
};

constexpr LoopScheduler make_default_loop_scheduler()
{
#if CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING
    return LoopScheduler::Interwave;
#else
    return LoopScheduler::Default;
#endif // if CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING
}

} // namespace ck

#ifndef __HIPCC_RTC__
inline std::ostream& operator<<(std::ostream& os, const ck::LoopScheduler& s)
{
    switch(s)
    {
    case ck::LoopScheduler::Default: os << "Default"; break;
    case ck::LoopScheduler::Interwave: os << "Interwave"; break;
    default: os << "";
    }
    return os;
}
#endif
