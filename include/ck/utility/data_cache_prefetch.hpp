// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/amd_buffer_coherence.hpp"

namespace ck {

template <AmdBufferCoherenceEnum Coherence_ = AmdBufferCoherenceEnum::DefaultCoherence>
struct GlobalPrefetchDataOp
{
    // addr needs to point to global memory!
    __device__ __forceinline__ void operator()([[maybe_unused]] const void* addr) const
    {
#if defined(__gfx125__)
        __builtin_amdgcn_global_prefetch(addr, static_cast<index_t>(Coherence_));
#endif
    }
};

template <AmdBufferCoherenceEnum Coherence_ = AmdBufferCoherenceEnum::DefaultCoherence>
struct FlatPrefetchDataOp
{
    __device__ __forceinline__ void operator()([[maybe_unused]] const void* addr) const
    {
#if defined(__gfx125__)
        __builtin_amdgcn_flat_prefetch(addr, static_cast<index_t>(Coherence_));
#endif
    }
};

} // namespace ck
