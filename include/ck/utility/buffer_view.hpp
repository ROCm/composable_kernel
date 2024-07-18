// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/buffer_view_declare.hpp"
#include "ck/utility/buffer_view_impl_generic.hpp"
#include "ck/utility/buffer_view_impl_global.hpp"
#include "ck/utility/buffer_view_impl_lds.hpp"
#include "ck/utility/buffer_view_impl_vgpr.hpp"

namespace ck {

template <AddressSpaceEnum BufferAddressSpace,
          AmdBufferCoherenceEnum Coherence = AmdBufferCoherenceEnum::DefaultCoherence,
          typename T,
          typename BufferSizeType>
__host__ __device__ constexpr auto make_buffer_view(T* p, BufferSizeType buffer_size)
{
    return BufferView<BufferAddressSpace, T, BufferSizeType, true, Coherence>{p, buffer_size};
}

template <
    AddressSpaceEnum BufferAddressSpace,
    AmdBufferCoherenceEnum Coherence = AmdBufferCoherenceEnum::DefaultCoherence,
    typename T,
    typename BufferSizeType,
    typename X,
    typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type = false>
__host__ __device__ constexpr auto
make_buffer_view(T* p, BufferSizeType buffer_size, X invalid_element_value)
{
    return BufferView<BufferAddressSpace, T, BufferSizeType, false, Coherence>{
        p, buffer_size, invalid_element_value};
}

} // namespace ck
