// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/numeric.hpp"

#include <type_traits>

namespace ck_tile {

namespace detail {
template <class T>
struct t
{
    using type = T;
};

// Helper method to automatically determine compute type
// Selects the largest type of the two. If both of them are packed data types, defaults to fp8.
template <typename ADataType, typename BDataType>
struct auto_compute_type
{
    static constexpr auto Resolve()
    {
        using LargestInputType = largest_type_t<ADataType, BDataType>;
        if constexpr(is_packed_type_v<LargestInputType>)
        {
            return t<fp8_t>{};
        }
        else
        {
            return t<LargestInputType>{};
        }
    }

    using type = typename decltype(Resolve())::type;
};

// Helper method to determine compute type, defaulting an explicitly passed-in compute type
template <typename ComputeDataType, typename ADataType, typename BDataType>
struct mfma_compute_type
{
    using type = std::conditional_t<std::is_void_v<ComputeDataType>,
                                    typename auto_compute_type<ADataType, BDataType>::type,
                                    ComputeDataType>;
};

}; // namespace detail

template <typename ComputeDataType, typename ADataType, typename BDataType>
using mfma_compute_type_t =
    typename detail::mfma_compute_type<ComputeDataType, ADataType, BDataType>::type;

// Helper method to determine compute type, defaulting to input data type
// If "ThisDataType" is packed (4-bit), will default to "OtherDataType". If both are packed,
// ComputeDataType is used.
template <typename ThisDataType, typename OtherDataType, typename ComputeDataType>
using mfma_compute_type_from_input_t = std::conditional_t<
    is_packed_type_v<ThisDataType>,
    std::conditional_t<is_packed_type_v<OtherDataType>, ComputeDataType, OtherDataType>,
    ThisDataType>;

} // namespace ck_tile
