// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include <type_traits>

namespace ck_tile {

namespace detail {

// Helper method to automatically determine compute type
// Selects the largest type of the two. If both of them are packed data types, defaults to fp8.
template <typename ADataType, typename BDataType>
struct auto_compute_type
{
    using LargestInputType = largest_type_t<ADataType, BDataType>;

    // Sanity check: there are no packed types larger than 1 byte yet, but if we add them
    // this logic should change
    static_assert(!is_packed_type_v<LargestInputType> || sizeof(LargestInputType) == sizeof(fp8_t));

    using type = std::conditional_t<is_packed_type_v<LargestInputType>, fp8_t, LargestInputType>;
};

// Helper method to determine compute type, defaulting an explicitly passed-in compute type
template <typename ComputeDataType, typename ADataType, typename BDataType>
struct mixed_prec_compute_type
{
    using type = std::conditional_t<std::is_void_v<ComputeDataType>,
                                    typename auto_compute_type<ADataType, BDataType>::type,
                                    ComputeDataType>;
};

} // namespace detail

template <typename ComputeDataType, typename ADataType, typename BDataType>
using mixed_prec_compute_type_t =
    typename detail::mixed_prec_compute_type<ComputeDataType, ADataType, BDataType>::type;

// Helper method to determine compute type, defaulting to input data type
// If "ThisDataType" is packed (4-bit), will default to "OtherDataType". If both are packed,
// ComputeDataType is used.
template <typename ThisDataType, typename OtherDataType, typename ComputeDataType>
using mixed_prec_compute_type_from_input_t = std::conditional_t<
    is_packed_type_v<ThisDataType>,
    std::conditional_t<is_packed_type_v<OtherDataType>, ComputeDataType, OtherDataType>,
    ThisDataType>;

} // namespace ck_tile
