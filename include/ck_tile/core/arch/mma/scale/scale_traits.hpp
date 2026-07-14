// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/mma/mma_data_format.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_params.hpp"

#include <type_traits>

namespace ck_tile::core::arch::mma::scale::detail {

template <typename DataType, int32_t ScaleFlag>
inline constexpr bool is_valid_ScaleVecType()
{
    [[maybe_unused]] constexpr int32_t data_type_check = PackedDataTypeToFlag_v<DataType>;

    if constexpr(std::is_same_v<DataType, pk_fp4_t>)
    {
        return ScaleFlag == static_cast<int32_t>(ScaleDataType::E8M0) ||
               ScaleFlag == static_cast<int32_t>(ScaleDataType::E5M3) ||
               ScaleFlag == static_cast<int32_t>(ScaleDataType::E4M3);
    }
    else
    {
        return ScaleFlag == static_cast<int32_t>(ScaleDataType::E8M0);
    }
}

template <typename ADataType, typename BDataType, int32_t ScaleAFlag, int32_t ScaleBFlag>
inline constexpr bool is_legal_combination =
    is_valid_ScaleVecType<ADataType, ScaleAFlag>() &&
    is_valid_ScaleVecType<BDataType, ScaleBFlag>() &&
    (!(std::is_same_v<ADataType, pk_fp4_t> && std::is_same_v<BDataType, pk_fp4_t>) ||
     ScaleAFlag == ScaleBFlag);

} // namespace ck_tile::core::arch::mma::scale::detail
