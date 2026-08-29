// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/e8m0.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "ck_tile/core/numeric/tfloat32.hpp"

#include <string>
#include <type_traits>

namespace ck_tile {

// clang-format off
template <typename T> struct DataTypeTraits;
template <> struct DataTypeTraits<float> { static constexpr const char * name = "fp32"; };
template <> struct DataTypeTraits<double> { static constexpr const char * name = "fp64"; };
template <> struct DataTypeTraits<int32_t> { static constexpr const char * name = "int32"; };
template <> struct DataTypeTraits<fp16_t> { static constexpr const char * name = "fp16"; };
template <> struct DataTypeTraits<bf16_t> { static constexpr const char * name = "bf16"; };
template <> struct DataTypeTraits<fp8_t> { static constexpr const char * name = "fp8"; };
template <> struct DataTypeTraits<bf8_t> { static constexpr const char * name = "bf8"; };
template <> struct DataTypeTraits<int8_t> { static constexpr const char * name = "int8"; };
template <> struct DataTypeTraits<pk_int4_t> { static constexpr const char * name = "pk_int4"; };
template <> struct DataTypeTraits<pk_fp4_t> { static constexpr const char * name = "pk_fp4"; };
template <> struct DataTypeTraits<pk_fp6x16_t> { static constexpr const char * name = "pk_fp6x16"; };
template <> struct DataTypeTraits<pk_bf6x16_t> { static constexpr const char * name = "pk_bf6x16"; };
template <> struct DataTypeTraits<pk_fp4_raw_t> { static constexpr const char * name = "pk_fp4_raw"; };
template <> struct DataTypeTraits<e8m0_t> { static constexpr const char * name = "e8m0"; };
template <> struct DataTypeTraits<tf32_t>{ static constexpr const char* name = "tf32"; };

template <memory_operation_enum MemOp> struct memOpToStr;
template <> struct memOpToStr<memory_operation_enum::set> { static constexpr const char * name = "set"; };
template <> struct memOpToStr<memory_operation_enum::atomic_add> { static constexpr const char * name = "atomic_add"; };
template <> struct memOpToStr<memory_operation_enum::atomic_max> { static constexpr const char * name = "atomic_max"; };
template <> struct memOpToStr<memory_operation_enum::add> { static constexpr const char * name = "add"; };
// clang-format on

template <typename ADataType_, typename BDataType_>
std::string gemm_prec_str()
{
    std::string base_str = std::string(DataTypeTraits<ADataType_>::name);
    if(!std::is_same_v<ADataType_, BDataType_>)
    {
        base_str += "_" + std::string(DataTypeTraits<BDataType_>::name);
    }
    return base_str;
}

template <memory_operation_enum MemOp_>
std::string mem_op_string()
{
    return std::string(memOpToStr<MemOp_>::name);
}

} // namespace ck_tile
