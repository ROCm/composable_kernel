// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"

namespace ck_tile {

// clang-format off
template <typename T> struct typeToStr;
template <> struct typeToStr<float> { static constexpr const char * name = "fp32"; };
template <> struct typeToStr<fp16_t> { static constexpr const char * name = "fp16"; };
template <> struct typeToStr<bf16_t> { static constexpr const char * name = "bf16"; };
template <> struct typeToStr<fp8_t> { static constexpr const char * name = "fp8"; };
template <> struct typeToStr<bf8_t> { static constexpr const char * name = "bf8"; };
template <> struct typeToStr<int8_t> { static constexpr const char * name = "int8"; };
template <> struct typeToStr<pk_int4_t> { static constexpr const char * name = "pk_int4"; };

template <memory_operation_enum MemOp> struct memOpToStr;
template <> struct memOpToStr<memory_operation_enum::set> { static constexpr const char * name = "set"; };
template <> struct memOpToStr<memory_operation_enum::atomic_add> { static constexpr const char * name = "atomic_add"; };
template <> struct memOpToStr<memory_operation_enum::atomic_max> { static constexpr const char * name = "atomic_max"; };
template <> struct memOpToStr<memory_operation_enum::add> { static constexpr const char * name = "add"; };
// clang-format on

template <typename ADataType_, typename BDataType_>
std::string gemm_prec_str()
{
    std::string base_str = std::string(typeToStr<ADataType_>::name);
    if(!std::is_same_v<ADataType_, BDataType_>)
    {
        base_str += "_" + std::string(typeToStr<BDataType_>::name);
    }
    return base_str;
}

template <memory_operation_enum MemOp_>
std::string mem_op_string()
{
    return std::string(memOpToStr<MemOp_>::name);
}

} // namespace ck_tile
