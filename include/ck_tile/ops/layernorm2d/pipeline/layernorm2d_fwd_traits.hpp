// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

enum class Layernorm2dFusedAddEnum
{
    NO_ADD = 0,
    // fused add before layernorm and store result to global
    PRE_ADD_STORE = 1,
    // fused add before layernorm, but not store result
    PRE_ADD = 2,
};

// clang-format off
template<Layernorm2dFusedAddEnum E> struct Layernorm2dFusedAddEnumName;
template<> struct Layernorm2dFusedAddEnumName<Layernorm2dFusedAddEnum::NO_ADD> { static constexpr const char * name = "no"; };
template<> struct Layernorm2dFusedAddEnumName<Layernorm2dFusedAddEnum::PRE_ADD_STORE> { static constexpr const char * name = "pras"; };
template<> struct Layernorm2dFusedAddEnumName<Layernorm2dFusedAddEnum::PRE_ADD> { static constexpr const char * name = "pra"; };
// clang-format on

enum class Layernorm2dFusedSweepEnum
{
    NO_SWEEP      = 0,
    DYNAMIC_QUANT = 1,
};

// clang-format off
template<Layernorm2dFusedSweepEnum E> struct Layernorm2dFusedSweepEnumName;
template<> struct Layernorm2dFusedSweepEnumName<Layernorm2dFusedSweepEnum::NO_SWEEP> { static constexpr const char * name = "no"; };
template<> struct Layernorm2dFusedSweepEnumName<Layernorm2dFusedSweepEnum::DYNAMIC_QUANT> { static constexpr const char * name = "dquant"; };
// clang-format on

template <bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kTwoPass_,
          Layernorm2dFusedAddEnum kFusedAdd_,
          Layernorm2dFusedSweepEnum kFusedSweep_>
struct Layernorm2dFwdTraits
{
    static constexpr bool kPadN                            = kPadN_;
    static constexpr bool kSaveMeanInvStd                  = kSaveMeanInvStd_;
    static constexpr bool kTwoPass                         = kTwoPass_;
    static constexpr Layernorm2dFusedAddEnum kFusedAdd     = kFusedAdd_;
    static constexpr Layernorm2dFusedSweepEnum kFusedSweep = kFusedSweep_;
};

} // namespace ck_tile
