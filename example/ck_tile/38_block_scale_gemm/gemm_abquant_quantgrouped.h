// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "gemm_utils.hpp"

#if defined(CK_USE_GFX950)
template <typename T, bool TransposeC = true>
using GemmConfig = GemmConfigEightWarps<T, TransposeC>;
template <typename T, bool TransposeC = true>
using GemmConfigPrefill = GemmConfigPreshuffleBEightWarps<T, TransposeC>;
#else
template <typename T, bool TransposeC = true>
using GemmConfig = GemmConfigABQuantPrefill<T, TransposeC>;
template <typename T, bool TransposeC = true>
using GemmConfigPrefill = GemmConfigPreshuffleB_ABQuant_Prefill<T, TransposeC>;
#endif
