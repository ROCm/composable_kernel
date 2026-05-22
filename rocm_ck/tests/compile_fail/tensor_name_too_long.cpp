// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: tensor name exceeds FixedString<16> capacity (15 chars + null).
// Expected error: "FixedString: input exceeds capacity"

#include <rocm_ck/fixed_string.hpp>

using namespace rocm_ck;

constexpr FixedString<16> bad("ABCDEFGHIJKLMNOP"); // 16 chars, needs 17 with null
