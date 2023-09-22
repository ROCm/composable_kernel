
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#ifdef __HIPCC_RTC__
namespace std {
template <bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;
} // namespace std
#endif
namespace ck {

template <bool B, typename T = void> using enable_if = std::enable_if<B, T>;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

} // namespace ck

#pragma clang diagnostic pop
