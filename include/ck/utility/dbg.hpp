// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace dbg {

#define VAR(v, i) __typeof(i) v=(i)
#define FORE(i, c) for(VAR(i, (c).begin()); i != (c).end(); ++i)

template <typename TH> void _dbg(const char *sdbg, TH h) {
  std::cerr << sdbg << "=" << h << "\n";
}
template <typename TH, typename... TA>
void _dbg(const char *sdbg, TH h, TA... t) {
  while (*sdbg != ',') {
    std::cerr << *sdbg++;
  }
  std::cerr << "=" << h << ",";
  _dbg(sdbg + 1, t...);
}

#define debug(...) _dbg(#__VA_ARGS__, __VA_ARGS__)
#define debugv(x)                                                              \
  {                                                                            \
    {                                                                          \
      std::cerr << #x << " = ";                                                \
      FORE(itt, (x)) std::cerr << *itt << ", ";                                \
      std::cerr << "\n";                                                       \
    }                                                                          \
  }

} // namespace dbg
} // namespace cki

