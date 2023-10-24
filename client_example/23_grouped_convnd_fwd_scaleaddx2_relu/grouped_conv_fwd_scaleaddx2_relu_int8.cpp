// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"
#include <tuple>

using InDataType  = int8_t;
using WeiDataType = int8_t;
using OutDataType = int8_t;
// Use std tuple instead ck tuple to avoid clang
// implicit instantiation of undefined template error.
using DDataTypes = std::tuple<float, float>;

#include "grouped_conv_fwd_scaleaddx2_relu.inc"
