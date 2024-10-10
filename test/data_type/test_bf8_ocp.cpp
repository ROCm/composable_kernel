// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bf8_ocp_t;
using ck::f8_convert_rne;
using ck::f8_convert_sr;
using ck::half_t;
using ck::type_convert;

TEST(BF8OCP, NumericLimits) {}

TEST(BF8OCP, ConvertFP32Nearest) {}

TEST(BF8OCP, ConvertFP32Stochastic) {}

TEST(BF8OCP, ConvertFP16Nearest) {}

TEST(BF8OCP, ConvertFP16Stochastic) {}
