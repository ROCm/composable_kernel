// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "test_gemm_streamk.hpp"
#include "test_gemm_streamk_types.hpp"
#include "test_gemm_streamk_util.hpp"
#include "gtest/gtest.h"

// This is the base class for all stream-k tests
#define STREAM_K_TEST_CLASS_BASE TestCkTileStreamK

// Macros to help generate test suite names from the parameters given
#define CONCATENATE_TEST_SUITE_NAME(PREFIX, TEST_PARAMS) PREFIX##_##TEST_PARAMS
// Helper macro to expand the arguments before passing them to CONCATENATE_TEST_SUITE_NAME
#define MAKE_TEST_SUITE_NAME_INTERNAL(TEST_BASE_NAME, TEST_PARAMS) CONCATENATE_TEST_SUITE_NAME(TEST_BASE_NAME, TEST_PARAMS)

// Final macro to be used to create the test suite name from the base class name and the test parameters
#define MAKE_TEST_SUITE_NAME(TEST_PARAMS) MAKE_TEST_SUITE_NAME_INTERNAL(STREAM_K_TEST_CLASS_BASE, TEST_PARAMS)

// Macro to declare a test suite with the given name and parameters, based on the base test class
#define DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS)  \
    template <typename Tuple>                                      \
    class TEST_SUITE_NAME : public STREAM_K_TEST_CLASS_BASE<Tuple> {};     \
    TYPED_TEST_SUITE(TEST_SUITE_NAME, TEST_SUITE_PARAMS);

