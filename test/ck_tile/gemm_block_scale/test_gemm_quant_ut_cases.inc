// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// AQuant tests
TYPED_TEST(TestCkTileGemmAQuant, AQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

// BQuant tests
TYPED_TEST(TestCkTileGemmBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

// BQuant tests
TYPED_TEST(TestCkTileGemmPreshuffleBBQuant, BQuantGroupedTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
// RowColQuant tests
TYPED_TEST(TestCkTileGemmRowColQuant, RowColQuantTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

// TensorQuant tests
TYPED_TEST(TestCkTileGemmTensorQuant, TensorQuantTest)
{
    this->run_test_with_validation(1024, 1024, 1024);
}
