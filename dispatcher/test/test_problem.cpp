// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/// Unit tests for Problem

#include "ck_tile/dispatcher/problem.hpp"
#include <cassert>
#include <iostream>

using namespace ck_tile::dispatcher;

void test_problem_construction()
{
    std::cout << "Test: Problem construction... ";
    
    // Default constructor
    Problem p1;
    assert(p1.M == 0);
    assert(p1.N == 0);
    assert(p1.K == 0);
    assert(p1.k_batch == 1);
    assert(!p1.is_valid());
    
    // Constructor with dimensions
    Problem p2(1024, 1024, 1024);
    assert(p2.M == 1024);
    assert(p2.N == 1024);
    assert(p2.K == 1024);
    assert(p2.is_valid());
    
    std::cout << "PASSED\n";
}

void test_problem_validation()
{
    std::cout << "Test: Problem validation... ";
    
    Problem p;
    
    // Invalid: all zeros
    p.M = 0; p.N = 0; p.K = 0;
    assert(!p.is_valid());
    
    // Invalid: negative
    p.M = -1; p.N = 1024; p.K = 1024;
    assert(!p.is_valid());
    
    // Invalid: zero K
    p.M = 1024; p.N = 1024; p.K = 0;
    assert(!p.is_valid());
    
    // Valid
    p.M = 1024; p.N = 1024; p.K = 1024;
    assert(p.is_valid());
    
    // Invalid k_batch
    p.k_batch = 0;
    assert(!p.is_valid());
    
    p.k_batch = 1;
    assert(p.is_valid());
    
    std::cout << "PASSED\n";
}

void test_problem_num_ops()
{
    std::cout << "Test: Problem num_ops... ";
    
    Problem p(100, 200, 300);
    
    // 2 * M * N * K (multiply-add = 2 ops)
    std::int64_t expected = 2 * 100 * 200 * 300;
    assert(p.num_ops() == expected);
    
    std::cout << "PASSED\n";
}

void test_problem_configuration()
{
    std::cout << "Test: Problem configuration... ";
    
    Problem p(1024, 1024, 1024);
    
    // Set preferences
    p.prefer_persistent = true;
    p.enable_validation = true;
    p.smem_budget = 65536;
    p.k_batch = 2;
    
    assert(p.prefer_persistent);
    assert(p.enable_validation);
    assert(p.smem_budget == 65536);
    assert(p.k_batch == 2);
    
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "=== Problem Unit Tests ===\n\n";
    
    test_problem_construction();
    test_problem_validation();
    test_problem_num_ops();
    test_problem_configuration();
    
    std::cout << "\n=== All Problem tests PASSED ===\n";
    return 0;
}

