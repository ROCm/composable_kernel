// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for GroupedConvProblem using assert() and std::cout

#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace ck_tile::dispatcher;

void test_grouped_conv_problem_defaults()
{
    std::cout << "  test_grouped_conv_problem_defaults... ";
    GroupedConvProblem p;
    assert(p.N == 1);
    assert(p.C == 64);
    assert(p.K == 64);
    assert(p.G == 1);
    assert(p.Hi() == 28);
    assert(p.Wi() == 28);
    assert(p.Y() == 3);
    assert(p.X() == 3);
    assert(p.op == GroupedConvOp::Forward);
    assert(p.stride[0] == 1 && p.stride[1] == 1 && p.stride[2] == 1);
    assert(p.padding[0] == 0 && p.padding[1] == 0 && p.padding[2] == 0);
    assert(p.dilation[0] == 1 && p.dilation[1] == 1 && p.dilation[2] == 1);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_2d()
{
    std::cout << "  test_grouped_conv_problem_2d... ";
    GroupedConvProblem p(4, 64, 128, 28, 28, 3, 3);
    p.compute_output_size();
    assert(p.N == 4);
    assert(p.C == 64);
    assert(p.K == 128);
    assert(p.Hi() == 28);
    assert(p.Wi() == 28);
    assert(p.Y() == 3);
    assert(p.X() == 3);
    assert(p.Ho() == 26);
    assert(p.Wo() == 26);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_strided()
{
    std::cout << "  test_grouped_conv_problem_strided... ";
    GroupedConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 64;
    p.G              = 1;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 2, 2};
    p.padding        = {0, 1, 1};
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    assert(p.Ho() == 7);
    assert(p.Wo() == 7);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_grouped()
{
    std::cout << "  test_grouped_conv_problem_grouped... ";
    GroupedConvProblem p;
    p.N              = 2;
    p.C              = 64;
    p.K              = 64;
    p.G              = 4;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding        = {0, 0, 0};
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    assert(p.G == 4);
    assert(p.C % p.G == 0);
    assert(p.K % p.G == 0);
    assert(p.is_valid());
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_depthwise()
{
    std::cout << "  test_grouped_conv_problem_depthwise... ";
    GroupedConvProblem p;
    p.N              = 2;
    p.C              = 64;
    p.K              = 64;
    p.G              = 64;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding        = {0, 0, 0};
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    assert(p.is_depthwise());
    assert(p.G == p.C && p.G == p.K);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_pointwise()
{
    std::cout << "  test_grouped_conv_problem_pointwise... ";
    GroupedConvProblem p;
    p.N              = 2;
    p.C              = 64;
    p.K              = 128;
    p.G              = 1;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 1, 1};
    p.stride         = {1, 1, 1};
    p.padding        = {0, 0, 0};
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    assert(p.is_pointwise());
    assert(p.Y() == 1 && p.X() == 1);
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_flops()
{
    std::cout << "  test_grouped_conv_problem_flops... ";
    GroupedConvProblem p;
    p.N              = 2;
    p.C              = 64;
    p.K              = 64;
    p.G              = 1;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding        = {0, 0, 0};
    p.dilation       = {1, 1, 1};
    p.compute_output_size();
    double flops = p.get_flops();
    assert(flops > 0);
    assert(flops == 2.0 * p.N * p.K * p.Ho() * p.Wo() * (p.C / p.G) * p.Y() * p.X());
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_is_valid()
{
    std::cout << "  test_grouped_conv_problem_is_valid... ";
    GroupedConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 64;
    p.G              = 1;
    p.input_spatial  = {1, 14, 14};
    p.filter_spatial = {1, 3, 3};
    p.compute_output_size();
    assert(p.is_valid());

    p.N = 0;
    assert(!p.is_valid());
    p.N = 1;

    p.C = 0;
    assert(!p.is_valid());
    p.C = 64;

    p.K = 0;
    assert(!p.is_valid());
    p.K = 64;

    p.G = 0;
    assert(!p.is_valid());
    p.G = 1;

    p.C = 64;
    p.K = 64;
    p.G = 3;
    assert(!p.is_valid());
    p.G = 4;
    assert(p.is_valid());
    std::cout << "PASSED\n";
}

void test_grouped_conv_problem_builder()
{
    std::cout << "  test_grouped_conv_problem_builder... ";
    auto p = GroupedConvProblemBuilder()
                 .batch(8)
                 .channels(128, 256)
                 .groups(4)
                 .input_size(32, 32)
                 .filter_size(3, 3)
                 .stride(2, 2)
                 .padding(1, 1)
                 .dilation(1, 1)
                 .operation(GroupedConvOp::Forward)
                 .build();
    assert(p.N == 8);
    assert(p.C == 128);
    assert(p.K == 256);
    assert(p.G == 4);
    assert(p.Hi() == 32);
    assert(p.Wi() == 32);
    assert(p.Y() == 3);
    assert(p.X() == 3);
    assert(p.stride[1] == 2 && p.stride[2] == 2);
    assert(p.padding[1] == 1 && p.padding[2] == 1);
    assert(p.op == GroupedConvOp::Forward);
    assert(p.is_valid());

    bool threw = false;
    try
    {
        (void)GroupedConvProblemBuilder()
            .batch(0)
            .channels(64, 64)
            .groups(1)
            .input_size(14, 14)
            .filter_size(3, 3)
            .build();
    }
    catch(const std::invalid_argument&)
    {
        threw = true;
    }
    assert(threw);
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Test Grouped Conv Problem ===\n\n";
    test_grouped_conv_problem_defaults();
    test_grouped_conv_problem_2d();
    test_grouped_conv_problem_strided();
    test_grouped_conv_problem_grouped();
    test_grouped_conv_problem_depthwise();
    test_grouped_conv_problem_pointwise();
    test_grouped_conv_problem_flops();
    test_grouped_conv_problem_is_valid();
    test_grouped_conv_problem_builder();
    std::cout << "\n=== All Tests Passed! ===\n\n";
    return 0;
}
