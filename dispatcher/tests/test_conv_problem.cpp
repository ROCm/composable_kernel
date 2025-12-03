// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_conv_problem.cpp
 * @brief Unit tests for convolution problem definition
 */

#include <iostream>
#include <cassert>
#include <cmath>

#include "ck_tile/dispatcher/conv_problem.hpp"

using namespace ck_tile::dispatcher;

void test_conv_problem_defaults()
{
    std::cout << "  test_conv_problem_defaults... ";

    ConvProblem p;

    // Default is 2D conv
    assert(p.N == 1);
    assert(p.C > 0);
    assert(p.K > 0);
    assert(p.G == 1);
    assert(p.op == ConvOp::Forward);

    std::cout << "PASSED\n";
}

void test_conv_problem_2d()
{
    std::cout << "  test_conv_problem_2d... ";

    ConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 128;
    p.G              = 1;
    p.input_spatial  = {1, 28, 28}; // D, H, W
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding        = {0, 1, 1};
    p.dilation       = {1, 1, 1};
    p.op             = ConvOp::Forward;
    p.compute_output_size();

    // Output should be 28x28 with same padding
    assert(p.output_spatial[1] == 28); // H
    assert(p.output_spatial[2] == 28); // W

    std::cout << "PASSED\n";
}

void test_conv_problem_3d()
{
    std::cout << "  test_conv_problem_3d... ";

    ConvProblem p;
    p.N              = 1;
    p.C              = 32;
    p.K              = 64;
    p.G              = 1;
    p.input_spatial  = {8, 16, 16}; // D, H, W
    p.filter_spatial = {3, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding        = {1, 1, 1};
    p.dilation       = {1, 1, 1};
    p.op             = ConvOp::Forward;
    p.compute_output_size();

    // Output should preserve spatial with same padding
    assert(p.output_spatial[0] == 8);  // D
    assert(p.output_spatial[1] == 16); // H
    assert(p.output_spatial[2] == 16); // W

    std::cout << "PASSED\n";
}

void test_conv_problem_strided()
{
    std::cout << "  test_conv_problem_strided... ";

    ConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 128;
    p.G              = 1;
    p.input_spatial  = {1, 28, 28};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 2, 2}; // Stride 2
    p.padding        = {0, 1, 1};
    p.dilation       = {1, 1, 1};
    p.op             = ConvOp::Forward;
    p.compute_output_size();

    // Output should be halved with stride 2
    assert(p.output_spatial[1] == 14); // H
    assert(p.output_spatial[2] == 14); // W

    std::cout << "PASSED\n";
}

void test_conv_problem_grouped()
{
    std::cout << "  test_conv_problem_grouped... ";

    ConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 64;
    p.G              = 4; // 4 groups
    p.input_spatial  = {1, 28, 28};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding        = {0, 1, 1};
    p.dilation       = {1, 1, 1};
    p.op             = ConvOp::Forward;
    p.compute_output_size();

    // Grouped conv should still work
    assert(p.G == 4);
    assert(p.C / p.G == 16); // Channels per group

    std::cout << "PASSED\n";
}

void test_conv_problem_depthwise()
{
    std::cout << "  test_conv_problem_depthwise... ";

    ConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 64;
    p.G              = 64; // Depthwise: G = C = K
    p.input_spatial  = {1, 28, 28};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding        = {0, 1, 1};
    p.dilation       = {1, 1, 1};
    p.op             = ConvOp::Forward;
    p.compute_output_size();

    assert(p.is_depthwise());

    std::cout << "PASSED\n";
}

void test_conv_problem_pointwise()
{
    std::cout << "  test_conv_problem_pointwise... ";

    ConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 128;
    p.G              = 1;
    p.input_spatial  = {1, 28, 28};
    p.filter_spatial = {1, 1, 1}; // 1x1 conv
    p.stride         = {1, 1, 1};
    p.padding        = {0, 0, 0};
    p.dilation       = {1, 1, 1};
    p.op             = ConvOp::Forward;
    p.compute_output_size();

    assert(p.is_pointwise());
    assert(p.output_spatial[1] == 28);
    assert(p.output_spatial[2] == 28);

    std::cout << "PASSED\n";
}

void test_conv_problem_flops()
{
    std::cout << "  test_conv_problem_flops... ";

    ConvProblem p;
    p.N              = 1;
    p.C              = 64;
    p.K              = 128;
    p.G              = 1;
    p.input_spatial  = {1, 28, 28};
    p.filter_spatial = {1, 3, 3};
    p.stride         = {1, 1, 1};
    p.padding        = {0, 1, 1};
    p.dilation       = {1, 1, 1};
    p.op             = ConvOp::Forward;
    p.compute_output_size();

    double flops = p.get_flops();

    // Expected: 2 * N * K * Ho * Wo * C * Y * X
    // = 2 * 1 * 128 * 28 * 28 * 64 * 3 * 3
    double expected = 2.0 * 1 * 128 * 28 * 28 * 64 * 3 * 3;

    assert(std::abs(flops - expected) < 1e-6);

    std::cout << "PASSED\n";
}

void test_conv_problem_backward()
{
    std::cout << "  test_conv_problem_backward... ";

    // Backward data
    ConvProblem p1;
    p1.N              = 1;
    p1.C              = 64;
    p1.K              = 128;
    p1.G              = 1;
    p1.input_spatial  = {1, 28, 28};
    p1.filter_spatial = {1, 3, 3};
    p1.stride         = {1, 1, 1};
    p1.padding        = {0, 1, 1};
    p1.dilation       = {1, 1, 1};
    p1.op             = ConvOp::BackwardData;
    p1.compute_output_size();

    assert(p1.op == ConvOp::BackwardData);

    // Backward weight
    ConvProblem p2;
    p2.N              = 1;
    p2.C              = 64;
    p2.K              = 128;
    p2.G              = 1;
    p2.input_spatial  = {1, 28, 28};
    p2.filter_spatial = {1, 3, 3};
    p2.stride         = {1, 1, 1};
    p2.padding        = {0, 1, 1};
    p2.dilation       = {1, 1, 1};
    p2.op             = ConvOp::BackwardWeight;
    p2.compute_output_size();

    assert(p2.op == ConvOp::BackwardWeight);

    std::cout << "PASSED\n";
}

void test_conv_op_enum()
{
    std::cout << "  test_conv_op_enum... ";

    assert(static_cast<int>(ConvOp::Forward) == 0);
    assert(static_cast<int>(ConvOp::BackwardData) == 1);
    assert(static_cast<int>(ConvOp::BackwardWeight) == 2);

    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Conv Problem Tests ===\n\n";

    test_conv_problem_defaults();
    test_conv_problem_2d();
    test_conv_problem_3d();
    test_conv_problem_strided();
    test_conv_problem_grouped();
    test_conv_problem_depthwise();
    test_conv_problem_pointwise();
    test_conv_problem_flops();
    test_conv_problem_backward();
    test_conv_op_enum();

    std::cout << "\n=== All Conv Problem Tests Passed! ===\n\n";
    return 0;
}
