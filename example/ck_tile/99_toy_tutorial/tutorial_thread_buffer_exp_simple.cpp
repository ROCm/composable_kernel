// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

// Simple program demonstrating thread_buffer usage with exp
// This is a CPU-side demonstration of the concepts

#include <cstdio>
#include <cmath>
#include "ck_tile/core.hpp"

using namespace ck_tile;

int main()
{
    printf("=== CK Tile thread_buffer with exp() Demo ===\n\n");

    // Create a thread_buffer with 4 floats
    thread_buffer<float, 4> y;

    printf("Step 1: Initialize thread_buffer with 4 values\n");
    y[0] = 0.0f;
    y[1] = 0.5f;
    y[2] = 1.0f;
    y[3] = 1.5f;

    printf("Input values: [%.2f, %.2f, %.2f, %.2f]\n\n", y[0], y[1], y[2], y[3]);

    // Method 1: Element-wise application (works everywhere - host and device)
    printf("Step 2: Apply exp() element-wise\n");
    thread_buffer<float, 4> exp_y;

    exp_y[0] = ck_tile::exp(y[0]);
    exp_y[1] = ck_tile::exp(y[1]);
    exp_y[2] = ck_tile::exp(y[2]);
    exp_y[3] = ck_tile::exp(y[3]);

    printf("Output values: [%.4f, %.4f, %.4f, %.4f]\n\n",
           exp_y[0], exp_y[1], exp_y[2], exp_y[3]);

    // Verify against standard library
    printf("Step 3: Verify results\n");
    bool passed = true;
    for (int i = 0; i < 4; i++)
    {
        float expected = expf(y[i]);
        float error = fabsf(exp_y[i] - expected);

        printf("  exp(%.2f) = %.6f (expected: %.6f, error: %.2e)\n",
               y[i], exp_y[i], expected, error);

        if (error > 1e-5f)
            passed = false;
    }

    printf("\n");

    // Show how to use get_as (converts thread_buffer<float, 4> to thread_buffer<fp32x4_t, 1>)
    printf("Step 4: Demonstrate get_as<fp32x4_t>()\n");
    printf("  thread_buffer<float, 4> can be viewed as thread_buffer<fp32x4_t, 1>\n");
    printf("  where fp32x4_t is 'float __attribute__((ext_vector_type(4)))'\n");

    auto y_as_vec = y.get_as<fp32x4_t>();
    printf("  y.get_as<fp32x4_t>() returns a thread_buffer with %ld element(s)\n",
           y_as_vec.size());
    printf("  Each element is a 4-wide vector type\n");

    // Access the actual vector
    fp32x4_t vec = y_as_vec[0];
    printf("  Vector contents: [%.2f, %.2f, %.2f, %.2f]\n",
           vec[0], vec[1], vec[2], vec[3]);

    printf("\n=== Summary ===\n");
    printf("thread_buffer is CK Tile's container for register-held data\n");
    printf("- Access elements with operator[]: buffer[i]\n");
    printf("- Apply functions element-wise: exp_y[i] = ck_tile::exp(y[i])\n");
    printf("- Convert to vector types with get_as<T>() for vectorized ops\n");
    printf("- Similar to __attribute__((ext_vector_type(N))) but with more features\n");

    printf("\nValidation: %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    return passed ? 0 : -1;
}
