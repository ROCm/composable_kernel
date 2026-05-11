// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

// Demonstrates applying a function to all elements without repetition

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "ck_tile/core.hpp"

using namespace ck_tile;

// ================================================================
// Helper: Apply a function to all elements of thread_buffer
// ================================================================
template <typename F, typename T, index_t N>
CK_TILE_HOST_DEVICE auto apply_to_all(const thread_buffer<T, N>& input, F func)
{
    thread_buffer<T, N> result;

    static_for<0, N, 1>{}([&](auto i) {
        result[i] = func(input[i]);
    });

    return result;
}

// ================================================================
// Lambda-based helper (even cleaner)
// ================================================================
template <typename DataType>
__global__ void clean_method_kernel(DataType* __restrict__ output,
                                    const DataType* __restrict__ input,
                                    int size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = 4;

    if (tid * stride < size)
    {
        // Load 4 elements
        thread_buffer<DataType, 4> y;
        for (int i = 0; i < 4; i++)
        {
            int idx = tid * stride + i;
            y[i] = (idx < size) ? input[idx] : 0.0f;
        }

        // ================================================================
        // METHOD: Use apply helper - NO REPETITION!
        // ================================================================
        auto exp_y = apply_to_all(y, [](auto x) { return ck_tile::exp(x); });

        // Alternative: You can also use a named lambda
        auto exp_func = [](auto x) { return ck_tile::exp(x); };
        auto exp_y2 = apply_to_all(y, exp_func);

        // Alternative: Direct call with function pointer (C++17)
        auto exp_y3 = apply_to_all(y, [](DataType x) { return ck_tile::exp(x); });

        // Store results
        for (int i = 0; i < 4; i++)
        {
            int idx = tid * stride + i;
            if (idx < size)
                output[idx] = exp_y[i];
        }
    }
}

// ================================================================
// Inline static_for approach (no helper function)
// ================================================================
template <typename DataType>
__global__ void inline_static_for_kernel(DataType* __restrict__ output,
                                         const DataType* __restrict__ input,
                                         int size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = 4;

    if (tid * stride < size)
    {
        thread_buffer<DataType, 4> y;
        for (int i = 0; i < 4; i++)
        {
            int idx = tid * stride + i;
            y[i] = (idx < size) ? input[idx] : 0.0f;
        }

        // ================================================================
        // METHOD: Use static_for directly - NO REPETITION!
        // ================================================================
        thread_buffer<DataType, 4> exp_y;

        static_for<0, 4, 1>{}([&](auto i) {
            exp_y[i] = ck_tile::exp(y[i]);
        });

        // Store results
        for (int i = 0; i < 4; i++)
        {
            int idx = tid * stride + i;
            if (idx < size)
                output[idx] = exp_y[i];
        }
    }
}

int main()
{
    const int size = 16;
    const int bytes = size * sizeof(float);

    printf("=== Apply Function Without Repetition ===\n\n");

    // Allocate host memory
    float* h_input = new float[size];
    float* h_output1 = new float[size];
    float* h_output2 = new float[size];

    // Initialize input
    printf("Input values:\n");
    for (int i = 0; i < size; i++)
    {
        h_input[i] = static_cast<float>(i) * 0.1f;
        printf("%5.2f ", h_input[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");

    // Allocate device memory
    float *d_input, *d_output1, *d_output2;

    if (hipMalloc(&d_input, bytes) != hipSuccess ||
        hipMalloc(&d_output1, bytes) != hipSuccess ||
        hipMalloc(&d_output2, bytes) != hipSuccess)
    {
        printf("Failed to allocate device memory\n");
        return -1;
    }

    hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice);

    // Launch kernels
    int threads_per_block = 64;
    int blocks = 1;

    printf("Method 1: Using apply_to_all() helper function\n");
    clean_method_kernel<<<blocks, threads_per_block>>>(d_output1, d_input, size);

    printf("Method 2: Using static_for directly\n\n");
    inline_static_for_kernel<<<blocks, threads_per_block>>>(d_output2, d_input, size);

    hipDeviceSynchronize();

    // Copy results back
    hipMemcpy(h_output1, d_output1, bytes, hipMemcpyDeviceToHost);
    hipMemcpy(h_output2, d_output2, bytes, hipMemcpyDeviceToHost);

    // Display code examples
    printf("=== Code Examples ===\n\n");

    printf("ORIGINAL (with repetition):\n");
    printf("  exp_y[0] = ck_tile::exp(y[0]);\n");
    printf("  exp_y[1] = ck_tile::exp(y[1]);\n");
    printf("  exp_y[2] = ck_tile::exp(y[2]);\n");
    printf("  exp_y[3] = ck_tile::exp(y[3]);\n\n");

    printf("IMPROVED Method 1 - apply_to_all() helper:\n");
    printf("  auto exp_y = apply_to_all(y, [](auto x) { return ck_tile::exp(x); });\n\n");

    printf("IMPROVED Method 2 - static_for:\n");
    printf("  thread_buffer<float, 4> exp_y;\n");
    printf("  static_for<0, 4, 1>{}([&](auto i) {\n");
    printf("      exp_y[i] = ck_tile::exp(y[i]);\n");
    printf("  });\n\n");

    printf("IMPROVED Method 3 - #pragma unroll loop:\n");
    printf("  thread_buffer<float, 4> exp_y;\n");
    printf("  #pragma unroll\n");
    printf("  for (int i = 0; i < 4; i++) {\n");
    printf("      exp_y[i] = ck_tile::exp(y[i]);\n");
    printf("  }\n\n");

    // Verify results
    printf("=== Results ===\n");
    printf("Method 1 output: ");
    for (int i = 0; i < 8; i++)
        printf("%5.2f ", h_output1[i]);
    printf("...\n");

    printf("Method 2 output: ");
    for (int i = 0; i < 8; i++)
        printf("%5.2f ", h_output2[i]);
    printf("...\n\n");

    bool all_correct = true;
    for (int i = 0; i < size; i++)
    {
        float expected = expf(h_input[i]);
        if (fabsf(h_output1[i] - expected) > 1e-5f ||
            fabsf(h_output2[i] - expected) > 1e-5f)
        {
            all_correct = false;
            break;
        }
    }

    printf("=== Recommendations ===\n");
    printf("For small fixed-size buffers (like CVec with 4 elements):\n");
    printf("  Best choice: apply_to_all() helper or static_for\n");
    printf("  Benefits:\n");
    printf("    - Write the operation once\n");
    printf("    - Fully unrolled at compile time\n");
    printf("    - Clean, functional style\n");
    printf("    - Type-safe with lambdas\n\n");

    printf("For runtime-sized or larger buffers:\n");
    printf("  Best choice: #pragma unroll loop\n");
    printf("  Benefits:\n");
    printf("    - Familiar loop syntax\n");
    printf("    - Works with runtime sizes\n");
    printf("    - Compiler handles unrolling\n\n");

    printf("Validation: %s\n", all_correct ? "PASSED ✓" : "FAILED ✗");

    // Cleanup
    hipFree(d_input);
    hipFree(d_output1);
    hipFree(d_output2);
    delete[] h_input;
    delete[] h_output1;
    delete[] h_output2;

    return all_correct ? 0 : -1;
}
