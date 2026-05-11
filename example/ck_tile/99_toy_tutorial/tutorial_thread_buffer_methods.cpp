// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

// Demonstrates different methods of applying operations to thread_buffer

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "ck_tile/core.hpp"

using namespace ck_tile;

template <typename DataType>
__global__ void method_comparison_kernel(DataType* __restrict__ output1,
                                         DataType* __restrict__ output2,
                                         DataType* __restrict__ output3,
                                         const DataType* __restrict__ input,
                                         int size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = 4; // Process 4 elements per thread

    if (tid * stride < size)
    {
        // Load 4 elements into thread_buffer
        thread_buffer<DataType, 4> y;

        for (int i = 0; i < 4; i++)
        {
            int idx = tid * stride + i;
            y[i] = (idx < size) ? input[idx] : 0.0f;
        }

        // ================================================================
        // METHOD 1: Element-wise indexing (most explicit, always works)
        // ================================================================
        thread_buffer<DataType, 4> result1;

        result1[0] = ck_tile::exp(y[0]);
        result1[1] = ck_tile::exp(y[1]);
        result1[2] = ck_tile::exp(y[2]);
        result1[3] = ck_tile::exp(y[3]);

        // ================================================================
        // METHOD 2: Using get_as to view as vector type, then manual ops
        // ================================================================
        thread_buffer<DataType, 4> result2;

        // get_as<fp32x4_t>() returns thread_buffer<fp32x4_t, 1>
        // Access [0] to get the actual fp32x4_t vector
        fp32x4_t y_vec = y.template get_as<fp32x4_t>()[0];

        // Apply exp to each element of the vector
        fp32x4_t result2_vec;
        result2_vec[0] = ck_tile::exp(y_vec[0]);
        result2_vec[1] = ck_tile::exp(y_vec[1]);
        result2_vec[2] = ck_tile::exp(y_vec[2]);
        result2_vec[3] = ck_tile::exp(y_vec[3]);

        // Convert back to thread_buffer
        result2.template set_as<fp32x4_t>(0, result2_vec);

        // ================================================================
        // METHOD 3: Loop-based (good for larger buffers)
        // ================================================================
        thread_buffer<DataType, 4> result3;

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            result3[i] = ck_tile::exp(y[i]);
        }

        // Store all results
        for (int i = 0; i < 4; i++)
        {
            int idx = tid * stride + i;
            if (idx < size)
            {
                output1[idx] = result1[i];
                output2[idx] = result2[i];
                output3[idx] = result3[i];
            }
        }
    }
}

int main()
{
    const int size = 16;
    const int bytes = size * sizeof(float);

    printf("=== thread_buffer Methods Comparison ===\n\n");

    // Allocate host memory
    float* h_input = new float[size];
    float* h_output1 = new float[size];
    float* h_output2 = new float[size];
    float* h_output3 = new float[size];

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
    float *d_input, *d_output1, *d_output2, *d_output3;

    if (hipMalloc(&d_input, bytes) != hipSuccess ||
        hipMalloc(&d_output1, bytes) != hipSuccess ||
        hipMalloc(&d_output2, bytes) != hipSuccess ||
        hipMalloc(&d_output3, bytes) != hipSuccess)
    {
        printf("Failed to allocate device memory\n");
        return -1;
    }

    // Copy input to device
    if (hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice) != hipSuccess)
    {
        printf("Failed to copy input to device\n");
        return -1;
    }

    // Launch kernel
    int threads_per_block = 64;
    int blocks = 1;

    printf("Launching kernel...\n\n");
    method_comparison_kernel<<<blocks, threads_per_block>>>(
        d_output1, d_output2, d_output3, d_input, size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
        return -1;
    }

    // Copy results back
    if (hipMemcpy(h_output1, d_output1, bytes, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(h_output2, d_output2, bytes, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(h_output3, d_output3, bytes, hipMemcpyDeviceToHost) != hipSuccess)
    {
        printf("Failed to copy output from device\n");
        return -1;
    }

    // Verify and display results
    printf("METHOD 1 - Element-wise indexing:\n");
    printf("  Code: result[i] = ck_tile::exp(y[i])\n");
    printf("  Output: ");
    for (int i = 0; i < size; i++)
    {
        printf("%5.2f ", h_output1[i]);
        if ((i + 1) % 8 == 0 && i + 1 < size) printf("\n          ");
    }
    printf("\n\n");

    printf("METHOD 2 - Using get_as<fp32x4_t>():\n");
    printf("  Code: fp32x4_t vec = y.get_as<fp32x4_t>()[0]\n");
    printf("        vec[i] = ck_tile::exp(vec[i])\n");
    printf("        result.set_as<fp32x4_t>(0, vec)\n");
    printf("  Output: ");
    for (int i = 0; i < size; i++)
    {
        printf("%5.2f ", h_output2[i]);
        if ((i + 1) % 8 == 0 && i + 1 < size) printf("\n          ");
    }
    printf("\n\n");

    printf("METHOD 3 - Loop with #pragma unroll:\n");
    printf("  Code: #pragma unroll\n");
    printf("        for (int i = 0; i < 4; i++)\n");
    printf("            result[i] = ck_tile::exp(y[i])\n");
    printf("  Output: ");
    for (int i = 0; i < size; i++)
    {
        printf("%5.2f ", h_output3[i]);
        if ((i + 1) % 8 == 0 && i + 1 < size) printf("\n          ");
    }
    printf("\n\n");

    // Verify all methods produce same results
    printf("=== Verification ===\n");
    bool all_match = true;
    bool all_correct = true;

    for (int i = 0; i < size; i++)
    {
        float expected = expf(h_input[i]);

        // Check if all methods agree
        if (fabsf(h_output1[i] - h_output2[i]) > 1e-6f ||
            fabsf(h_output1[i] - h_output3[i]) > 1e-6f)
        {
            printf("Methods disagree at index %d!\n", i);
            all_match = false;
        }

        // Check if result is correct
        float error = fabsf(h_output1[i] - expected);
        if (error > 1e-5f)
        {
            printf("Incorrect result at index %d: expected %.6f, got %.6f\n",
                   i, expected, h_output1[i]);
            all_correct = false;
        }
    }

    if (all_match && all_correct)
    {
        printf("✓ All methods produce identical, correct results\n");
    }

    printf("\n=== Method Recommendations ===\n");
    printf("Method 1 (element-wise): Most explicit, easiest to read\n");
    printf("Method 2 (get_as):       Useful when you need actual vector type\n");
    printf("Method 3 (loop):         Best for larger buffers, compiler unrolls\n");
    printf("\nFor your use case (CVec with 4 elements): Use Method 1 or 3\n");

    // Cleanup
    hipFree(d_input);
    hipFree(d_output1);
    hipFree(d_output2);
    hipFree(d_output3);
    delete[] h_input;
    delete[] h_output1;
    delete[] h_output2;
    delete[] h_output3;

    printf("\nValidation: %s\n", (all_match && all_correct) ? "PASSED ✓" : "FAILED ✗");

    return (all_match && all_correct) ? 0 : -1;
}
