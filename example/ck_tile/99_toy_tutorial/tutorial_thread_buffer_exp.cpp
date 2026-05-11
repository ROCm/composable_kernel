// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "ck_tile/core.hpp"

using namespace ck_tile;

// Simple kernel that demonstrates thread_buffer with get_as and exp
template <typename DataType>
__global__ void thread_buffer_exp_kernel(DataType* __restrict__ output,
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
            if (idx < size)
                y[i] = input[idx];
            else
                y[i] = 0.0f;
        }

        // Method 1: Apply exp element-wise using thread_buffer indexing
        thread_buffer<DataType, 4> exp_y;

        exp_y[0] = ck_tile::exp(y[0]);
        exp_y[1] = ck_tile::exp(y[1]);
        exp_y[2] = ck_tile::exp(y[2]);
        exp_y[3] = ck_tile::exp(y[3]);

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
    printf("Starting thread_buffer exp test...\n");
    fflush(stdout);

    const int size = 16;
    const int bytes = size * sizeof(float);

    printf("Allocating host memory...\n");
    fflush(stdout);

    // Allocate host memory
    float* h_input = new float[size];
    float* h_output = new float[size];

    printf("Host memory allocated\n");
    fflush(stdout);

    // Initialize input with simple values
    printf("Initializing input...\n");
    fflush(stdout);

    for (int i = 0; i < size; i++)
    {
        h_input[i] = static_cast<float>(i) * 0.1f;
    }

    printf("Input values:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%6.3f ", h_input[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }
    printf("\n");
    fflush(stdout);

    // Allocate device memory
    float* d_input;
    float* d_output;
    if (hipMalloc(&d_input, bytes) != hipSuccess) {
        printf("Failed to allocate device input\n");
        return -1;
    }
    if (hipMalloc(&d_output, bytes) != hipSuccess) {
        printf("Failed to allocate device output\n");
        return -1;
    }

    // Copy input to device
    if (hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        printf("Failed to copy input to device\n");
        return -1;
    }

    // Launch kernel
    int threads_per_block = 64;
    int blocks = (size + (threads_per_block * 4) - 1) / (threads_per_block * 4);

    printf("Launching kernel with %d blocks, %d threads per block\n", blocks, threads_per_block);
    thread_buffer_exp_kernel<<<blocks, threads_per_block>>>(d_output, d_input, size);

    // Check for kernel launch errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
        return -1;
    }

    // Copy output back to host
    if (hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        printf("Failed to copy output from device\n");
        return -1;
    }

    // Verify results
    printf("\nOutput values (exp(input)):\n");
    bool passed = true;
    for (int i = 0; i < size; i++)
    {
        float expected = expf(h_input[i]);  // Use expf instead of std::exp
        float error = fabsf(h_output[i] - expected) / expected;

        printf("%6.3f ", h_output[i]);
        if ((i + 1) % 8 == 0) printf("\n");

        if (error > 1e-5f)
        {
            printf("\nError at index %d: expected %.6f, got %.6f (rel error: %.2e)\n",
                   i, expected, h_output[i], error);
            passed = false;
        }
    }
    printf("\n");

    printf("\nValidation: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return passed ? 0 : -1;
}
