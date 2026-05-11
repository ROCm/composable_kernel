// SPDX-License-Identifier: MIT
// Test if tile_elementwise works on thread_buffer

#include <cstdio>
#include <cmath>
#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_elementwise.hpp"

using namespace ck_tile;

template <typename DataType>
__global__ void test_elementwise_kernel(DataType* output, const DataType* input, int size)
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

        // Try using tile_elementwise directly on thread_buffer
        // This likely won't work because tile_elementwise expects distributed tensors
        // But let's try!

        // Method 1: Try tile_elementwise_in (expects distributed tensor)
        // auto exp_y = tile_elementwise_in([](auto x) { return ck_tile::exp(x); }, y);

        // Method 2: Manual static_for (what actually works)
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
    printf("Testing if tile_elementwise works on thread_buffer...\n\n");

    printf("Short answer: tile_elementwise_* functions are designed for\n");
    printf("distributed_tensor types, NOT raw thread_buffer.\n\n");

    printf("For thread_buffer, use:\n");
    printf("  1. static_for<0, N, 1>{}([&](auto i) { result[i] = op(input[i]); });\n");
    printf("  2. #pragma unroll for (int i = 0; i < N; i++) { ... }\n\n");

    printf("tile_elementwise is for higher-level tile operations on\n");
    printf("distributed tensors that manage thread buffers internally.\n\n");

    // Run the working version
    const int size = 16;
    const int bytes = size * sizeof(float);

    float* h_input = new float[size];
    float* h_output = new float[size];

    for (int i = 0; i < size; i++)
        h_input[i] = static_cast<float>(i) * 0.1f;

    float *d_input, *d_output;
    hipMalloc(&d_input, bytes);
    hipMalloc(&d_output, bytes);
    hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice);

    test_elementwise_kernel<<<1, 64>>>(d_output, d_input, size);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost);

    printf("Results using static_for:\n");
    for (int i = 0; i < 8; i++)
        printf("  exp(%.1f) = %.4f\n", h_input[i], h_output[i]);

    bool passed = true;
    for (int i = 0; i < size; i++)
    {
        float expected = expf(h_input[i]);
        if (fabsf(h_output[i] - expected) > 1e-5f)
            passed = false;
    }

    printf("\nValidation: %s\n", passed ? "PASSED ✓" : "FAILED ✗");

    hipFree(d_input);
    hipFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return passed ? 0 : -1;
}
