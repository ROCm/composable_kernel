// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Minimal buffer_view LDS Test
 *
 * Purpose: Simplest possible example of using buffer_view with LDS
 * - Demonstrates buffer_view creation for LDS memory
 * - Shows basic write/read operations using operator() and operator[]
 * - Single-threaded (thread 0 only) for maximum simplicity
 * - Sequential access pattern (no transpose, no bank conflict analysis)
 * - Perfect for learning buffer_view API and debugging with rocgdb
 *
 * This is a learning tool to understand buffer_view before tackling
 * complex production kernels with tile_distribution, tile_window, etc.
 */

#include "ck_tile/core.hpp"
#include <iostream>
#include <hip/hip_runtime.h>

using namespace ck_tile;

// Use FP16 to match production kernels
using DataType = half_t;

// Small buffer size for easy debugging
constexpr int kLdsSize = 64;  // 64 FP16 elements = 128 bytes

// Minimal kernel: thread 0 writes then reads LDS using buffer_view
__global__ void minimal_buffer_view_kernel(DataType* output)
{
    // Allocate LDS memory
    __shared__ DataType lds_memory[kLdsSize];

    int tid = threadIdx.x;

    // Only thread 0 operates (simplest case)
    if(tid == 0)
    {
        // Create buffer_view wrapping LDS memory
        // This is the core CK-Tile abstraction for LDS access
        auto lds_buf = make_buffer_view<address_space_enum::lds>(
            lds_memory,           // __shared__ pointer
            number<kLdsSize>{}    // buffer size
        );

        // ===== WRITE PHASE =====
        // Write sequential values using operator()(index)
        // operator() is used for write access
        for(int i = 0; i < kLdsSize; i++)
        {
            lds_buf(i) = DataType(i);  // Write value i to position i
        }

        // Memory barrier (not strictly needed for single thread, but good practice)
        __syncthreads();

        // ===== READ PHASE =====
        // Read values back using operator[](index)
        // operator[] is used for read access
        for(int i = 0; i < kLdsSize; i++)
        {
            DataType value = lds_buf[i];  // Read value from position i
            output[i] = value;            // Copy to global memory
        }
    }

    __syncthreads();  // Ensure all memory operations complete
}

int main()
{
    std::cout << "\n=== Minimal buffer_view LDS Test ===\n" << std::endl;

    // Print configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  LDS buffer size: " << kLdsSize << " FP16 elements" << std::endl;
    std::cout << "  LDS byte size: " << (kLdsSize * sizeof(DataType)) << " bytes" << std::endl;
    std::cout << "  Access pattern: Sequential (no transpose)" << std::endl;
    std::cout << "  Thread model: Thread 0 only\n" << std::endl;

    // Allocate device memory for output
    DataType* d_output;
    (void)hipMalloc(&d_output, kLdsSize * sizeof(DataType));

    // Launch kernel with 1 block, 64 threads (though only thread 0 works)
    std::cout << "Launching kernel..." << std::endl;
    minimal_buffer_view_kernel<<<1, 64>>>(d_output);
    (void)hipDeviceSynchronize();

    // Copy results to host
    DataType h_output[kLdsSize];
    (void)hipMemcpy(h_output, d_output, kLdsSize * sizeof(DataType), hipMemcpyDeviceToHost);

    // Verify results
    bool passed = true;
    for(int i = 0; i < kLdsSize; i++)
    {
        int expected = i;
        int actual = static_cast<int>(h_output[i]);
        if(actual != expected)
        {
            std::cout << "ERROR at index " << i
                      << ": expected " << expected
                      << ", got " << actual << std::endl;
            passed = false;
        }
    }

    // Print first 16 values
    std::cout << "\nResults (first 16 values):" << std::endl;
    for(int i = 0; i < 16; i++)
    {
        std::cout << "  [" << i << "] = " << static_cast<float>(h_output[i]) << std::endl;
    }

    // Print all values in compact form
    std::cout << "\nAll " << kLdsSize << " values: ";
    for(int i = 0; i < kLdsSize; i++)
    {
        std::cout << static_cast<float>(h_output[i]);
        if(i < kLdsSize - 1) std::cout << " ";
    }
    std::cout << std::endl;

    // Print result
    std::cout << "\n";
    if(passed)
    {
        std::cout << "✓ Test PASSED" << std::endl;
    }
    else
    {
        std::cout << "✗ Test FAILED" << std::endl;
    }

    // Cleanup
    (void)hipFree(d_output);

    return passed ? 0 : 1;
}
