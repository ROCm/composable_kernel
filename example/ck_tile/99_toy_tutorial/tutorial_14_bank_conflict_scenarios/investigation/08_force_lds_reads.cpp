// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

/*
 * Force LDS Reads - Version that prevents compiler optimization
 *
 * This version uses ALL threads and creates true dependencies that
 * prevent the compiler from optimizing away LDS operations.
 */

#include "ck_tile/core.hpp"
#include <iostream>
#include <hip/hip_runtime.h>

using namespace ck_tile;

using DataType = half_t;
constexpr int kLdsSize = 64;

// Kernel: Each thread writes, then ALL threads read from LDS
// This creates true data dependencies that prevent optimization
__global__ void force_lds_reads_kernel(DataType* output)
{
    __shared__ DataType lds_memory[kLdsSize];

    int tid = threadIdx.x;

    // Create buffer_view for LDS
    auto lds_buf = make_buffer_view<address_space_enum::lds>(
        lds_memory,
        number<kLdsSize>{}
    );

    // ===== WRITE PHASE: Each thread writes its own element =====
    if(tid < kLdsSize)
    {
        lds_buf(tid) = DataType(tid);
    }

    __syncthreads();  // Ensure all writes complete

    // ===== READ PHASE: Each thread reads DIFFERENT element =====
    // Thread i reads from position (i + 1) % kLdsSize
    // This creates cross-thread dependency preventing optimization
    if(tid < kLdsSize)
    {
        int read_idx = (tid + 1) % kLdsSize;
        DataType value = lds_buf[read_idx];
        output[tid] = value;
    }
}

int main()
{
    std::cout << "\n=== Force LDS Reads Test ===\n" << std::endl;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  LDS buffer size: " << kLdsSize << " FP16 elements" << std::endl;
    std::cout << "  All " << kLdsSize << " threads active" << std::endl;
    std::cout << "  Each thread writes tid, reads (tid+1)%64\n" << std::endl;

    DataType* d_output;
    (void)hipMalloc(&d_output, kLdsSize * sizeof(DataType));

    std::cout << "Launching kernel..." << std::endl;
    force_lds_reads_kernel<<<1, kLdsSize>>>(d_output);
    (void)hipDeviceSynchronize();

    DataType h_output[kLdsSize];
    (void)hipMemcpy(h_output, d_output, kLdsSize * sizeof(DataType), hipMemcpyDeviceToHost);

    // Verify: thread i should read value (i+1)%64
    bool passed = true;
    for(int i = 0; i < kLdsSize; i++)
    {
        int expected = (i + 1) % kLdsSize;
        int actual = static_cast<int>(h_output[i]);
        if(actual != expected)
        {
            std::cout << "ERROR at thread " << i
                      << ": expected " << expected
                      << ", got " << actual << std::endl;
            passed = false;
        }
    }

    std::cout << "\nResults (first 16 values):" << std::endl;
    for(int i = 0; i < 16; i++)
    {
        int expected = (i + 1) % kLdsSize;
        std::cout << "  Thread " << i << " read: " << static_cast<int>(h_output[i])
                  << " (expected " << expected << ")" << std::endl;
    }

    std::cout << "\n";
    if(passed)
    {
        std::cout << "✓ Test PASSED - LDS reads are real!" << std::endl;
    }
    else
    {
        std::cout << "✗ Test FAILED" << std::endl;
    }

    (void)hipFree(d_output);

    return passed ? 0 : 1;
}
