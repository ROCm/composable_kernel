// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct and KERNEL_NAME

#define HIP_CHECK(cmd)                                                                          \
    do                                                                                          \
    {                                                                                           \
        hipError_t error = (cmd);                                                               \
        if(error != hipSuccess)                                                                 \
        {                                                                                       \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                 \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while(0)

int main(int argc, char* argv[])
{
    // Parse command line arguments
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1024", "M dimension")
        .insert("n", "1024", "N dimension")
        .insert("k", "1024", "K dimension")
        .insert("w", "50", "Warmup iterations")
        .insert("r", "100", "Repeat iterations");

    if(!arg_parser.parse(argc, argv))
    {
        return EXIT_FAILURE;
    }

    int m      = arg_parser.get_int("m");
    int n      = arg_parser.get_int("n");
    int k      = arg_parser.get_int("k");
    int warmup = arg_parser.get_int("w");
    int repeat = arg_parser.get_int("r");

    // Calculate strides (row-major for C layout)
    int stride_a = k; // M x K matrix in row-major
    int stride_b = n; // K x N matrix in row-major
    int stride_c = n; // M x N matrix in row-major

    // Allocate host memory
    size_t size_a = m * k * sizeof(ck_tile::half_t);
    size_t size_b = k * n * sizeof(ck_tile::half_t);
    size_t size_c = m * n * sizeof(ck_tile::half_t);

    std::vector<ck_tile::half_t> h_a(m * k);
    std::vector<ck_tile::half_t> h_b(k * n);
    std::vector<ck_tile::half_t> h_c(m * n);

    // Initialize with random data
    for(int i = 0; i < m * k; i++)
    {
        h_a[i] = ck_tile::half_t(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    for(int i = 0; i < k * n; i++)
    {
        h_b[i] = ck_tile::half_t(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }

    // Allocate device memory
    void* d_a;
    void* d_b;
    void* d_c;

    HIP_CHECK(hipMalloc(&d_a, size_a));
    HIP_CHECK(hipMalloc(&d_b, size_b));
    HIP_CHECK(hipMalloc(&d_c, size_c));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size_a, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), size_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_c, 0, size_c));

    // Create GemmHostArgs
    ck_tile::GemmHostArgs args = {
        d_a,      // a_ptr
        d_b,      // b_ptr
        d_c,      // c_ptr
        1,        // k_batch (split_k)
        m,        // M
        n,        // N
        k,        // K
        stride_a, // stride_A
        stride_b, // stride_B
        stride_c  // stride_C
    };

    // Create stream config
    ck_tile::stream_config stream{
        nullptr, // stream
        true,    // time_kernel
        false,   // log_level
        warmup,  // n_warmup
        repeat,  // n_repeat
        true,    // use_gpu_timer
        false,   // flush_cache
        5        // rotating_count
    };

    try
    {
        // Call the kernel's launch function directly
        float avg_time = SelectedKernel::launch(args, stream);

        // Calculate performance metrics
        size_t flop     = size_t(2) * m * n * k;
        size_t num_byte = sizeof(ck_tile::half_t) * (m * k + k * n + m * n);

        float tflops    = static_cast<float>(flop) / 1.E9 / avg_time;
        float bandwidth = num_byte / 1.E6 / avg_time;

        std::cout << "Running kernel: " << KERNEL_NAME << std::endl;
        std::cout << "Problem size: M=" << m << ", N=" << n << ", K=" << k << std::endl;
        std::cout << "Time: " << avg_time << " ms" << std::endl;
        std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
        std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
        return EXIT_FAILURE;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    return 0;
}
