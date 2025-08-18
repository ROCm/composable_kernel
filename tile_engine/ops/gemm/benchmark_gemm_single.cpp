// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <string>
#include <getopt.h>
#include <cstdlib>
#include <random>

#include "ck_tile/host.hpp"
#include "gemm_single_common.hpp"

// This header will be generated for each individual kernel and included via -include flag
// The specific header is defined by GEMM_SINGLE_INSTANCE_HPP macro and contains:
// - ADataType, BDataType, CDataType type definitions
// - KERNEL_NAME constant
// - SelectedKernel struct with launch() method

void print_help(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -m, --m_size <value>          M dimension size (default: 1024)\n"
              << "  -n, --n_size <value>          N dimension size (default: 1024)\n"
              << "  -k, --k_size <value>          K dimension size (default: 1024)\n"
              << "  -b, --k_batch <value>         K batch size for split-K (default: 1)\n"
              << "  -s, --stride_a <value>        Stride A (default: auto)\n"
              << "  -t, --stride_b <value>        Stride B (default: auto)\n"
              << "  -u, --stride_c <value>        Stride C (default: auto)\n"
              << "  -r, --repeat <value>          Number of iterations (default: 100)\n"
              << "  -w, --warmup <value>          Number of warmup iterations (default: 10)\n"
              << "  -v, --validate                Enable validation\n"
              << "  -h, --help                    Print this help message\n";
}

int main(int argc, char* argv[])
{
    // Default parameters
    ck_tile::index_t M        = 1024;
    ck_tile::index_t N        = 1024;
    ck_tile::index_t K        = 1024;
    ck_tile::index_t k_batch  = 1;
    ck_tile::index_t stride_A = 0; // 0 means auto
    ck_tile::index_t stride_B = 0;
    ck_tile::index_t stride_C = 0;
    int repeat                = 100;
    int warmup                = 10;
    bool validate             = false;

    // Parse command line arguments
    static struct option long_options[] = {{"m_size", required_argument, nullptr, 'm'},
                                           {"n_size", required_argument, nullptr, 'n'},
                                           {"k_size", required_argument, nullptr, 'k'},
                                           {"k_batch", required_argument, nullptr, 'b'},
                                           {"stride_a", required_argument, nullptr, 's'},
                                           {"stride_b", required_argument, nullptr, 't'},
                                           {"stride_c", required_argument, nullptr, 'u'},
                                           {"repeat", required_argument, nullptr, 'r'},
                                           {"warmup", required_argument, nullptr, 'w'},
                                           {"validate", no_argument, nullptr, 'v'},
                                           {"help", no_argument, nullptr, 'h'},
                                           {nullptr, 0, nullptr, 0}};

    int opt;
    int option_index = 0;
    while((opt = getopt_long(argc, argv, "m:n:k:b:s:t:u:r:w:vh", long_options, &option_index)) !=
          -1)
    {
        switch(opt)
        {
        case 'm': M = std::atoi(optarg); break;
        case 'n': N = std::atoi(optarg); break;
        case 'k': K = std::atoi(optarg); break;
        case 'b': k_batch = std::atoi(optarg); break;
        case 's': stride_A = std::atoi(optarg); break;
        case 't': stride_B = std::atoi(optarg); break;
        case 'u': stride_C = std::atoi(optarg); break;
        case 'r': repeat = std::atoi(optarg); break;
        case 'w': warmup = std::atoi(optarg); break;
        case 'v': validate = true; break;
        case 'h': print_help(argv[0]); return 0;
        default: print_help(argv[0]); return 1;
        }
    }

    // Auto-calculate strides if not provided
    if(stride_A == 0)
    {
        stride_A = K; // Assuming row-major for A
    }
    if(stride_B == 0)
    {
        stride_B = N; // Assuming row-major for B
    }
    if(stride_C == 0)
    {
        stride_C = N; // Assuming row-major for C
    }

    // Initialize GPU
    int device_count = 0;
    if(hipGetDeviceCount(&device_count) != hipSuccess)
    {
        std::cerr << "Failed to get device count\n";
        return 1;
    }

    if(device_count <= 0)
    {
        std::cerr << "No GPU devices found\n";
        return 1;
    }

    // Create stream config
    ck_tile::stream_config stream;
    stream.flush_cache_    = false;
    stream.rotating_count_ = 8;
    stream.log_level_      = 0;
    stream.cold_niters_    = warmup;
    stream.nrepeat_        = repeat;

    // Allocate device memory
    size_t size_a = M * stride_A * sizeof(ADataType);
    size_t size_b = K * stride_B * sizeof(BDataType);
    size_t size_c = M * stride_C * sizeof(CDataType);

    void* d_a;
    void* d_b;
    void* d_c;

    if(hipMalloc(&d_a, size_a) != hipSuccess)
    {
        std::cerr << "Failed to allocate device memory for matrix A\n";
        return 1;
    }
    if(hipMalloc(&d_b, size_b) != hipSuccess)
    {
        std::cerr << "Failed to allocate device memory for matrix B\n";
        static_cast<void>(hipFree(d_a));
        return 1;
    }
    if(hipMalloc(&d_c, size_c) != hipSuccess)
    {
        std::cerr << "Failed to allocate device memory for matrix C\n";
        static_cast<void>(hipFree(d_a));
        static_cast<void>(hipFree(d_b));
        return 1;
    }

    // Initialize with random data
    void* h_a = malloc(size_a);
    void* h_b = malloc(size_b);

    initialize_tensor_random<ADataType>(h_a, M * stride_A);
    initialize_tensor_random<BDataType>(h_b, K * stride_B);

    if(hipMemcpy(d_a, h_a, size_a, hipMemcpyHostToDevice) != hipSuccess)
    {
        std::cerr << "Failed to copy data to device for matrix A\n";
        free(h_a);
        free(h_b);
        static_cast<void>(hipFree(d_a));
        static_cast<void>(hipFree(d_b));
        static_cast<void>(hipFree(d_c));
        return 1;
    }
    if(hipMemcpy(d_b, h_b, size_b, hipMemcpyHostToDevice) != hipSuccess)
    {
        std::cerr << "Failed to copy data to device for matrix B\n";
        free(h_a);
        free(h_b);
        static_cast<void>(hipFree(d_a));
        static_cast<void>(hipFree(d_b));
        static_cast<void>(hipFree(d_c));
        return 1;
    }
    if(hipMemset(d_c, 0, size_c) != hipSuccess)
    {
        std::cerr << "Failed to memset device memory for matrix C\n";
        free(h_a);
        free(h_b);
        static_cast<void>(hipFree(d_a));
        static_cast<void>(hipFree(d_b));
        static_cast<void>(hipFree(d_c));
        return 1;
    }

    free(h_a);
    free(h_b);

    if(validate)
    {
        std::cout << "Validation mode enabled (not fully implemented)\n";
    }

    // Create GEMM arguments
    ck_tile::GemmHostArgs args{
        d_a,      // a_ptr
        d_b,      // b_ptr
        d_c,      // c_ptr
        M,        // M
        N,        // N
        K,        // K
        stride_A, // stride_A
        stride_B, // stride_B
        stride_C, // stride_C
        k_batch   // k_batch
    };

    // Run the kernel
    std::cout << "Running kernel: " << KERNEL_NAME << "\n";
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K;
    if(k_batch > 1)
    {
        std::cout << ", k_batch=" << k_batch;
    }
    std::cout << "\n";

    try
    {
        float time = SelectedKernel::launch(args, stream);

        // Calculate performance metrics
        double flops  = 2.0 * M * N * K * k_batch;
        double tflops = flops / (time * 1e-3) / 1e12;

        std::cout << "Time: " << time << " ms\n";
        std::cout << "Performance: " << tflops << " TFLOPS\n";
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error running kernel: " << e.what() << "\n";
        static_cast<void>(hipFree(d_a));
        static_cast<void>(hipFree(d_b));
        static_cast<void>(hipFree(d_c));
        return 1;
    }

    // Cleanup
    static_cast<void>(hipFree(d_a));
    static_cast<void>(hipFree(d_b));
    static_cast<void>(hipFree(d_c));

    return 0;
}
