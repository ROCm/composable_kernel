// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/batchnorm2d/kernel/batchnorm2d_simple.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// CPU reference implementation
void batchnorm_cpu_reference(const std::vector<float>& x,
                             std::vector<float>& y,
                             const std::vector<float>& scale,
                             const std::vector<float>& bias,
                             int m,
                             int k,
                             float epsilon)
{
    for(int ch = 0; ch < m; ch++)
    {
        // Compute mean
        float sum = 0.0f;
        for(int i = 0; i < k; i++)
        {
            sum += x[ch * k + i];
        }
        float mean = sum / k;
        
        // Compute variance
        float var_sum = 0.0f;
        for(int i = 0; i < k; i++)
        {
            float diff = x[ch * k + i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / k;
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for(int i = 0; i < k; i++)
        {
            float normalized = (x[ch * k + i] - mean) * inv_std;
            y[ch * k + i] = scale[ch] * normalized + bias[ch];
        }
    }
}

// Initialize data with random values
void init_data(std::vector<float>& data, int seed = 42)
{
    std::srand(seed);
    for(auto& val : data)
    {
        val = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }
}

// Compare results
bool compare_results(const std::vector<float>& y_gpu,
                    const std::vector<float>& y_cpu,
                    float rtol = 1e-4f,
                    float atol = 1e-5f)
{
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int error_count = 0;
    
    for(size_t i = 0; i < y_gpu.size(); i++)
    {
        float diff = std::abs(y_gpu[i] - y_cpu[i]);
        float rel_diff = diff / (std::abs(y_cpu[i]) + 1e-10f);
        
        max_diff = std::max(max_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
        
        if(diff > atol && rel_diff > rtol)
        {
            if(error_count < 10)  // Print first 10 errors
            {
                std::cout << "Mismatch at index " << i 
                         << ": GPU=" << y_gpu[i]
                         << ", CPU=" << y_cpu[i]
                         << ", diff=" << diff << std::endl;
            }
            error_count++;
        }
    }
    
    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;
    std::cout << "Errors: " << error_count << " / " << y_gpu.size() << std::endl;
    
    return error_count == 0;
}

int main()
{
    std::cout << "=== BatchNorm2D Simple POC ===" << std::endl;
    
    // Problem size
    constexpr int M = 4;    // Channels
    constexpr int K = 256;  // Elements per channel
    constexpr float epsilon = 1e-5f;
    
    std::cout << "M (channels): " << M << std::endl;
    std::cout << "K (elements per channel): " << K << std::endl;
    std::cout << "Total elements: " << M * K << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    
    // Allocate host memory
    std::vector<float> x_host(M * K);
    std::vector<float> y_host(M * K);
    std::vector<float> y_ref(M * K);
    std::vector<float> scale_host(M);
    std::vector<float> bias_host(M);
    
    // Initialize data
    init_data(x_host, 42);
    init_data(scale_host, 43);
    init_data(bias_host, 44);
    
    // Set scale and bias to simple values for easier debugging
    for(int i = 0; i < M; i++)
    {
        scale_host[i] = 1.0f;  // No scaling
        bias_host[i] = 0.0f;   // No bias
    }
    
    std::cout << "\n=== CPU Reference ===" << std::endl;
    batchnorm_cpu_reference(x_host, y_ref, scale_host, bias_host, M, K, epsilon);
    
    std::cout << "Sample input (channel 0, first 5 elements):" << std::endl;
    for(int i = 0; i < 5; i++)
    {
        std::cout << "  x[" << i << "] = " << x_host[i] << std::endl;
    }
    
    std::cout << "Sample output (channel 0, first 5 elements):" << std::endl;
    for(int i = 0; i < 5; i++)
    {
        std::cout << "  y_ref[" << i << "] = " << y_ref[i] << std::endl;
    }
    
    // Allocate device memory
    float* x_dev;
    float* y_dev;
    float* scale_dev;
    float* bias_dev;
    
    hipMalloc(&x_dev, M * K * sizeof(float));
    hipMalloc(&y_dev, M * K * sizeof(float));
    hipMalloc(&scale_dev, M * sizeof(float));
    hipMalloc(&bias_dev, M * sizeof(float));
    
    // Copy to device
    hipMemcpy(x_dev, x_host.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(scale_dev, scale_host.data(), M * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(bias_dev, bias_host.data(), M * sizeof(float), hipMemcpyHostToDevice);
    
    std::cout << "\n=== GPU Kernel ===" << std::endl;
    
    // Prepare kernel arguments
    using Kernel = ck_tile::BatchNorm2dSimple<256>;
    
    typename Kernel::HostArgs hargs;
    hargs.p_x = x_dev;
    hargs.p_y = y_dev;
    hargs.p_scale = scale_dev;
    hargs.p_bias = bias_dev;
    hargs.epsilon = epsilon;
    hargs.m = M;
    hargs.k = K;
    
    auto kargs = Kernel::MakeKargs(hargs);
    auto grid_size = Kernel::GridSize(hargs);
    auto block_size = Kernel::BlockSize();
    
    std::cout << "Grid size: " << grid_size.x << std::endl;
    std::cout << "Block size: " << block_size << std::endl;
    std::cout << "Kernel name: " << Kernel::GetName() << std::endl;
    
    // Launch kernel
    ck_tile::kernel_batchnorm2d_simple<256><<<grid_size, block_size>>>(kargs);
    
    // Check for launch errors
    hipError_t launch_err = hipGetLastError();
    if(launch_err != hipSuccess)
    {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(launch_err) << std::endl;
        return -1;
    }
    
    // Wait for kernel to complete
    hipError_t sync_err = hipDeviceSynchronize();
    if(sync_err != hipSuccess)
    {
        std::cerr << "Kernel execution failed: " << hipGetErrorString(sync_err) << std::endl;
        return -1;
    }
    
    std::cout << "Kernel executed successfully!" << std::endl;
    
    // Copy result back
    hipMemcpy(y_host.data(), y_dev, M * K * sizeof(float), hipMemcpyDeviceToHost);
    
    std::cout << "\nSample GPU output (channel 0, first 5 elements):" << std::endl;
    for(int i = 0; i < 5; i++)
    {
        std::cout << "  y_gpu[" << i << "] = " << y_host[i] << std::endl;
    }
    
    // Compare results
    std::cout << "\n=== Verification ===" << std::endl;
    bool passed = compare_results(y_host, y_ref);
    
    // Cleanup
    hipFree(x_dev);
    hipFree(y_dev);
    hipFree(scale_dev);
    hipFree(bias_dev);
    
    if(passed)
    {
        std::cout << "\n✓ TEST PASSED!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\n✗ TEST FAILED!" << std::endl;
        return -1;
    }
}
