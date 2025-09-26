// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

// Include gemm_common.hpp for DataTypeTraits and other utilities
#include "tile_engine/ops/gemm/gemm_common.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct, KERNEL_NAME, and data types
// Following tile_engine's exact pattern

class GemmTileEngineTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
        // Use exact tile dimensions to ensure proper grid calculation
        // The kernel tiles are 128x128x64, so use exact multiples
        m_ = 128;   // Exactly tile_m
        n_ = 128;   // Exactly tile_n 
        k_ = 64;    // Exactly tile_k
        
        // Calculate strides (following tile_engine pattern)
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>) {
            stride_a_ = k_;
        } else {
            stride_a_ = m_;
        }
        
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>) {
            stride_b_ = n_;
        } else {
            stride_b_ = k_;
        }
        
        if constexpr(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>) {
            stride_c_ = n_;
        } else {
            stride_c_ = m_;
        }
    }

    // Test dimensions
    int m_, n_, k_;
    int stride_a_, stride_b_, stride_c_;
};

TEST_F(GemmTileEngineTest, BasicFunctionality)
{
    std::cout << "=== DEBUG: Starting BasicFunctionality test ===" << std::endl;
    std::cout << "DEBUG: Problem dimensions - M=" << m_ << ", N=" << n_ << ", K=" << k_ << std::endl;
    std::cout << "DEBUG: Strides - A=" << stride_a_ << ", B=" << stride_b_ << ", C=" << stride_c_ << std::endl;
    
    // Following tile_engine's EXACT pattern from gemm_profiler.hpp
    const ALayout layout_a = ALayout{};
    const BLayout layout_b = BLayout{};
    const CLayout layout_c = CLayout{};
    
    // Calculate proper strides using tile_engine's method
    int split_k = 1;
    int stride_a_calc = ck_tile::get_default_stride(m_, k_, 0, is_row_major(layout_a));
    int stride_b_calc = ck_tile::get_default_stride(k_, n_, 0, is_row_major(layout_b));
    int stride_c_calc = ck_tile::get_default_stride(m_, n_, 0, is_row_major(layout_c));
    
    std::cout << "DEBUG: Layout info - A: ";
    if constexpr(is_row_major(layout_a)) {
        std::cout << "RowMajor";
    } else {
        std::cout << "ColMajor";
    }
    std::cout << ", B: ";
    if constexpr(is_row_major(layout_b)) {
        std::cout << "RowMajor";
    } else {
        std::cout << "ColMajor";
    }
    std::cout << ", C: ";
    if constexpr(is_row_major(layout_c)) {
        std::cout << "RowMajor";
    } else {
        std::cout << "ColMajor";
    }
    std::cout << std::endl;
    
    std::cout << "DEBUG: Calculated strides - A=" << stride_a_calc 
              << ", B=" << stride_b_calc << ", C=" << stride_c_calc << std::endl;

    // Create HostTensors with proper descriptors (following tile_engine pattern)
    std::cout << "DEBUG: Creating HostTensors..." << std::endl;
    ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
        m_, k_, stride_a_calc, is_row_major(layout_a)));
    ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
        k_, n_, stride_b_calc, is_row_major(layout_b)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(ck_tile::host_tensor_descriptor(
        m_, n_, stride_c_calc, is_row_major(layout_c)));
    ck_tile::HostTensor<CDataType> c_m_n_host_result(ck_tile::host_tensor_descriptor(
        m_, n_, stride_c_calc, is_row_major(layout_c)));
    
    std::cout << "DEBUG: HostTensors created successfully" << std::endl;

    // Initialize tensors using tile_engine's FillConstant method for verification
    std::cout << "DEBUG: Initializing tensors with tile_engine's fill functions..." << std::endl;
    ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_k_n);
    
    // Calculate reference result: 1*1*k for each element
    ck_tile::FillConstant<CDataType>{static_cast<CDataType>(k_)}(c_m_n_host_result);
    
    std::cout << "DEBUG: Tensors initialized - Expected result per element: " << k_ << std::endl;

    // Allocate device memory using tile_engine's pattern
    std::cout << "DEBUG: Allocating device memory..." << std::endl;
    try {
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());
        
        std::cout << "DEBUG: Device memory allocated successfully" << std::endl;

        // Copy to device using tile_engine's pattern
        std::cout << "DEBUG: Copying data to device..." << std::endl;
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();
        
        std::cout << "DEBUG: Data copied to device successfully" << std::endl;

        // Create GEMM arguments using constructor (fixes potential aggregate initialization bug)
        // Constructor order: a_ptr, b_ptr, e_ptr, k_batch, M, N, K, stride_A, stride_B, stride_E
        ck_tile::GemmHostArgs gemm_args(
            a_m_k_dev_buf.GetDeviceBuffer(),
            b_k_n_dev_buf.GetDeviceBuffer(),
            c_m_n_dev_buf.GetDeviceBuffer(),
            split_k,  // k_batch comes FIRST in constructor
            m_,
            n_,
            k_,
            stride_a_calc,
            stride_b_calc,
            stride_c_calc
        );
        
        std::cout << "DEBUG: GemmHostArgs created with pointers - A=" << gemm_args.a_ptr 
                  << ", B=" << gemm_args.b_ptr << ", C=" << gemm_args.c_ptr << std::endl;

        // Create stream_config exactly like tile_engine with proper timing parameters
        std::cout << "DEBUG: Creating stream config..." << std::endl;
        ck_tile::stream_config stream_config{nullptr,    // stream
                                             true,       // time_kernel
                                             1,          // log_level (enable logging)
                                             2,          // n_warmup 
                                             5,          // n_repeat
                                             true,       // is_gpu_timer
                                             false,      // flush_cache
                                             100};       // rotating_count
        
        // Launch kernel exactly like tile_engine
        std::cout << "DEBUG: Launching kernel with proper stream config..." << std::endl;
        float kernel_time = 0.0f;
        bool launch_successful = false;
        
        try {
            kernel_time = SelectedKernel::launch(gemm_args, stream_config);
            launch_successful = true;
            std::cout << "DEBUG: Kernel launch SUCCESS - execution time: " << kernel_time << " ms" << std::endl;
        } catch(const std::runtime_error& e) {
            std::cout << "DEBUG: Kernel launch FAILED - " << e.what() << std::endl;
            FAIL() << "Kernel arguments not supported: " << e.what();
        } catch(const std::exception& e) {
            std::cout << "DEBUG: Kernel launch FAILED - " << e.what() << std::endl;
            FAIL() << "Kernel launch exception: " << e.what();
        }
        
        ASSERT_TRUE(launch_successful) << "Kernel launch failed";
        EXPECT_GT(kernel_time, 0.0f) << "Kernel execution time should be positive, got: " << kernel_time;

        // Copy result back using tile_engine's pattern
        std::cout << "DEBUG: Copying result back from device..." << std::endl;
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        std::cout << "DEBUG: Result copied back successfully" << std::endl;

        // Print first few results for debugging
        std::cout << "DEBUG: First 10 result values: ";
        for(int i = 0; i < std::min(10, static_cast<int>(c_m_n_dev_result.get_element_space_size())); ++i) {
            std::cout << static_cast<float>(c_m_n_dev_result.data()[i]) << " ";
        }
        std::cout << std::endl;

        // Verify results using tile_engine's comparison approach
        std::cout << "DEBUG: Verifying results..." << std::endl;
        const float tolerance = 1e-3f;
        int mismatches = 0;
        int total_elements = m_ * n_;
        
        for(int i = 0; i < total_elements; ++i) {
            float expected = static_cast<float>(c_m_n_host_result.data()[i]);
            float actual = static_cast<float>(c_m_n_dev_result.data()[i]);
            if(std::abs(actual - expected) > tolerance) {
                mismatches++;
                if(mismatches <= 5) { // Only print first 5 mismatches
                    std::cout << "DEBUG: Mismatch at index " << i 
                              << " expected " << expected 
                              << " got " << actual << std::endl;
                }
            }
        }
        
        std::cout << "DEBUG: Total mismatches: " << mismatches << " out of " << total_elements << std::endl;
        
        EXPECT_EQ(mismatches, 0) << "Found " << mismatches << " mismatches in GEMM results";
        
        std::cout << "DEBUG: BasicFunctionality test completed successfully!" << std::endl;
        
    } catch(const std::exception& e) {
        std::cout << "DEBUG: Exception caught: " << e.what() << std::endl;
        FAIL() << "Exception during test: " << e.what();
    } catch(...) {
        std::cout << "DEBUG: Unknown exception caught" << std::endl;
        FAIL() << "Unknown exception during test";
    }
}

TEST_F(GemmTileEngineTest, KernelInfo)
{
    // Simple test to verify kernel information is available
    EXPECT_TRUE(strlen(KERNEL_NAME) > 0) << "Kernel name should not be empty";
    
    std::cout << "Testing kernel: " << KERNEL_NAME << std::endl;
    std::cout << "Problem size: " << m_ << "x" << n_ << "x" << k_ << std::endl;
}
