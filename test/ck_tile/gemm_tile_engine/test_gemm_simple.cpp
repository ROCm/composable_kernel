// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

// The kernel header is included via the compile command line with -include flag
// It defines SelectedKernel struct, KERNEL_NAME, and data types
// Following tile_engine's exact pattern

class GemmTileEngineTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        // Simple test dimensions
        m_ = 128;
        n_ = 128;
        k_ = 64;

        // Calculate strides (following tile_engine pattern)
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            stride_a_ = k_;
        }
        else
        {
            stride_a_ = m_;
        }

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            stride_b_ = n_;
        }
        else
        {
            stride_b_ = k_;
        }

        if constexpr(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            stride_c_ = n_;
        }
        else
        {
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
    std::cout << "DEBUG: Problem dimensions - M=" << m_ << ", N=" << n_ << ", K=" << k_
              << std::endl;
    std::cout << "DEBUG: Strides - A=" << stride_a_ << ", B=" << stride_b_ << ", C=" << stride_c_
              << std::endl;

    // Create GEMM arguments (following tile_engine's GemmHostArgs pattern)
    ck_tile::GemmHostArgs args;
    args.M        = m_;
    args.N        = n_;
    args.K        = k_;
    args.stride_A = stride_a_;
    args.stride_B = stride_b_;
    args.stride_C = stride_c_;
    args.k_batch  = 1; // Single batch for basic test

    std::cout << "DEBUG: GemmHostArgs initialized successfully" << std::endl;

    // Allocate host memory
    std::cout << "DEBUG: Allocating host memory..." << std::endl;
    std::vector<ADataType> a_host(m_ * k_);
    std::vector<BDataType> b_host(k_ * n_);
    std::vector<CDataType> c_host(m_ * n_);
    std::vector<CDataType> c_ref(m_ * n_);

    std::cout << "DEBUG: Host memory allocated - A size=" << a_host.size()
              << ", B size=" << b_host.size() << ", C size=" << c_host.size() << std::endl;

    // Initialize with simple values for verification
    std::cout << "DEBUG: Initializing host data..." << std::endl;
    for(int i = 0; i < m_ * k_; ++i)
    {
        a_host[i] = static_cast<ADataType>(1.0f);
    }
    for(int i = 0; i < k_ * n_; ++i)
    {
        b_host[i] = static_cast<BDataType>(1.0f);
    }
    for(int i = 0; i < m_ * n_; ++i)
    {
        c_host[i] = static_cast<CDataType>(0.0f);
        c_ref[i]  = static_cast<CDataType>(k_); // Expected result: 1*1*k for each element
    }
    std::cout << "DEBUG: Host data initialized - Expected result per element: " << k_ << std::endl;

    // Allocate device memory
    std::cout << "DEBUG: Allocating device memory..." << std::endl;
    try
    {
        ck_tile::DeviceMem a_dev(sizeof(ADataType) * a_host.size());
        std::cout << "DEBUG: Device memory A allocated successfully" << std::endl;

        ck_tile::DeviceMem b_dev(sizeof(BDataType) * b_host.size());
        std::cout << "DEBUG: Device memory B allocated successfully" << std::endl;

        ck_tile::DeviceMem c_dev(sizeof(CDataType) * c_host.size());
        std::cout << "DEBUG: Device memory C allocated successfully" << std::endl;

        // Copy to device
        std::cout << "DEBUG: Copying data to device..." << std::endl;
        a_dev.ToDevice(a_host.data());
        std::cout << "DEBUG: Matrix A copied to device" << std::endl;

        b_dev.ToDevice(b_host.data());
        std::cout << "DEBUG: Matrix B copied to device" << std::endl;

        c_dev.ToDevice(c_host.data());
        std::cout << "DEBUG: Matrix C copied to device" << std::endl;

        // Set device pointers (using correct member names from tile_engine)
        args.a_ptr = a_dev.GetDeviceBuffer();
        args.b_ptr = b_dev.GetDeviceBuffer();
        args.c_ptr = c_dev.GetDeviceBuffer();

        std::cout << "DEBUG: Device pointers set - A=" << args.a_ptr << ", B=" << args.b_ptr
                  << ", C=" << args.c_ptr << std::endl;

        // Launch kernel (following tile_engine's exact pattern)
        std::cout << "DEBUG: Preparing to launch kernel..." << std::endl;
        ck_tile::stream_config stream_config{};
        std::cout << "DEBUG: Stream config created, launching kernel..." << std::endl;

        bool success = SelectedKernel::launch(args, stream_config);
        std::cout << "DEBUG: Kernel launch returned: " << (success ? "SUCCESS" : "FAILURE")
                  << std::endl;

        ASSERT_TRUE(success) << "Kernel launch failed";

        // Copy result back
        std::cout << "DEBUG: Copying result back from device..." << std::endl;
        c_dev.FromDevice(c_host.data());
        std::cout << "DEBUG: Result copied back successfully" << std::endl;

        // Print first few results for debugging
        std::cout << "DEBUG: First 10 result values: ";
        for(int i = 0; i < std::min(10, static_cast<int>(c_host.size())); ++i)
        {
            std::cout << static_cast<float>(c_host[i]) << " ";
        }
        std::cout << std::endl;

        // Verify results with tolerance
        std::cout << "DEBUG: Verifying results..." << std::endl;
        const float tolerance = 1e-3f;
        int mismatches        = 0;
        for(int i = 0; i < m_ * n_; ++i)
        {
            float expected = static_cast<float>(c_ref[i]);
            float actual   = static_cast<float>(c_host[i]);
            if(std::abs(actual - expected) > tolerance)
            {
                mismatches++;
                if(mismatches <= 5)
                { // Only print first 5 mismatches
                    std::cout << "DEBUG: Mismatch at index " << i << " expected " << expected
                              << " got " << actual << std::endl;
                }
            }
        }

        std::cout << "DEBUG: Total mismatches: " << mismatches << " out of " << (m_ * n_)
                  << std::endl;

        for(int i = 0; i < m_ * n_; ++i)
        {
            float expected = static_cast<float>(c_ref[i]);
            float actual   = static_cast<float>(c_host[i]);
            EXPECT_NEAR(actual, expected, tolerance)
                << "Mismatch at index " << i << " expected " << expected << " got " << actual;
        }

        std::cout << "DEBUG: BasicFunctionality test completed" << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cout << "DEBUG: Exception caught: " << e.what() << std::endl;
        FAIL() << "Exception during test: " << e.what();
    }
    catch(...)
    {
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
