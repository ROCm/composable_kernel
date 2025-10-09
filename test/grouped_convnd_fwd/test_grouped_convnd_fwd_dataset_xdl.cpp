// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>          // Standard C library (exit codes, malloc)
#include <iostream>         // C++ I/O streams (cout, cerr)
#include <initializer_list> // C++ initializer list support (unused here)
#include <vector>           // C++ vector container - stores test cases
#include <fstream>          // File I/O for CSV reading
#include <sstream>          // String stream for CSV parsing
#include <string>           // String operations
#include <gtest/gtest.h>    // Google Test framework - provides TYPED_TEST, EXPECT_TRUE

#include "profiler/profile_grouped_conv_fwd_impl.hpp" // The actual GPU profiler that does convolution work

// CSV Reader Function for Loading Test Cases
// Reads convolution parameters from CSV file and returns vector of ConvParam structures
std::vector<ck::utils::conv::ConvParam> load_csv_test_cases(const std::string& filename)
{
    std::vector<ck::utils::conv::ConvParam> conv_params; // Return vector
    std::ifstream file(filename);                        // Open CSV file

    if(!file.is_open())
    {
        std::cerr << "ERROR: Cannot open CSV file: " << filename << std::endl;
        return conv_params; // Return empty vector on error
    }

    std::string line;
    int line_number = 0;

    // Read file line by line
    while(std::getline(file, line))
    {
        line_number++;
        // Skip comment lines (starting with #) and empty lines
        if(line.empty() || line[0] == '#')
        {
            continue;
        }

        // Skip header line (contains column names)
        if(line.find("NDim,Groups,BatchSize") != std::string::npos)
        {
            continue;
        }

        // Parse CSV line using stringstream
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        // Split line by commas
        while(std::getline(ss, cell, ','))
        {
            row.push_back(cell);
        }

        // Validate row has correct number of columns
        if(row.size() < 19)
        { // Need at least 19 columns for 2D (excluding TestName)
            std::cerr << "WARNING: Line " << line_number << " has insufficient columns ("
                      << row.size() << "), skipping" << std::endl;
            continue;
        }

        try
        {
            // Parse CSV data into ConvParam structure
            // CSV Format:
            // NDim,Groups,BatchSize,OutChannels,InChannels,KernelH,KernelW,InputH,InputW,OutputH,OutputW,StrideH,StrideW,DilationH,DilationW,LeftPadH,LeftPadW,RightPadH,RightPadW,TestName
            int NDim        = std::stoi(row[0]);
            int Groups      = std::stoi(row[1]);
            int BatchSize   = std::stoi(row[2]);
            int OutChannels = std::stoi(row[3]);
            int InChannels  = std::stoi(row[4]);

            if(NDim == 2)
            {
                // 2D Convolution: {NDim, Groups, BatchSize, OutChannels, InChannels,
                // {KernelH,KernelW}, {InputH,InputW}, {StrideH,StrideW}, {DilationH,DilationW},
                // {LeftPadH,LeftPadW}, {RightPadH,RightPadW}}
                ck::utils::conv::ConvParam param = {
                    NDim,                                     // NDim = 2
                    Groups,                                   // Groups
                    BatchSize,                                // Batch size
                    OutChannels,                              // Output channels
                    InChannels,                               // Input channels
                    {std::stoi(row[5]), std::stoi(row[6])},   // Kernel: {H, W}
                    {std::stoi(row[7]), std::stoi(row[8])},   // Input: {H, W}
                    {std::stoi(row[11]), std::stoi(row[12])}, // Stride: {H, W}
                    {std::stoi(row[13]), std::stoi(row[14])}, // Dilation: {H, W}
                    {std::stoi(row[15]), std::stoi(row[16])}, // Left pad: {H, W}
                    {std::stoi(row[17]), std::stoi(row[18])}  // Right pad: {H, W}
                };
                conv_params.push_back(param);
            }
            else if(NDim == 3)
            {
                // 3D Convolution: Need more columns for 3D parameters
                if(row.size() < 26)
                {
                    std::cerr << "WARNING: 3D convolution on line " << line_number
                              << " needs 26+ columns, has " << row.size() << ", skipping"
                              << std::endl;
                    continue;
                }
                // 3D Convolution: {NDim, Groups, BatchSize, OutChannels, InChannels,
                // {KernelD,KernelH,KernelW}, {InputD,InputH,InputW}, {OutputD,OutputH,OutputW},
                // {StrideD,StrideH,StrideW}, {DilationD,DilationH,DilationW},
                // {LeftPadD,LeftPadH,LeftPadW}, {RightPadD,RightPadH,RightPadW}}
                ck::utils::conv::ConvParam param = {
                    NDim,                                                       // NDim = 3
                    Groups,                                                     // Groups
                    BatchSize,                                                  // Batch size
                    OutChannels,                                                // Output channels
                    InChannels,                                                 // Input channels
                    {std::stoi(row[5]), std::stoi(row[6]), std::stoi(row[7])},  // Kernel: {D, H, W}
                    {std::stoi(row[8]), std::stoi(row[9]), std::stoi(row[10])}, // Input: {D, H, W}
                    {std::stoi(row[14]),
                     std::stoi(row[15]),
                     std::stoi(row[16])}, // Stride: {D, H, W}
                    {std::stoi(row[17]),
                     std::stoi(row[18]),
                     std::stoi(row[19])}, // Dilation: {D, H, W}
                    {std::stoi(row[20]),
                     std::stoi(row[21]),
                     std::stoi(row[22])}, // Left pad: {D, H, W}
                    {std::stoi(row[23]),
                     std::stoi(row[24]),
                     std::stoi(row[25])} // Right pad: {D, H, W}
                };
                conv_params.push_back(param);
            }
            else
            {
                std::cerr << "WARNING: Unsupported NDim=" << NDim << " on line " << line_number
                          << ", skipping" << std::endl;
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << "ERROR: Failed to parse line " << line_number << ": " << e.what()
                      << std::endl;
            continue;
        }
    }

    file.close();
    std::cout << "Loaded " << conv_params.size() << " test cases from " << filename << std::endl;
    return conv_params;
}

// Template class that works with different data types and tensor layouts
template <typename Tuple>
class TestGroupedConvndFwd : public ::testing::Test // Inherit from Google Test base class
{
    protected:
    using DataType =
        std::tuple_element_t<0, Tuple>; // Extract data type from tuple (fp32, fp16, bf16, int8)
    using InLayout =
        std::tuple_element_t<1, Tuple>; // Extract input tensor layout (NHWGC, NDHWGC, etc.)
    using WeiLayout =
        std::tuple_element_t<2, Tuple>; // Extract weight tensor layout (GKYXC, GKZYXC, etc.)
    using OutLayout =
        std::tuple_element_t<3, Tuple>; // Extract output tensor layout (NHWGK, NDHWGK, etc.)
    using IndexType = ck::long_index_t; // 64-bit integer type for tensor dimensions

    // THE KEY CONTAINER: This stores all test case parameters
    // Each test will push_back() ConvParam structures here
    std::vector<ck::utils::conv::ConvParam> conv_params;

    // Template function to run tests for N-dimensional spatial convolution (2D or 3D)
    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty()); // Google Test assertion: ensure we have test cases
        bool pass = true;                  // Track overall pass/fail across all test cases

        // MAIN LOOP: Execute every test case that was added to conv_params
        for(auto& param : conv_params)
        {
            // CALL THE ACTUAL GPU PROFILER - This is where convolution happens!
            pass = pass &&
                   ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                               InLayout,   // Input tensor layout
                                                               WeiLayout,  // Weight tensor layout
                                                               OutLayout,  // Output tensor layout
                                                               DataType,   // Input data type
                                                               DataType,   // Weight data type
                                                               DataType,   // Output data type
                                                               DataType,   // Accumulation type
                                                               DataType,   // Bias type
                                                               IndexType>( // Index type (int64)
                       true, // do_verification: Compare GPU result with CPU reference
                       1, // init_method: How to initialize random test data (1 = uniform -5 to 5)
                       false,  // do_log: Don't print detailed tensor values
                       false,  // time_kernel: Don't do performance timing (just correctness)
                       param); // ConvParam: {NDim, Groups, Batch, OutChannels, InChannels,
                               // KernelSize, InputSize, ...}
        }
        EXPECT_TRUE(pass); // Google Test assertion: ALL test cases must pass
    }
};

using namespace ck::tensor_layout::convolution; // Import tensor layout names (NHWGC, GKYXC, etc.)

// GOOGLE TEST TYPE COMBINATIONS: Define what data types and layouts to test
// This creates 4 separate test instances for 2D convolution:
using KernelTypes2d =
    ::testing::Types<std::tuple<float, NHWGC, GKYXC, NHWGK>,       // fp32 test
                     std::tuple<ck::half_t, NHWGC, GKYXC, NHWGK>,  // fp16 test
                     std::tuple<ck::bhalf_t, NHWGC, GKYXC, NHWGK>, // bfloat16 test
                     std::tuple<int8_t, NHWGC, GKYXC, NHWGK>>;     // int8 test

// This creates 3 separate test instances for 3D convolution (no int8 support for 3D):
using KernelTypes3d =
    ::testing::Types<std::tuple<float, NDHWGC, GKZYXC, NDHWGK>,        // fp32 3D test
                     std::tuple<ck::half_t, NDHWGC, GKZYXC, NDHWGK>,   // fp16 3D test
                     std::tuple<ck::bhalf_t, NDHWGC, GKZYXC, NDHWGK>>; // bfloat16 3D test

// Create specialized test classes that inherit from the base template class
template <typename Tuple>
class TestGroupedConvndFwd2d : public TestGroupedConvndFwd<Tuple> // 2D convolution test class
{
};

template <typename Tuple>
class TestGroupedConvndFwd3d : public TestGroupedConvndFwd<Tuple> // 3D convolution test class
{
};

// GOOGLE TEST MAGIC: Create test suites
// This tells Google Test to create 4 test instances for 2D (fp32, fp16, bf16, int8)
TYPED_TEST_SUITE(TestGroupedConvndFwd2d, KernelTypes2d);
// This tells Google Test to create 3 test instances for 3D (fp32, fp16, bf16)
TYPED_TEST_SUITE(TestGroupedConvndFwd3d, KernelTypes3d);

// THE ACTUAL 2D TEST - This runs 4 times (once for each data type: fp32, fp16, bf16, int8)
TYPED_TEST(TestGroupedConvndFwd2d, Test2D)
{
    // LOAD TEST CASES FROM CSV FILE instead of hardcoded cases
    // Try different locations for the CSV file (build directory vs source directory)
    std::vector<std::string> csv_paths = {
        "../test_data/conv_test_set_2d_dataset.csv", // From build directory to source
    };

    bool loaded = false;
    for(const auto& csv_path : csv_paths)
    {
        auto csv_cases = load_csv_test_cases(csv_path);
        if(!csv_cases.empty())
        {
            // Successfully loaded CSV data - add all test cases to conv_params
            for(const auto& test_case : csv_cases)
            {
                this->conv_params.push_back(test_case);
            }
            std::cout << "Loaded " << csv_cases.size() << " 2D test cases from " << csv_path
                      << std::endl;
            loaded = true;
            break;
        }
    }

    // FAIL if CSV loading fails - no fallback!
    if(!loaded)
    {
        std::cerr << "ERROR: Failed to load CSV test data from any of these locations:"
                  << std::endl;
        for(const auto& path : csv_paths)
        {
            std::cerr << "  - " << path << std::endl;
        }
        std::cerr << "\nPlease ensure CSV test data exists in one of these locations." << std::endl;
        std::cerr << "Run generate_test_dataset.sh in test_data/ to create test datasets."
                  << std::endl;

        // Force test failure - no test cases means test should fail
        EXPECT_TRUE(loaded) << "CSV test data loading failed";
    }

    // Execute all test cases with 2D convolution
    // This calls Run<2>() which loops through conv_params and calls GPU profiler for each
    this->template Run<2>();
}

// THE ACTUAL 3D TEST - This runs 3 times (once for each data type: fp32, fp16, bf16)
TYPED_TEST(TestGroupedConvndFwd3d, Test3D)
{
    // LOAD TEST CASES FROM CSV FILE instead of hardcoded cases
    // Try different locations for the CSV file (build directory vs source directory)
    std::vector<std::string> csv_paths = {
        "../test_data/conv_test_set_3d_dataset.csv", // From build directory to source
    };

    bool loaded = false;
    for(const auto& csv_path : csv_paths)
    {
        auto csv_cases = load_csv_test_cases(csv_path);
        if(!csv_cases.empty())
        {
            // Successfully loaded CSV data - add all test cases to conv_params
            for(const auto& test_case : csv_cases)
            {
                this->conv_params.push_back(test_case);
            }
            std::cout << "Loaded " << csv_cases.size() << " 3D test cases from " << csv_path
                      << std::endl;
            loaded = true;
            break;
        }
    }

    // FAIL if CSV loading fails - no fallback!
    if(!loaded)
    {
        std::cerr << "ERROR: Failed to load CSV test data from any of these locations:"
                  << std::endl;
        for(const auto& path : csv_paths)
        {
            std::cerr << "  - " << path << std::endl;
        }
        std::cerr << "\nPlease ensure CSV test data exists in one of these locations." << std::endl;
        std::cerr << "Run generate_test_dataset.sh in test_data/ to create test datasets."
                  << std::endl;

        // Force test failure - no test cases means test should fail
        EXPECT_TRUE(loaded) << "CSV test data loading failed";
    }

    // Execute all test cases with 3D convolution
    // This calls Run<3>() which loops through conv_params and calls GPU profiler for each
    this->template Run<3>();
}
