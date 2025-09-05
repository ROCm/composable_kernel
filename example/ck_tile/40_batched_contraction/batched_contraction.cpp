// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp"
#include "contraction_utils.hpp"
#include "run_batched_contraction_example.inc"

// Core kernel launcher function - this is the main kernel interface
template <typename ADataType, typename BDataType, typename EDataType>
float batched_contraction(const ck_tile::index_t M,
                          const ck_tile::index_t N,
                          const ck_tile::index_t K,
                          const ck_tile::index_t batch_count,
                          const void* a_ptr,
                          const void* b_ptr,
                          void* e_ptr,
                          const ck_tile::stream_config& s)
{
    // Define problem
    using Problem = ck_tile::BatchedContractionProblem<ADataType,
                                                       BDataType,
                                                       EDataType,
                                                       1, // NumDimG
                                                       1, // NumDimM
                                                       1, // NumDimN
                                                       1  // NumDimK
                                                       >;

    using Kernel = ck_tile::BatchedContractionKernel<Problem>;

    // Prepare kernel arguments
    typename Kernel::Kargs kargs;
    kargs.p_a            = static_cast<const ADataType*>(a_ptr);
    kargs.p_b            = static_cast<const BDataType*>(b_ptr);
    kargs.p_e            = static_cast<EDataType*>(e_ptr);
    kargs.M              = M;
    kargs.N              = N;
    kargs.K              = K;
    kargs.batch_count    = batch_count;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_e_batch = M * N;

    // Calculate grid and block dimensions
    const auto grids  = Kernel::GridSize(M, N, batch_count);
    const auto blocks = Kernel::BlockSize();

    // Check if arguments are supported
    if(!Kernel::IsSupportedArguments())
    {
        throw std::runtime_error("Arguments not supported! Skipping batched contraction!");
    }

    // Logging
    if(s.log_level_ > 0)
    {
        std::cout << "Launching BatchedContractionKernel:" << " M=" << M << " N=" << N << " K=" << K
                  << " batch=" << batch_count << " grid=(" << grids.x << "," << grids.y << ","
                  << grids.z << ")" << " block=(" << blocks.x << "," << blocks.y << "," << blocks.z
                  << ")" << std::endl;
    }

    // Launch kernel
    float ave_time =
        ck_tile::launch_kernel(s, ck_tile::make_kernel(Kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

// Function to handle different data types
template <typename DataType>
int run_batched_contraction_example_prec_type(ck_tile::ArgParser& arg_parser)
{
    // Use type config to get proper accumulation type
    using TypeConfig = BatchedContractionTypeConfig<DataType>;
    return run_batched_contraction_example<typename TypeConfig::ADataType,
                                           typename TypeConfig::BDataType,
                                           typename TypeConfig::EDataType>(arg_parser);
}

// Main function
int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    try
    {
        std::string data_type = arg_parser.get_str("prec");

        if(data_type == "fp32")
        {
            return !run_batched_contraction_example_prec_type<float>(arg_parser);
        }
        else if(data_type == "fp16")
        {
            return !run_batched_contraction_example_prec_type<ck_tile::half_t>(arg_parser);
        }
        else if(data_type == "bf16")
        {
            return !run_batched_contraction_example_prec_type<ck_tile::bf16_t>(arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported data type: " + data_type);
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
