// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm.hpp"

#include "run_gemm_example.inc"

int run_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "R")
    {
        if(data_type == "fp16")
        {
            return run_gemm_example_with_layouts<ck_tile::half_t>(argc, argv, Row{}, Row{}, Row{});
        }
        else if(data_type == "bf16")
        {
            return run_gemm_example_with_layouts<ck_tile::bf16_t>(argc, argv, Row{}, Row{}, Row{});
        }
        else if(data_type == "fp8")
        {
            return run_gemm_example_with_layouts<ck_tile::fp8_t>(argc, argv, Row{}, Row{}, Row{});
        }
        else if(data_type == "bf8")
        {
            return run_gemm_example_with_layouts<ck_tile::bf8_t>(argc, argv, Row{}, Row{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
        }
    }
    else if(a_layout == "R" && b_layout == "C")
    {
        if(data_type == "fp16")
        {
            return run_gemm_example_with_layouts<ck_tile::half_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf16")
        {
            return run_gemm_example_with_layouts<ck_tile::bf16_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "fp8")
        {
            return run_gemm_example_with_layouts<ck_tile::fp8_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf8")
        {
            return run_gemm_example_with_layouts<ck_tile::bf8_t>(argc, argv, Row{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
        }
    }
    else if(a_layout == "C" && b_layout == "C")
    {
        if(data_type == "fp16")
        {
            return run_gemm_example_with_layouts<ck_tile::half_t>(argc, argv, Col{}, Col{}, Row{});
        }
        else if(data_type == "bf16")
        {
            return run_gemm_example_with_layouts<ck_tile::bf16_t>(argc, argv, Col{}, Col{}, Row{});
        }
        else if(data_type == "fp8")
        {
            return run_gemm_example_with_layouts<ck_tile::fp8_t>(argc, argv, Col{}, Col{}, Row{});
        }
        else if(data_type == "bf8")
        {
            return run_gemm_example_with_layouts<ck_tile::bf8_t>(argc, argv, Col{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
        }
    }
    else if(a_layout == "C" && b_layout == "R")
    {
        if(data_type == "fp16")
        {
            return run_gemm_example_with_layouts<ck_tile::half_t>(argc, argv, Col{}, Row{}, Row{});
        }
        else if(data_type == "bf16")
        {
            return run_gemm_example_with_layouts<ck_tile::bf16_t>(argc, argv, Col{}, Row{}, Row{});
        }
        else if(data_type == "fp8")
        {
            return run_gemm_example_with_layouts<ck_tile::fp8_t>(argc, argv, Col{}, Row{}, Row{});
        }
        else if(data_type == "bf8")
        {
            return run_gemm_example_with_layouts<ck_tile::bf8_t>(argc, argv, Col{}, Row{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
