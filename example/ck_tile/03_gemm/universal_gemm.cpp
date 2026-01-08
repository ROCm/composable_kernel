// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <string>

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "universal_gemm_invoker.hpp"

template <template <typename PrecType> typename GemmConfig>
int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    using Invoker = UniversalInvoker;

    if(data_type == "bf16")
    {
        // Only support RC layout (A=Row, B=Column) to reduce compile time
        if(a_layout != "R" || b_layout != "C")
        {
            throw std::runtime_error("Only RC layout (A=Row, B=Column) is supported!");
        }
        using Row = ck_tile::tensor_layout::gemm::RowMajor;
        using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
        // Directly call with fixed layout to avoid std::visit instantiating all combinations
        return run_gemm_example_with_layouts<GemmConfig<ck_tile::bf16_t>,
                                             Invoker,
                                             ck_tile::bf16_t,
                                             ck_tile::bf16_t,
                                             ck_tile::bf16_t,
                                             Row,
                                             Col,
                                             Row>(arg_parser, Row{}, Col{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    auto result     = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {   run_gemm_example<GemmConfigComputeV4>(arg_parser);
        run_gemm_example<GemmConfigComputeAsync>(arg_parser);
        return 0;
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        // Return a non-zero code to indicate failure
        return EXIT_FAILURE;
    }
}
