// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "gemm_weight_preshuffle_invoker.hpp"

template <typename GemmConfig,
          typename APrecType,
          typename BPrecType       = APrecType,
          typename CPrecType       = APrecType,
          typename ComputeDataType = APrecType>
int run_gemm_example_prec_type(std::string a_layout,
                               std::string b_layout,
                               ck_tile::ArgParser& arg_parser)
{
    using Row       = ck_tile::tensor_layout::gemm::RowMajor;
    using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
    bool preshuffle = GemmConfig::Preshuffle;
    using Invoker   = WeightPreshuffleInvoker;

    if(preshuffle && (a_layout != "R" || b_layout != "C"))
    {
        throw std::runtime_error(
            "Preshuffle is supported only for A(Row major), B(column major) input matrices!");
    }

    if(a_layout == "R" && b_layout == "C")
    {
        return run_gemm_example_with_layouts<GemmConfig,
                                             Invoker,
                                             APrecType,
                                             BPrecType,
                                             CPrecType,
                                             Row,
                                             Col,
                                             Row,
                                             ComputeDataType>(arg_parser, Row{}, Col{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported memory layout for the input matrices!");
    }
}

template <template <typename PreType> typename GemmConfig>
int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, ck_tile::half_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser);
    }
#ifdef CK_GFX950_SUPPORT
    else if(data_type == "tf32")
    {
        // TF32 uses template-specialized GemmConfig with correct tile config
        return run_gemm_example_prec_type<GemmConfig<ck_tile::tf32_t>,
                                          float,
                                          float,
                                          float,
                                          ck_tile::tf32_t>(a_layout, b_layout, arg_parser);
    }
#endif
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          ck_tile::fp8_t,
                                          ck_tile::fp8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          ck_tile::bf8_t,
                                          ck_tile::bf8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "int4")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          ck_tile::fp8_t,
                                          ck_tile::pk_int4_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
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
    {
#if CK_TILE_USE_WMMA
        return !run_gemm_example<GemmConfigPreshufflePrefill_Wmma>(arg_parser);
#else
        return !run_gemm_example<GemmConfigPreshufflePrefill>(arg_parser);
#endif
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
