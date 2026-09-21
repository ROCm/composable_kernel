// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "universal_gemm_invoker.hpp"

template <template <typename PreType> typename GemmConfig>
int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    using Invoker = UniversalInvoker;

    // Validate mixed precision combinations
    if(data_type == "fp8fp4")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          Invoker,
                                          ck_tile::fp8_t,
                                          ck_tile::pk_fp4_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf8fp4")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          Invoker,
                                          ck_tile::bf8_t,
                                          ck_tile::pk_fp4_t,
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
    {
        std::cerr << "Failed to parse arguments\n";
        return -1;
    }

#if CK_TILE_USE_WMMA
    try
    {
        return !run_gemm_example<GemmConfigMixedPrec_Wmma>(arg_parser);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
#else
    // TODO: Add Mixed Prec Support for MFMA
    return EXIT_FAILURE;
#endif
}
