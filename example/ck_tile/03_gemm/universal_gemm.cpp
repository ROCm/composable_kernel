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

// Universal GEMM-specific wrapper that handles test_async flag
template <typename GemmConfig,
          typename ADataType,
          typename BDataType = ADataType,
          typename CDataType = ADataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_gemm_example_with_layouts_universal(ck_tile::ArgParser& arg_parser,
                                            const ALayout a_layout = ALayout{},
                                            const BLayout b_layout = BLayout{},
                                            const CLayout c_layout = CLayout{})
{
    using Invoker     = UniversalInvoker;
    using AccDataType = typename GemmTypeConfig<ADataType, BDataType, CDataType>::AccDataType;

    // Check for async input scheduler test mode
    bool test_async = arg_parser.get_int("test_async");
    if(test_async)
    {
        // Extract parameters for async test (same as shared implementation)
        const ck_tile::index_t M      = arg_parser.get_int("m");
        const ck_tile::index_t N      = arg_parser.get_int("n");
        const ck_tile::index_t K      = arg_parser.get_int("k");
        const ck_tile::index_t kbatch = arg_parser.get_int("split_k");

        using Row                     = ck_tile::tensor_layout::gemm::RowMajor;
        constexpr bool is_a_row_major = std::is_same_v<ALayout, Row>;
        constexpr bool is_b_row_major = std::is_same_v<BLayout, Row>;
        constexpr bool is_c_row_major = std::is_same_v<CLayout, Row>;

        const ck_tile::index_t stride_A = is_a_row_major ? K : M;
        const ck_tile::index_t stride_B = is_b_row_major ? N : K;
        const ck_tile::index_t stride_C = is_c_row_major ? N : M;

        // Allocate and initialize tensors
        ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
            M, K, stride_A, ck_tile::bool_constant<is_a_row_major>{}));
        ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
            K, N, stride_B, ck_tile::bool_constant<is_b_row_major>{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(ck_tile::host_tensor_descriptor(
            M, N, stride_C, ck_tile::bool_constant<is_c_row_major>{}));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::GemmHostArgs args = {a_m_k_dev_buf.GetDeviceBuffer(),
                                      b_k_n_dev_buf.GetDeviceBuffer(),
                                      c_m_n_dev_buf.GetDeviceBuffer(),
                                      kbatch,
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      stride_C};

        Invoker::template test_async_input_scheduler<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     ck_tile::tuple<>,
                                                     AccDataType,
                                                     CDataType,
                                                     ALayout,
                                                     BLayout,
                                                     ck_tile::tuple<>,
                                                     CLayout,
                                                     ck_tile::element_wise::PassThrough>(
            args, ck_tile::stream_config{nullptr, false, 1});

        // Copy result from device for verification
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

        // Compute CPU reference
        ck_tile::HostTensor<CDataType> c_m_n_ref(ck_tile::host_tensor_descriptor(
            M, N, stride_C, ck_tile::bool_constant<is_c_row_major>{}));
        c_m_n_ref.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_ref);

        // Verify results
        const float max_accumulated_value =
            *std::max_element(c_m_n_ref.mData.begin(), c_m_n_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        bool pass = do_verify(c_m_n_dev_result, c_m_n_ref, rtol_atol, "CPU");

        std::cout << "Async input scheduler test: " << (pass ? "PASS" : "FAIL") << std::endl;
        return pass;
    }

    // Normal path - delegate to shared implementation
    return run_gemm_example_with_layouts<GemmConfig, Invoker, ADataType, BDataType, CDataType>(
        arg_parser, a_layout, b_layout, c_layout);
}

// Universal GEMM-specific prec_type dispatcher that uses the wrapper
template <typename GemmConfig,
          typename APrecType,
          typename BPrecType = APrecType,
          typename CPrecType = APrecType>
int run_gemm_example_prec_type_universal(std::string a_layout,
                                         std::string b_layout,
                                         ck_tile::ArgParser& arg_parser)
{
    using Row       = ck_tile::tensor_layout::gemm::RowMajor;
    using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
    bool preshuffle = GemmConfig::Preshuffle;

    if(preshuffle && std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    {
        throw std::runtime_error("Preshuffle is not supported for this int4 datatype!");
    }

    if(preshuffle && a_layout != "R" && b_layout != "C")
    {
        throw std::runtime_error(
            "Preshuffle is supported only for A(Row major), B(column major) input matrices!");
    }

    using LayoutVariant = std::variant<Row, Col>;

    auto string_to_layout = [](const std::string& layout) -> LayoutVariant {
        if(layout == "R")
            return Row{};
        if(layout == "C")
            return Col{};
        throw std::runtime_error("Unsupported layout: " + layout);
    };

    auto a_layout_variant = string_to_layout(a_layout);
    auto b_layout_variant = string_to_layout(b_layout);

    return std::visit(
        [&](auto a_layout_type, auto b_layout_type) -> int {
            if constexpr(std::is_same_v<BPrecType, ck_tile::pk_int4_t> &&
                         std::is_same_v<decltype(b_layout_type), Row>)
            {
                throw std::runtime_error("Unsupported memory layout for the input matrices when "
                                         "BPrecType is ck_tile::pk_int4_t!");
            }
            else
            {
                return run_gemm_example_with_layouts_universal<GemmConfig,
                                                               APrecType,
                                                               BPrecType,
                                                               CPrecType>(
                    arg_parser, a_layout_type, b_layout_type, Row{});
            }
        },
        a_layout_variant,
        b_layout_variant);
}

template <template <typename PrecType> typename GemmConfig>
int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type_universal<GemmConfig<ck_tile::half_t>, ck_tile::half_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type_universal<GemmConfig<ck_tile::bf16_t>, ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type_universal<GemmConfig<ck_tile::fp8_t>,
                                                    ck_tile::fp8_t,
                                                    ck_tile::fp8_t,
                                                    ck_tile::half_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_example_prec_type_universal<GemmConfig<ck_tile::bf8_t>,
                                                    ck_tile::bf8_t,
                                                    ck_tile::bf8_t,
                                                    ck_tile::half_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "int8")
    {
        return run_gemm_example_prec_type_universal<GemmConfig<ck_tile::int8_t>,
                                                    ck_tile::int8_t,
                                                    ck_tile::int8_t,
                                                    ck_tile::int32_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "fp16i4")
    {
        // TODO: Add support for bhalf_t ADataType
        if constexpr(GemmConfig<ck_tile::half_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
        {
            return run_gemm_example_prec_type_universal<GemmConfig<ck_tile::half_t>,
                                                        ck_tile::half_t,
                                                        ck_tile::pk_int4_t,
                                                        ck_tile::half_t>(
                a_layout, b_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported pipeline for this operation !!!");
        }
    }
    else if(data_type == "fp8i4")
    {
        if constexpr(GemmConfig<ck_tile::fp8_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
        {
            return run_gemm_example_prec_type_universal<GemmConfig<ck_tile::fp8_t>,
                                                        ck_tile::fp8_t,
                                                        ck_tile::pk_int4_t,
                                                        ck_tile::half_t>(
                a_layout, b_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported pipeline for this operation !!!");
        }
    }
    else if(data_type == "bf8i4")
    {
        if constexpr(GemmConfig<ck_tile::bf8_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
        {
            return run_gemm_example_prec_type_universal<GemmConfig<ck_tile::bf8_t>,
                                                        ck_tile::bf8_t,
                                                        ck_tile::pk_int4_t,
                                                        ck_tile::half_t>(
                a_layout, b_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported pipeline for this operation !!!");
        }
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
        return !run_gemm_example<GemmConfigComputeV3_WMMA>(arg_parser);
#else
        return !run_gemm_example<GemmConfigComputeV3_2>(arg_parser);
#endif
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        // Return a non-zero code to indicate failure
        return EXIT_FAILURE;
    }
}
