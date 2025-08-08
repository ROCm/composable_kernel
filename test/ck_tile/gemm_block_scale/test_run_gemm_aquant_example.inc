// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstring>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <random>

#include "ck_tile/core/config.hpp"
#include "ck_tile/host.hpp"
#include "test_gemm_aquant_utils.hpp"

template <typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ComputeDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          uint32_t QuantGroupSize>
float gemm_calc_aquant(const ck_tile::AQuantGemmHostArgs& args, const ck_tile::stream_config& s)
{
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 1;

    static_assert(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>);

    constexpr ck_tile::index_t M_Tile = 16;
    constexpr ck_tile::index_t N_Tile = 64;
    constexpr ck_tile::index_t K_Tile = 256;

    constexpr ck_tile::index_t M_Warp = 1;
    constexpr ck_tile::index_t N_Warp = 4;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 16;
    constexpr ck_tile::index_t N_Warp_Tile = 16;
    constexpr ck_tile::index_t K_Warp_Tile = 32;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    using CodegenGemmTraits =
        ck_tile::TileGemmAQuantTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                 BDataType,
                                                                 AccDataType,
                                                                 CodegenGemmShape,
                                                                 CodegenGemmTraits,
                                                                 ComputeDataType>;

    using BaseGemmPipeline = ck_tile::BaseAQuantGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

    const ck_tile::index_t K_split      = (args.K + K_Tile - 1) / K_Tile * K_Tile;
    const ck_tile::index_t num_loop     = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop             = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num  = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    constexpr bool transposed_warp_gemm = false;

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;

        using CodegenPipelineProblem =
            ck_tile::GemmAQuantPipelineProblem<ADataType,
                                               AQDataType,
                                               BDataType,
                                               AccDataType,
                                               CodegenGemmShape,
                                               CodegenGemmTraits,
                                               QuantGroupSize,
                                               ComputeDataType,
                                               ck_tile::GemmPipelineScheduler::Intrawave,
                                               has_hot_loop_v,
                                               tail_number_v>;
        using CodegenGemmPipeline = ck_tile::AQuantGemmPipelineAgBgCrCompV3<CodegenPipelineProblem>;
        using GemmEpilogue        = ck_tile::CShuffleEpilogue<
                   ck_tile::CShuffleEpilogueProblem<ADataType,
                                                    BDataType,
                                                    ck_tile::tuple<>,
                                                    AccDataType,
                                                    CDataType,
                                                    ck_tile::tuple<>,
                                                    CLayout,
                                                    ck_tile::element_wise::PassThrough,
                                                    CodegenPipelineProblem::kBlockSize,
                                                    TilePartitioner::MPerBlock,
                                                    TilePartitioner::NPerBlock,
                                                    M_Warp,
                                                    N_Warp,
                                                    M_Warp_Tile,
                                                    N_Warp_Tile,
                                                    K_Warp_Tile,
                                                    transposed_warp_gemm,
                                                    ck_tile::memory_operation_enum::set>>;
        using Kernel =
            ck_tile::AQuantGemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(args.k_batch != 1)
        {
            throw std::runtime_error("split-k is not supported yet!");
        }

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << CodegenGemmShape::GetName() << '\n'
                      << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                      << "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    };
    return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
}

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename CLayout,
          uint32_t QuantGroupSize>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem& aq_m_aqk_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t AQK,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_AQ,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat)
{
    ck_tile::AQuantGemmHostArgs args;
    args.a_ptr     = a_m_k_dev_buf.GetDeviceBuffer();
    args.aq_ptr    = aq_m_aqk_dev_buf.GetDeviceBuffer();
    args.b_ptr     = b_k_n_dev_buf.GetDeviceBuffer();
    args.c_ptr     = c_m_n_dev_buf.GetDeviceBuffer();
    args.k_batch   = kbatch;
    args.M         = M;
    args.N         = N;
    args.K         = K;
    args.QK        = AQK;
    args.stride_A  = stride_A;
    args.stride_B  = stride_B;
    args.stride_C  = stride_C;
    args.stride_AQ = stride_AQ;

    float ave_time = gemm_calc_aquant<ADataType,
                                      AQDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      BDataType,
                                      ALayout,
                                      BLayout,
                                      CLayout,
                                      QuantGroupSize>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::size_t flop     = std::size_t(2) * M * N * K;
    std::size_t num_byte = sizeof(ADataType) * M * K + sizeof(AQDataType) * M * AQK +
                           sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideAQ =" << stride_AQ << " StrideB =" << stride_B
              << " StrideC =" << stride_C << " A_Layout =" << ALayout::name
              << " B_Layout =" << BLayout::name << " C_Layout =" << CLayout::name
              << " A_Type = " << DataTypeTraits<ADataType>::name
              << " AQ_Type = " << DataTypeTraits<AQDataType>::name
              << " B_Type = " << DataTypeTraits<BDataType>::name
              << " Acc_Type = " << DataTypeTraits<AccDataType>::name
              << " C_Type = " << DataTypeTraits<CDataType>::name << " : " << ave_time << " ms, "
              << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

template <typename TypeConfig,
          uint32_t QuantGroupSize,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename CLayout>
bool run_gemm_test_with_layouts(int argc,
                                char* argv[],
                                const ALayout a_layout                  = ALayout{},
                                const AQLayout aq_layout                = AQLayout{},
                                const BLayout b_layout                  = BLayout{},
                                [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return false;

    using ADataType   = typename TypeConfig::ADataType;
    using AQDataType  = typename TypeConfig::QDataType;
    using BDataType   = typename TypeConfig::BDataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using CDataType   = typename TypeConfig::CDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    if(K % QuantGroupSize != 0)
    {
        throw std::runtime_error("K must be aligned with QuantGroupSize");
    }

    ck_tile::index_t AQK = K / QuantGroupSize;

    ck_tile::index_t stride_A  = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_AQ = arg_parser.get_int("stride_q");
    ck_tile::index_t stride_B  = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C  = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");

    stride_A  = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_AQ = ck_tile::get_default_stride(M, AQK, stride_AQ, is_row_major(aq_layout));
    stride_B  = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C  = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<AQDataType> aq_m_aqk(
        ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, is_row_major(aq_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint32_t> fill_seed(0, 500);

    if(init_method == 0)
    {
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            ck_tile::FillUniformDistribution<ck_tile::pk_int4_t>{-5.0f, 5.0f, fill_seed(gen)}(
                a_m_k);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f, fill_seed(gen)}(a_m_k);
        }
        ck_tile::FillUniformDistribution<AQDataType>{-2.0f, 2.0f, fill_seed(gen)}(aq_m_aqk);
        ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f, fill_seed(gen)}(b_k_n);
    }
    else if(init_method == 1)
    {
        std::cout << "Monotonic initialization is not supported." << std::endl;
        return true;
    }
    else if(init_method == 2)
    {
        ck_tile::FillConstant<ADataType>{static_cast<ADataType>(0x22)}(a_m_k);
        ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(0.5f)}(aq_m_aqk);
        ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x38)}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        aq_m_aqk.SetZero();
        b_k_n.SetZero();
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem aq_m_aqk_dev_buf(aq_m_aqk.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    invoke_gemm<ADataType,
                AQDataType,
                BDataType,
                AccDataType,
                CDataType,
                ALayout,
                AQLayout,
                BLayout,
                CLayout,
                QuantGroupSize>(a_m_k_dev_buf,
                                aq_m_aqk_dev_buf,
                                b_k_n_dev_buf,
                                c_m_n_dev_buf,
                                M,
                                N,
                                K,
                                AQK,
                                stride_A,
                                stride_AQ,
                                stride_B,
                                stride_C,
                                kbatch,
                                n_warmup,
                                n_repeat);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm_quant<ADataType,
                                      AQDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      QuantGroupSize,
                                      true>(a_m_k, aq_m_aqk, b_k_n, c_m_n_host_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        if(!pass)
        {
            std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "CPU verification " << (pass ? "Passed!" : "Failed ...") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        std::cout << "GPU verification is not implemented yet. Re-run with -v=1" << std::endl;
        return false;
    }

    return pass;
}

template <typename TypeConfig, uint32_t QuantGroupSize>
bool run_gemm_test_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if constexpr(std::is_same_v<typename TypeConfig::ADataType, ck_tile::pk_int4_t> ||
                 std::is_same_v<typename TypeConfig::ADataType, ck_tile::fp8_t> ||
                 std::is_same_v<typename TypeConfig::ADataType, ck_tile::bf8_t>)
    {
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_test_with_layouts<TypeConfig, QuantGroupSize>(
                argc, argv, Row{}, Row{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for A.");
    }

    return true;
}

bool run_gemm_test(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return false;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>{});
        return run_gemm_test_prec_type<TypeConfig, 128>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "bf8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, float>{});
        return run_gemm_test_prec_type<TypeConfig, 128>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i4fp8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t,
                                                        ck_tile::fp8_t,
                                                        float,
                                                        ck_tile::fp8_t>{});
        return run_gemm_test_prec_type<TypeConfig, 128>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i4bf8")
    {
        using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t,
                                                        ck_tile::bf8_t,
                                                        float,
                                                        ck_tile::bf8_t>{});
        return run_gemm_test_prec_type<TypeConfig, 128>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i4f32fp8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t, ck_tile::fp8_t, float, float>{});
        return run_gemm_test_prec_type<TypeConfig, 128>(a_layout, b_layout, argc, argv);
    }
    else if(data_type == "i4f32bf8")
    {
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::pk_int4_t, ck_tile::bf8_t, float, float>{});
        return run_gemm_test_prec_type<TypeConfig, 128>(a_layout, b_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int run_gemm_combinations(std::string const& data_type)
{
    // Define possible values for each parameter
    std::vector<std::vector<std::string>> mnk_values = {{
                                                            "1",
                                                            "2048",
                                                            "5120",
                                                        },
                                                        {
                                                            "2",
                                                            "2048",
                                                            "5120",
                                                        },
                                                        {
                                                            "16",
                                                            "2048",
                                                            "5120",
                                                        },
                                                        {
                                                            "17",
                                                            "2048",
                                                            "5120",
                                                        },
                                                        {
                                                            "2047",
                                                            "5120",
                                                            "1024",
                                                        },
                                                        {
                                                            "2048",
                                                            "5120",
                                                            "1024",
                                                        }};
    std::vector<std::string> prec_values             = {data_type};

    // We'll store all our arguments as strings first
    std::vector<std::string> arg_strings = {"test_tile_gemm_aquant_basic",
                                            "", // m placeholder
                                            "", // n placeholder
                                            "", // k placeholder
                                            "", // prec placeholder
                                            "-init=0",
                                            "-v=1",
                                            "-warmup=0",
                                            "-repeat=1"};

    // Create an array of const char pointers for argv
    constexpr size_t ARG_COUNT   = 9;
    constexpr size_t ARG_MAX_LEN = 64;
    char args[ARG_COUNT][ARG_MAX_LEN];
    char* argv[ARG_COUNT];

    // Run all combinations
    bool is_success = true;
    for(const auto& mnk : mnk_values)
    {
        arg_strings[1] = "-m=" + mnk[0];
        arg_strings[2] = "-n=" + mnk[1];
        arg_strings[3] = "-k=" + mnk[2];

        for(const auto& prec : prec_values)
        {
            arg_strings[4] = "-prec=" + prec;

            // Set up the argv array with pointers to the string data
            for(size_t i = 0; i < ARG_COUNT; i++)
            {
                strncpy(args[i], arg_strings[i].c_str(), ARG_MAX_LEN);
                argv[i] = args[i];
            }

            std::cout << "Arguments received: ";
            for(size_t i = 1; i < ARG_COUNT; ++i)
            {
                std::cout << argv[i] << " ";
            }
            std::cout << std::endl;

            // Call the function with the current configuration
            try
            {
                is_success = run_gemm_test(ARG_COUNT, argv) && is_success;
            }
            catch(const ArgumentsNotSupportedException& e)
            {
                std::cerr << "Caught ArgumentsNotSupportedException: " << e.what() << '\n';
                // ArgumentsNotSupportedException  is not an error. Do not change is_success
            }
            catch(const std::runtime_error& e)
            {
                std::cerr << "Caught runtime error: " << e.what() << '\n';
                is_success = false;
            }
        }
    }
    return is_success ? EXIT_SUCCESS : EXIT_FAILURE;
}
