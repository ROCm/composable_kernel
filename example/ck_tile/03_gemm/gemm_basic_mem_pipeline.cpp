
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "gemm_basic.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "1", "batch size")
        .insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "4096", "k dimension")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename ALayout, typename BLayout, typename CLayout>
float gemm_calc(const gemm_basic_args& args, const ck_tile::stream_config& s)
{
    // ToDo: This will be modified by the codegen code later.
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 128;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;

    // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadA = true;
    constexpr bool kPadB = true;
    constexpr bool kPadC = true;

    constexpr int kBlockPerCu = 1;

    // ===============================================

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmTilePartitioner<GemmShape>;

    using GemmEpilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, false, kPadC>>;

    using BaseGemmPipeline =
        ck_tile::BaseGemmPipelineAgBgCrMem<ck_tile::BlockGemmPipelineProblem<ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             GemmShape,
                                                                             ALayout,
                                                                             BLayout,
                                                                             CLayout>>;

    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(args.K);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  CDataType,
                                                  GemmShape,
                                                  ALayout,
                                                  BLayout,
                                                  CLayout,
                                                  kPadA,
                                                  kPadB,
                                                  kPadC,
                                                  ck_tile::GemmPipelineScheduler::Intrawave,
                                                  has_hot_loop_v,
                                                  tail_number_v>>;
        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKargs(args.p_a,
                                       args.p_b,
                                       args.p_c,
                                       args.M,
                                       args.N,
                                       args.K,
                                       args.stride_A,
                                       args.stride_B,
                                       args.stride_C);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.kbatch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(s.log_level_ > 0)
        {
            std::cout << "Lunching kernel with args:"
                      << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        return ave_time;
    };

    if(has_hot_loop)
    {
        // Tail pipeline One to Seven
        if(tail_num == ck_tile::TailNumber::One)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
        }
        else if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }

        if constexpr(BaseGemmPipeline::PrefetchStages > 2)
        {
            if(tail_num == ck_tile::TailNumber::Two)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 3)
        {
            if(tail_num == ck_tile::TailNumber::Three)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 4)
        {
            if(tail_num == ck_tile::TailNumber::Four)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 5)
        {
            if(tail_num == ck_tile::TailNumber::Five)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 6)
        {
            if(tail_num == ck_tile::TailNumber::Six)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{});
            }
        }
        if constexpr(BaseGemmPipeline::PrefetchStages > 7)
        {
            if(tail_num == ck_tile::TailNumber::Seven)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});
            }
        }
    }
    else
    {
        // Tail number always 1
        if(tail_num == ck_tile::TailNumber::One)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
        }
    }

    return ave_time;
}

template <typename ALayout, typename BLayout, typename CLayout>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat)
{
    gemm_basic_args args;
    args.p_a      = a_m_k_dev_buf.GetDeviceBuffer();
    args.p_b      = b_k_n_dev_buf.GetDeviceBuffer();
    args.p_c      = c_m_n_dev_buf.GetDeviceBuffer();
    args.kbatch   = kbatch;
    args.M        = M;
    args.N        = N;
    args.K        = K;
    args.stride_A = stride_A;
    args.stride_B = stride_B;
    args.stride_C = stride_C;

    float ave_time = gemm_calc<ALayout, BLayout, CLayout>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::string op_name{"Gemm{MemBoundPipeline}"};

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run " << op_name << "kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideB =" << stride_B << " StrideC =" << stride_C
              << " : " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    return ave_time;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t batch_size = arg_parser.get_int("b");
    int n_warmup                = arg_parser.get_int("warmup");
    int n_repeat                = arg_parser.get_int("repeat");

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using namespace ck_tile::literals;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride = [](std::size_t row,
                                   std::size_t col,
                                   std::size_t stride,
                                   auto layout) {
        if(stride == 0)
        {
            // give a chance if stride is zero, return a default packed stride
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return col;
            }
            else
            {
                return row;
            }
        }
        else
            return stride;
    };

    stride_A = f_get_default_stride(M, K, stride_A, ALayout{});
    stride_B = f_get_default_stride(K, N, stride_B, BLayout{});
    stride_C = f_get_default_stride(M, N, stride_C, CLayout{});

    ck_tile::HostTensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, stride_A, ALayout{}));
    ck_tile::HostTensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, stride_B, BLayout{}));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

    // TODO: add different init types

    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    invoke_gemm<ALayout, BLayout, CLayout>(a_m_k_dev_buf,
                                           b_k_n_dev_buf,
                                           c_m_n_dev_buf,
                                           M,
                                           N,
                                           K,
                                           stride_A,
                                           stride_B,
                                           stride_C,
                                           batch_size,
                                           n_warmup,
                                           n_repeat);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);

        pass = ck_tile::check_err(c_m_n_dev_result, c_m_n_host_ref);

        std::cout << "The CPU veification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        ck_tile::HostTensor<CDataType> c_m_n_gpu_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_gpu_ref.get_element_space_size_in_bytes());
        c_m_n_gpu_ref.SetZero();
        c_m_n_gpu_buf_ref.SetZero();

        ck_tile::reference_gemm_gpu<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k_dev_buf, b_k_n_dev_buf, c_m_n_gpu_buf_ref, M, N, K, stride_A, stride_B, stride_C);

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_gpu_ref.data());
        pass = ck_tile::check_err(c_m_n_dev_result, c_m_n_gpu_ref);

        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}
